"""When to spend an OCR call on a plate crop.

Separated from ``FrameProcessor`` because it is the one part of the cascade
worth reasoning about in isolation: it is pure, it holds no state, it touches
no model and no I/O, and every threshold in it is a site-calibration knob.
Taking an explicit context record rather than reaching into pipeline state is
what makes it testable without a frame, a model or a track.

The policy this encodes, for a FIXED gate camera:

    plate too small    -> wait   (the vehicle is still approaching)
    plate readable     -> consider, on a cadence set by how good it is
    excellent frame    -> run now, ahead of the other tracks in this frame
    best view passed   -> spend what is left of the budget
    budget exhausted   -> stop, and let the early lock or the validator decide

Why a size floor is not enough on its own. At a fixed 3-4 ft gate mount a
plate crosses the readable-size threshold several seconds before it reaches
its best view, and the OCR calls spent during that approach are the least
useful ones available — smallest, most skewed, most motion-blurred. Waiting
while the plate is still growing costs nothing (the vehicle is coming to us,
and at a gate it is about to stop) and moves the same budget onto frames where
the plate is bigger, sharper and more frontal.

NO BACKLOG, BY CONSTRUCTION
---------------------------
This module returns a verdict about the frame in hand and nothing else. There
is no queue, no deferred work item and no "recognize this later" path anywhere
in the cascade: an OCR call either happens synchronously on the frame that
produced the crop, or it never happens. The top-K observations a track retains
are kept for the *event image* and for the approach/departure signal, and are
never revisited by the recognizer.

That is a deliberate constraint, not an oversight. A queue would decouple
recognition from capture, and the moment the pipeline fell behind it would
start reading plates from frames that no longer describe where the vehicle is
— which is precisely the failure ``RtspReader``'s latest-frame policy exists to
prevent one layer down.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --- default derivations -----------------------------------------------------
#
# Every default below is derived from a measurement that already exists in the
# codebase rather than picked to look reasonable. The derivations are spelled
# out so a site engineer can redo them against their own camera instead of
# trusting ours.
#
# ``plate_quality`` defines MIN_USABLE_WIDTH_PX = 45 (character strokes below
# this are smaller than sensor noise) and GOOD_WIDTH_PX = 110 (full marks on
# the resolution term). DEFAULT_MIN_PLATE_WIDTH_READ sits between them, biased
# toward the low end: the cost of reading a marginal plate is one cheap OCR
# call, while the cost of refusing one is a missed vehicle.
DEFAULT_MIN_PLATE_WIDTH_READ = 70

# The quality score is 0.40*resolution + 0.30*sharpness + 0.15*aspect +
# 0.15*exposure. Working backwards from that formula:
#
#   "good"      0.55 ~ resolution 0.60 (about 84 px wide), moderate sharpness,
#                      sane aspect and exposure. Readable, not ideal.
#   "excellent" 0.75 ~ resolution 1.00 (>= 110 px), genuinely sharp, sane
#                      aspect and exposure. This is the frame we were waiting
#                      for, and at a gate it is the one where the vehicle has
#                      stopped at the boom.
DEFAULT_GOOD_QUALITY = 0.55
DEFAULT_EXCELLENT_QUALITY = 0.75


#: Wait reasons that mean "nothing about this track will change in the next
#: frame or two". While one of these stands, the plate DETECTOR backs off as
#: well, because the crop it would produce is one we have already decided not
#: to read. Cadence waits are excluded — those are already short — and so is
#: the per-frame cap, where the track lost a contest it should re-enter
#: immediately with a fresh crop.
BACKOFF_REASONS = frozenset({"plate_too_small", "still_approaching", "holding_reserve"})


class OcrAction(str, Enum):
    #: Spend a call now.
    RUN = "run"
    #: Do not spend a call, but expect a better frame from this track shortly.
    WAIT = "wait"
    #: Do not spend a call, and do not expect this track to improve.
    SKIP = "skip"


@dataclass(frozen=True)
class OcrVerdict:
    action: OcrAction
    #: Short machine-stable string, surfaced in stats and asserted in tests.
    reason: str
    #: Higher wins when more tracks want an OCR call than the per-frame cap
    #: allows. Quality, because that is what the call converts into accuracy.
    priority: float = 0.0

    @property
    def run(self) -> bool:
        return self.action is OcrAction.RUN


@dataclass(frozen=True)
class OcrPolicy:
    """Thresholds for the OCR scheduler. All configurable; see
    ``configs/default.yaml`` under ``pipeline.ocr`` for the documented copy.
    """

    # -- readable size, kept separate from detectable size ------------------
    #: Minimum plate width in pixels before recognition is even considered.
    #: The plate DETECTOR keeps its own, much lower, floor (models.yaml
    #: ``plate_detector.min_width_px``, default 24): we want to know a plate is
    #: there, and how big it is, long before we are willing to read it. That
    #: separation is what makes "wait for the vehicle to come closer" possible
    #: — you cannot wait for something you have not detected.
    min_plate_width_read: int = DEFAULT_MIN_PLATE_WIDTH_READ

    # -- quality tiers ------------------------------------------------------
    #: At or above this a crop is worth a call on the normal cadence.
    good_quality: float = DEFAULT_GOOD_QUALITY
    #: At or above this a crop pre-empts the cadence and the other tracks.
    excellent_quality: float = DEFAULT_EXCELLENT_QUALITY

    # -- budget -------------------------------------------------------------
    #: Total OCR calls one track may ever consume.
    #:
    #: Six, not twelve. Twelve was sized for "about four seconds of
    #: eligibility" when every eligible frame was read; with diversity-aware
    #: retention and correlated-evidence discounting, most of those extra
    #: reads were near-duplicates that no longer count as independent
    #: observations anyway — so they cost a recognizer call each and bought
    #: almost nothing.
    #:
    #: Six is the floor that still clears the validator's ``min_reads`` of 3
    #: with margin: two reads can fail the fragment or weight filters and a
    #: plate is still decidable. Below six that margin disappears and a
    #: filtered read means no plate at all.
    budget_per_track: int = 6
    #: Calls held back from the approach phase so the departure frames can
    #: always be read. Without a reserve a vehicle that dawdles on the way in
    #: spends its whole budget before it reaches the boom.
    reserve: int = 2
    #: Extra calls granted ONCE to a track whose plate is one or two
    #: characters from settled when the budget runs out.
    #:
    #: The budget is a blunt instrument: it stops at 12 calls whether the
    #: plate is settled, hopeless, or a single ambiguous character away from
    #: certain. The last case is worth more evidence and the other two are
    #: not, and the fused per-character posteriors can tell them apart — so
    #: this spends a little more only where it changes the answer. Two on a
    #: base budget of six: enough to move an ambiguous position's posterior,
    #: small enough that a plate the model simply cannot read does not inflate
    #: its cost. Set 0 to disable.
    retry_bonus: int = 2
    #: The most unstable positions a retry is worth granting for. Above this
    #: the plate is not nearly-settled, it is unread, and one more look will
    #: not fix it.
    retry_max_unstable: int = 2
    #: Reads needed before the retry check is meaningful.
    retry_min_reads: int = 3

    #: Ceiling on OCR calls across ALL tracks in a single frame. A controlled
    #: single-lane gate can use 1; 2 is the default so a motorcycle beside a
    #: car does not starve. This is the hard guarantee that a queue of five
    #: vehicles cannot turn one frame into five recognizer calls.
    max_per_frame: int = 2

    # -- adaptive cadence, in processed frames ------------------------------
    #: Excellent frames are rate-limited only enough to avoid reading the same
    #: instant twice; consecutive excellent frames are exactly what the
    #: validator wants.
    interval_excellent: int = 1
    interval_good: int = 2
    #: Marginal crops still vote, but they should not crowd out better ones,
    #: so they are sampled about once a second at the default frame rate.
    interval_marginal: int = 5

    # -- approach / departure -----------------------------------------------
    #: How many observations back to measure growth over. Three observations
    #: is about one second of approach at the default cadence.
    growth_window: int = 3
    #: Growth over that window that counts as "still approaching". 8% is
    #: comfortably above the few-percent jitter of a stationary vehicle's box
    #: and well below the growth of one actually driving in.
    growth_ratio: float = 1.08
    #: Shrinkage from the peak width that counts as "the best view has
    #: passed". 15% down from peak is unambiguous departure, not jitter.
    departure_ratio: float = 0.85
    #: Consecutive waits after which the scheduler reads anyway. The safety
    #: valve on the approach logic: a vehicle creeping in over a long lane
    #: must not be waited on forever. Four waits is at most ~1.3 s.
    max_consecutive_waits: int = 4

    #: Multiplier on ``plate_interval`` while a track is in a wait we already
    #: understand — too small, still approaching, holding the reserve.
    #:
    #: Without this the scheduler saves the recognizer call but keeps paying
    #: for the plate-detector call that produced the crop it declined, several
    #: times a second, for the whole approach. Having decided a vehicle is a
    #: second away from being worth reading, re-asking three times a second is
    #: the same mistake one stage earlier.
    #:
    #: 3 takes the check to about once a second at the default 6 fps /
    #: plate_interval 2 — fast enough to notice a plate becoming readable, or
    #: a vehicle starting to leave, well inside the cadence that follows. Set
    #: 1 to disable the backoff.
    wait_backoff: int = 3

    # -- staleness ----------------------------------------------------------
    #: Skip recognition on a frame older than this at the moment processing
    #: began. 400 ms is ~2.4 frame periods at the default 6 fps: enough slack
    #: for normal jitter, tight enough that a backed-up pipeline stops
    #: spending its most expensive call on a stale view. Set 0 to disable.
    max_frame_age_ms: float = 400.0

    # -- retention ----------------------------------------------------------
    #: Best-quality plate observations retained per track, for the event
    #: image, the approach signal and the targeted-retry pool. Not an OCR
    #: queue — see the module docstring.
    #:
    #: Six rather than five now that retention is diversity-aware and a retry
    #: draws from it: the set has to hold enough genuinely different views
    #: that an unread one is available when the retry fires.
    keep_observations: int = 6
    #: Appearance-hash distance (of 64) below which a new observation is
    #: treated as the same view as one already retained, and collapses into it
    #: instead of taking a second slot. See ai/quality/crop_hash.
    diversity_hamming: int = 6

    def __post_init__(self) -> None:
        if self.reserve >= self.budget_per_track:
            raise ValueError(
                f"ocr.reserve ({self.reserve}) must be below ocr.budget_per_track "
                f"({self.budget_per_track}), or no call is ever spent on the approach"
            )
        if self.good_quality > self.excellent_quality:
            raise ValueError("ocr.good_quality must not exceed ocr.excellent_quality")
        if self.max_per_frame < 1:
            raise ValueError("ocr.max_per_frame must be at least 1")


@dataclass(frozen=True)
class OcrContext:
    """Everything ``decide`` needs about one candidate, and nothing else."""

    #: Quality score of the crop in hand, 0..1, from ``score_plate``.
    quality: float
    #: Width of the crop in pixels.
    width: int
    #: The existing ``PipelineConfig.min_plate_quality`` floor. Passed in
    #: rather than duplicated in the policy so there is one such threshold.
    quality_floor: float
    #: Processed frames since this track last had an OCR call spent on it.
    frames_since_ocr: int
    #: Calls already spent on this track.
    ocr_spent: int
    #: Consecutive scheduler waits since the last call.
    consecutive_waits: int
    #: Largest plate width observed for this track so far, 0 if none.
    peak_width: int
    #: width_now / width_(growth_window observations ago); 1.0 when unknown.
    approach_growth: float = 1.0
    #: Frame was too old to be worth the most expensive call in the pipeline.
    stale: bool = False


def decide(ctx: OcrContext, policy: OcrPolicy) -> OcrVerdict:
    """Whether to spend an OCR call on this crop, now.

    The order of the rules is the policy. Each is commented with the situation
    at a gate that it exists for.
    """
    priority = ctx.quality

    # A stale frame no longer describes where the vehicle is. Never spend the
    # pipeline's most expensive call on one.
    if ctx.stale:
        return OcrVerdict(OcrAction.SKIP, "stale_frame", priority)

    # Detectable but not readable: the vehicle is still too far away. This is
    # the common case for most of a track's life and the reason the detector's
    # width floor and this one are different numbers.
    if ctx.width < policy.min_plate_width_read:
        return OcrVerdict(OcrAction.WAIT, "plate_too_small", priority)

    # Below the existing quality floor nothing legible comes out, whatever the
    # size says. FrameProcessor also checks this; kept here so the scheduler is
    # correct when called on its own.
    if ctx.quality < ctx.quality_floor:
        return OcrVerdict(OcrAction.SKIP, "below_quality_floor", priority)

    remaining = policy.budget_per_track - ctx.ocr_spent
    if remaining <= 0:
        return OcrVerdict(OcrAction.SKIP, "budget_exhausted", priority)

    excellent = ctx.quality >= policy.excellent_quality
    good = ctx.quality >= policy.good_quality
    departing = ctx.peak_width > 0 and ctx.width <= ctx.peak_width * policy.departure_ratio
    approaching = ctx.approach_growth >= policy.growth_ratio
    forced = ctx.consecutive_waits >= policy.max_consecutive_waits
    in_reserve = remaining <= policy.reserve

    # The frame we were waiting for. Pre-empts the cadence, the reserve and —
    # via `priority` — the other tracks competing for this frame's calls.
    if excellent and ctx.frames_since_ocr >= policy.interval_excellent:
        return OcrVerdict(OcrAction.RUN, "excellent_frame", priority)

    # Endgame. Either the vehicle is leaving and this view is as good as it
    # will get, or we have waited long enough and holding out is now the
    # bigger risk. Both release the reserve.
    if (departing or forced) and ctx.frames_since_ocr >= policy.interval_good:
        return OcrVerdict(
            OcrAction.RUN, "departing" if departing else "patience_exhausted", priority
        )

    # Reserve is for the endgame above and for an excellent frame. Anything
    # else waits.
    if in_reserve:
        return OcrVerdict(OcrAction.WAIT, "holding_reserve", priority)

    # Still approaching and not yet good: a materially better frame is about a
    # second away. This is the rule that moves budget from the approach to the
    # boom, and the main source of the CPU saving.
    if approaching and not good:
        return OcrVerdict(OcrAction.WAIT, "still_approaching", priority)

    if good and ctx.frames_since_ocr >= policy.interval_good:
        return OcrVerdict(OcrAction.RUN, "good_frame", priority)

    if ctx.frames_since_ocr >= policy.interval_marginal:
        return OcrVerdict(OcrAction.RUN, "marginal_cadence", priority)

    return OcrVerdict(OcrAction.WAIT, "cadence", priority)
