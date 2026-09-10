"""Per-track accumulator.

One of these exists for every vehicle the tracker is following. It collects
every plate read with the evidence needed to weigh it later, and keeps the
best vehicle and plate crops seen so far — "best" meaning highest quality, not
most recent, because the sharpest view of a plate is usually mid-approach
rather than at the moment the vehicle crosses the line.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ai.plate_recognizer import grammar_decode, postprocess
from ..ai.quality import crop_hash
from ..ai.quality.plate_quality import QualityBreakdown
from ..ai.types import PlateObservation, PlateRead, Track
from ..video.line_crossing import CrossDirection
from . import char_fusion


@dataclass
class WeightedRead:
    """One recognizer output plus everything needed to weight its vote."""

    text: str
    raw_text: str
    rec_confidence: float
    per_char_confidence: list[float]
    plate_det_confidence: float
    quality: float
    grammar: float
    frame_ts: float
    plate_crop: Optional[np.ndarray] = None
    corrections: list[str] = field(default_factory=list)
    #: Which processed frame produced this read, and what the crop looked
    #: like. Together they let fusion tell four independent looks at a plate
    #: from one look sampled four times — see char_fusion._independence_scale.
    frame_idx: int = 0
    appearance: int = 0

    @property
    def weight(self) -> float:
        """w = rec^1.5 * (0.5 + 0.5*det) * quality * grammar

        The exponent sharpens the recognizer term so a 0.9 read counts for
        appreciably more than a 0.6 one; the detector term never goes to zero,
        because a plate the detector was unsure about can still be read
        correctly.
        """
        return (
            (max(0.0, self.rec_confidence) ** 1.5)
            * (0.5 + 0.5 * max(0.0, min(1.0, self.plate_det_confidence)))
            * max(0.0, min(1.0, self.quality))
            * max(0.0, min(1.0, self.grammar))
        )

    def char_confidence(self, i: int) -> float:
        if 0 <= i < len(self.per_char_confidence):
            return self.per_char_confidence[i]
        return self.rec_confidence


@dataclass
class PlateHypothesisEvidence:
    """The best crops belonging to ONE candidate plate string.

    Exists because a track's images and its plate must not be chosen by
    independent criteria. The old code kept a single global best crop per
    track and picked the plate by a weighted vote, so when a track's identity
    drifted across vehicles the two selections could land on different ones —
    an event stating plate ``HR29BG7381`` while storing a photo of the car
    carrying ``UP14FU2031``, because that later frame happened to score a
    higher quality.

    Keyed per hypothesis, the images travel with the plate string they belong
    to. Whichever string wins validation, its own crops come with it.

    Plate and vehicle crops carry SEPARATE high-water marks: a retry read has
    no vehicle crop (it is recognized from a banked plate crop), and a
    hypothesis must not lose a good vehicle image to a later frame that had
    none. Both still come from frames that produced THIS plate string, which
    is the property that matters.
    """

    text: str
    plate_crop: Optional[np.ndarray] = None
    plate_quality: float = -1.0
    plate_frame_idx: int = 0
    plate_ts: float = 0.0
    vehicle_crop: Optional[np.ndarray] = None
    vehicle_quality: float = -1.0
    vehicle_frame_idx: int = 0
    vehicle_ts: float = 0.0
    #: Reads that produced this string, and the window they span. Reported in
    #: the audit trail so the image's provenance is inspectable.
    read_count: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0

    def consider(
        self,
        quality: float,
        frame_idx: int,
        ts: float,
        plate_crop: Optional[np.ndarray],
        vehicle_crop: Optional[np.ndarray],
    ) -> None:
        self.read_count += 1
        self.first_ts = ts if self.read_count == 1 else min(self.first_ts, ts)
        self.last_ts = max(self.last_ts, ts)
        if plate_crop is not None and quality > self.plate_quality:
            self.plate_crop = plate_crop
            self.plate_quality = quality
            self.plate_frame_idx = frame_idx
            self.plate_ts = ts
        if vehicle_crop is not None and quality > self.vehicle_quality:
            self.vehicle_crop = vehicle_crop
            self.vehicle_quality = quality
            self.vehicle_frame_idx = frame_idx
            self.vehicle_ts = ts

    def describe(self) -> str:
        return (
            f"{self.text}: {self.read_count} reads over "
            f"{max(0.0, self.last_ts - self.first_ts):.1f}s, "
            f"plate@frame{self.plate_frame_idx} vehicle@frame{self.vehicle_frame_idx}"
        )


#: Most plate hypotheses a track retains crops for. The OCR budget caps reads
#: at 6 (+2 retry), so a track rarely produces more distinct strings than
#: this; the cap exists so a pathological track cannot accumulate vehicle
#: crops without bound. Eviction drops the weakest by quality.
MAX_HYPOTHESES = 8


@dataclass
class TrackState:
    track_id: int
    camera_id: int
    vehicle_type: str = "vehicle"
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    reads: list[WeightedRead] = field(default_factory=list)
    detect_confidence: float = 0.0
    centroid_history: list[tuple[float, float]] = field(default_factory=list)

    best_vehicle_crop: Optional[np.ndarray] = None
    best_plate_crop: Optional[np.ndarray] = None
    #: High-water plate-crop quality, driven by every OBSERVATION.
    best_quality: float = 0.0
    #: High-water mark for the vehicle image specifically. See
    #: note_vehicle_crop for why it is not the same number.
    best_vehicle_quality: float = 0.0

    direction: CrossDirection = CrossDirection.NONE
    crossed_at: float = 0.0
    finalize_reason: str = ""
    disputed: bool = False
    #: Set once this track has produced its event. The state is KEPT in the
    #: processor after emitting so the still-live track cannot be re-created
    #: as a fresh state and cross the line again on the very next frame.
    emitted: bool = False
    #: Set once the plate is settled beyond reasonable doubt, at which point
    #: the plate stages stop running for this track entirely.
    locked: bool = False

    frames_since_plate_attempt: int = 999
    plate_attempts: int = 0

    # -- OCR budget and scheduling ----------------------------------------
    #
    # Deliberately separate counters from the plate-attempt ones above.
    # Localizing a plate and recognizing it differ by one to two orders of
    # magnitude in cost, so they get separate budgets and separate cadences;
    # collapsing them is what makes a pipeline spend its most expensive call
    # on its worst frames.
    ocr_calls: int = 0
    frames_since_ocr: int = 999
    consecutive_ocr_waits: int = 0
    #: Extra OCR calls granted once, because the plate was one or two
    #: characters from settled. See grant_retry.
    retry_allowance: int = 0
    retry_granted: bool = False
    #: Set once the finalize-time retry pass has run, so it can never run
    #: twice for one track however many times finalize is reached.
    retry_resolved: bool = False
    #: Which character positions the retry was bought to resolve.
    retry_positions: list[int] = field(default_factory=list)
    #: Why the scheduler last declined, for stats and for the preview overlay.
    last_ocr_reason: str = ""

    #: Best-quality plate observations, newest-first among equals. Bounded to
    #: the policy's ``keep_observations``. This is NOT an OCR work queue —
    #: nothing recognizes an observation after its frame (see
    #: ``video/ocr_scheduler``); it exists for the event image and to tell an
    #: approaching vehicle from a departing one.
    plate_observations: list[PlateObservation] = field(default_factory=list)
    observation_count: int = 0
    peak_plate_width: int = 0
    #: Recent plate widths, oldest first, bounded to the growth window.
    recent_plate_widths: list[int] = field(default_factory=list)

    #: Per-plate-string crops. The event's images are drawn from the entry
    #: matching whichever string wins validation, NOT from the track's global
    #: quality peak — see PlateHypothesisEvidence and evidence_for.
    hypothesis_evidence: dict = field(default_factory=dict)

    # Colour is voted across frames like the plate is: a single frame's
    # sample swings wildly with glare, shadow and white balance.
    _vehicle_color_votes: dict = field(default_factory=dict)
    _plate_color_votes: dict = field(default_factory=dict)

    # -- ingestion ---------------------------------------------------------
    def touch(self, track: Track, now: float) -> None:
        self.last_seen = now
        self.detect_confidence = max(self.detect_confidence, track.confidence)
        self.centroid_history = list(track.centroid_history)
        if track.class_name and track.class_name != "vehicle":
            self.vehicle_type = track.class_name

    def add_read(
        self,
        read: PlateRead,
        plate_det_confidence: float,
        quality: QualityBreakdown,
        frame_ts: float,
        plate_crop: Optional[np.ndarray] = None,
        vehicle_crop: Optional[np.ndarray] = None,
        grammar_cfg: Optional[grammar_decode.GrammarDecodeConfig] = None,
        frame_idx: int = 0,
        appearance: int = 0,
    ) -> WeightedRead:
        # Repair each read BEFORE it votes, not after the vote is settled.
        # Two reasons: a read that repairs to a valid plate deserves full
        # grammar weight rather than the 0.6 an invalid string gets, and two
        # differently-wrong reads that repair to the same plate should have
        # their votes combined instead of splitting the ballot.
        #
        # resolve() rather than repair(): the same argument applies to a read
        # carrying a stray glyph off the hologram or the border, and that read
        # is a DIFFERENT LENGTH from its clean siblings — so without trimming
        # it not only votes wrong, it drags the positional vote's modal length
        # with it.
        # Evidence-driven correction FIRST, map-based repair second.
        #
        # The order is the point. `resolve` substitutes from a hand-written
        # confusion map, which encodes which glyphs a human thinks look alike;
        # `grammar_decode` substitutes from the recognizer's own distribution
        # at that position, which is what the model actually confused on this
        # crop. For DL7CP8161 read as CLZCP0161 the map yields `2` at
        # position 2 (confidently wrong) while the model's own runner-up is
        # `7`. Where both could act, the evidence should win.
        #
        # `resolve` itself is unchanged and still runs whenever the decoder
        # declines or has no alternatives to work from — which is every
        # recognizer that reports only aggregate confidence.
        decoded = grammar_decode.decode(
            read.text,
            read.per_char_alternatives,
            read.per_char_confidence,
            grammar_cfg,
        )
        if decoded.changed:
            text = decoded.text
            corrections = list(decoded.corrections)
        else:
            repaired = postprocess.resolve(read.text)
            text = repaired.text
            corrections = list(repaired.corrections)
            if decoded.declined:
                # A correction was considered and refused. Recording that is
                # the difference between "we never looked" and "we looked and
                # the evidence did not support it".
                corrections.append(f"grammar-decode declined: {decoded.declined}")
        weighted = WeightedRead(
            text=text,
            raw_text=read.raw_text or read.text,
            rec_confidence=read.confidence,
            per_char_confidence=list(read.per_char_confidence),
            plate_det_confidence=plate_det_confidence,
            quality=quality.score,
            grammar=postprocess.grammar_factor(text),
            frame_ts=frame_ts,
            plate_crop=plate_crop,
            corrections=corrections,
            frame_idx=frame_idx,
            appearance=appearance,
        )
        self.reads.append(weighted)

        # Bind this frame's crops to the plate string it produced, so the
        # event's images can later be taken from the hypothesis that actually
        # wins validation rather than from the track's global quality peak.
        self._note_hypothesis(text, quality.score, frame_idx, frame_ts, plate_crop, vehicle_crop)

        # The global bests are retained as a LAST-RESORT fallback only — see
        # `evidence_for`. They are what produced the mismatched images, so
        # nothing should reach for them while hypothesis evidence exists.
        if quality.score >= self.best_quality:
            self.best_quality = quality.score
            if plate_crop is not None:
                self.best_plate_crop = plate_crop
        self.note_vehicle_crop(vehicle_crop, quality.score)

        # Cap memory on a vehicle that parks in the ROI: the earliest reads of
        # a long dwell are the distant, low-quality ones anyway.
        if len(self.reads) > 60:
            self.reads.sort(key=lambda r: r.weight, reverse=True)
            del self.reads[40:]
            self.reads.sort(key=lambda r: r.frame_ts)
        return weighted

    # -- plate-hypothesis evidence -----------------------------------------
    def _note_hypothesis(
        self,
        text: str,
        quality: float,
        frame_idx: int,
        ts: float,
        plate_crop: Optional[np.ndarray],
        vehicle_crop: Optional[np.ndarray],
    ) -> None:
        if not text:
            return
        evidence = self.hypothesis_evidence.get(text)
        if evidence is None:
            evidence = PlateHypothesisEvidence(text=text)
            self.hypothesis_evidence[text] = evidence
        evidence.consider(quality, frame_idx, ts, plate_crop, vehicle_crop)

        if len(self.hypothesis_evidence) > MAX_HYPOTHESES:
            weakest = min(
                self.hypothesis_evidence.values(),
                key=lambda e: max(e.plate_quality, e.vehicle_quality),
            )
            if weakest.text != text:
                dropped = self.hypothesis_evidence.pop(weakest.text)
                dropped.plate_crop = None
                dropped.vehicle_crop = None

    def evidence_for(self, plate: str) -> Optional[PlateHypothesisEvidence]:
        """Crops belonging to the winning plate, or None.

        Three tiers, most specific first:

        1. A hypothesis whose string IS the winning plate. The normal case:
           some read said exactly this, and its crops are the right images.
        2. A hypothesis the winning plate plausibly came FROM. The validator
           can emit a string no single read produced — positional fusion
           assembles one character by character, and grammar repair or
           registry snapping may alter it afterwards. ``same_vehicle`` is the
           existing test for "these two strings plausibly describe one
           vehicle", so the closest compatible hypothesis carries the images.
        3. Nothing. The plate is not supported by any read's imagery, and the
           caller must say so rather than substitute an unrelated frame.

        Within a tier the best-quality hypothesis wins, so this never
        degrades image quality relative to the old behaviour EXCEPT where the
        old behaviour was reaching into another vehicle's frames.
        """
        if not plate:
            return None
        exact = self.hypothesis_evidence.get(plate)
        if exact is not None:
            return exact

        compatible = [
            e for e in self.hypothesis_evidence.values()
            if postprocess.same_vehicle(e.text, plate)
        ]
        if not compatible:
            return None
        # Prefer the closest string, then the best crop, so a repaired or
        # snapped plate takes the images of the read it was derived from.
        return max(
            compatible,
            key=lambda e: (
                postprocess.similarity(e.text, plate),
                max(e.plate_quality, e.vehicle_quality),
            ),
        )

    def consider_colors(self, vehicle: Optional[str], plate: Optional[str]) -> None:
        if vehicle:
            self._vehicle_color_votes[vehicle] = self._vehicle_color_votes.get(vehicle, 0) + 1
        if plate:
            self._plate_color_votes[plate] = self._plate_color_votes.get(plate, 0) + 1

    @property
    def vehicle_color(self) -> Optional[str]:
        if not self._vehicle_color_votes:
            return None
        return max(self._vehicle_color_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def plate_color(self) -> Optional[str]:
        if not self._plate_color_votes:
            return None
        return max(self._plate_color_votes.items(), key=lambda kv: kv[1])[0]

    def note_vehicle_crop(self, crop: Optional[np.ndarray], quality: Optional[float] = None) -> None:
        """Keep an image even for a vehicle whose plate is never read — an
        unknown vehicle with no plate is still an event worth showing.

        ``quality`` is the plate crop's score, used as a proxy for how good a
        view of the vehicle this frame is: a sharp, large plate means the
        vehicle was close and in focus too. Without a score (no plate found
        yet) the first crop is kept, so there is always an image; with one,
        the best-scoring frame wins.

        Tracked with its own high-water mark rather than sharing
        ``best_quality``, which now moves with plate OBSERVATIONS: a track can
        observe a better plate than any it managed to read, and the vehicle
        image must not get stuck on the first frame as a side effect.
        """
        if crop is None:
            return
        if quality is None:
            if self.best_vehicle_crop is None:
                self.best_vehicle_crop = crop
            return
        if self.best_vehicle_crop is None or quality >= self.best_vehicle_quality:
            self.best_vehicle_crop = crop
            self.best_vehicle_quality = quality

    def mark_disputed(self) -> None:
        self.disputed = True

    # -- plate observations ------------------------------------------------
    def note_observation(
        self,
        observation: PlateObservation,
        keep: int = 6,
        growth_window: int = 3,
        min_distance: int = crop_hash.DEFAULT_MIN_DISTANCE,
    ) -> None:
        """Record one look at the plate, whether or not it gets recognized.

        Retains the top ``keep`` by quality, but QUALITY-DIVERSE: a new
        observation that looks near-identical to one already held replaces it
        if better and is discarded if not, rather than taking a second slot.

        Why diversity and not just quality. A vehicle stopped at the boom
        produces frames that differ only by sensor noise, and every one of
        them scores about the same. Pure top-K then fills all six slots with
        six copies of one view — which is the worst possible use of the set,
        because its two consumers both need genuinely different frames: the
        targeted retry needs a crop the recognizer has not already failed on,
        and the approach signal needs to see change. Six near-duplicates
        provide one crop and no signal.
        """
        self.observation_count += 1
        self.peak_plate_width = max(self.peak_plate_width, observation.width)

        self.recent_plate_widths.append(observation.width)
        # +1 so `approach_growth` can compare across a full window.
        if len(self.recent_plate_widths) > max(1, growth_window) + 1:
            del self.recent_plate_widths[0]

        # Collapse against a near-identical view already held.
        for index, existing in enumerate(self.plate_observations):
            if crop_hash.is_distinct(observation.appearance, existing.appearance, min_distance):
                continue
            if observation.quality > existing.quality:
                self.plate_observations[index] = observation
                existing.release()
            else:
                observation.release()
            self._refresh_best(observation)
            return

        self.plate_observations.append(observation)
        if len(self.plate_observations) > max(1, keep):
            # Drop the weakest, not the oldest: the best view of a plate is
            # usually mid-approach, and a fresher but worse crop should never
            # displace it.
            worst = min(range(len(self.plate_observations)), key=lambda i: self.plate_observations[i].quality)
            self.plate_observations.pop(worst).release()

        self._refresh_best(observation)

    def _refresh_best(self, observation: PlateObservation) -> None:
        if observation.quality >= self.best_quality and observation.crop is not None:
            self.best_quality = observation.quality
            self.best_plate_crop = observation.crop

    def unread_observations(self) -> list[PlateObservation]:
        """Retained crops the recognizer has not been spent on, best first.

        The pool a targeted retry draws from. An observation the recognizer
        already saw will produce the same characters again — a deterministic
        model on an identical crop has nothing new to say — so a retry that
        reuses one is a wasted call.
        """
        return sorted(
            (o for o in self.plate_observations if not o.ocr_ran and o.crop is not None),
            key=lambda o: o.quality,
            reverse=True,
        )

    def approach_growth(self, window: int = 3) -> float:
        """How much the plate has grown over the last ``window`` observations.

        1.0 when there is not enough history to tell, which the scheduler
        reads as "not approaching" — the conservative answer, because it lets
        recognition proceed rather than waiting on evidence we do not have.
        """
        widths = self.recent_plate_widths
        if len(widths) < 2:
            return 1.0
        oldest = widths[max(0, len(widths) - 1 - window)]
        if oldest <= 0:
            return 1.0
        return widths[-1] / oldest

    @property
    def best_observation(self) -> Optional[PlateObservation]:
        if not self.plate_observations:
            return None
        return max(self.plate_observations, key=lambda o: o.quality)

    # -- OCR budget --------------------------------------------------------
    def ocr_remaining(self, budget: int) -> int:
        return max(0, budget + self.retry_allowance - self.ocr_calls)

    def nearly_settled(
        self,
        max_unstable: int = 2,
        min_reads: int = 2,
        min_posterior: float = 0.0,
        min_margin: float = 0.0,
    ) -> Optional[list[int]]:
        """Positions still in doubt, when the plate is otherwise settled.

        Returns the unstable position indices when there are between one and
        ``max_unstable`` of them, and None otherwise — None meaning either
        "nothing is in doubt" or "too much is in doubt to be worth one more
        look". Both of those are answers to leave alone: the first needs no
        retry, and the second will not be rescued by a single extra read.

        This is the signal behind targeted retry. It is deliberately the same
        fusion the validator will run at finalize, so a retry is granted on
        exactly the evidence the verdict will later be based on.
        """
        if len(self.reads) < min_reads:
            return None
        fusion = char_fusion.fuse(self.reads)
        if fusion is None:
            return None
        unstable = fusion.unstable(
            min_posterior or char_fusion.DEFAULT_MIN_POSTERIOR,
            min_margin or char_fusion.DEFAULT_MIN_MARGIN,
        )
        if not unstable or len(unstable) > max_unstable:
            return None
        return [p.index for p in unstable]

    def grant_retry(self, extra_calls: int, positions: list[int]) -> None:
        """Extend this track's OCR budget once, because the plate is one or
        two characters from settled.

        Once only, and recorded: a retry that does not resolve the position
        must not be able to grant itself another. ``retry_positions`` is kept
        for the audit trail — it is what the extra spend was bought for.
        """
        if self.retry_granted:
            return
        self.retry_granted = True
        self.retry_allowance = max(0, extra_calls)
        self.retry_positions = list(positions)

    def note_ocr_call(self) -> None:
        self.ocr_calls += 1
        self.frames_since_ocr = 0
        self.consecutive_ocr_waits = 0
        self.last_ocr_reason = ""

    def note_ocr_wait(self, reason: str) -> None:
        self.consecutive_ocr_waits += 1
        self.last_ocr_reason = reason

    # -- gating ------------------------------------------------------------
    def should_attempt_plate(self, min_interval_frames: int, max_attempts: int = 40) -> bool:
        if self.locked:
            return False
        if self.plate_attempts >= max_attempts:
            return False
        return self.frames_since_plate_attempt >= min_interval_frames

    def note_plate_attempt(self) -> None:
        self.frames_since_plate_attempt = 0
        self.plate_attempts += 1

    def consider_lock(self, min_reads: int = 4, min_support: float = 0.9) -> None:
        """Stop spending CPU on a track whose plate is no longer in doubt.

        On a busy gate this roughly halves plate-stage calls, which is the
        single largest saving available once the cascade is in place.
        """
        if self.locked or len(self.reads) < min_reads:
            return
        totals: dict[str, float] = {}
        for read in self.reads:
            totals[read.text] = totals.get(read.text, 0.0) + read.weight
        total = sum(totals.values())
        if total <= 0:
            return
        best_text, best_weight = max(totals.items(), key=lambda kv: kv[1])
        if best_weight / total >= min_support and postprocess.is_valid(best_text):
            self.locked = True

    # -- reporting ---------------------------------------------------------
    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def read_count(self) -> int:
        return len(self.reads)

    @property
    def distinct_texts(self) -> int:
        return len({r.text for r in self.reads})
