"""Per-character temporal fusion.

Given N reads of one plate, decide each character position independently and
say how sure we are of it. One implementation, used by two callers with
different questions:

  * the validator, at finalize: "what is the plate, and which positions am I
    not sure about?"
  * the OCR scheduler, mid-track: "is this plate nearly settled — would one
    more read finish it?"

WHY THIS IS NOT THE OLD POSITIONAL VOTE
---------------------------------------
The previous implementation accumulated ``weight * char_confidence`` per
character and normalized by the mass it had assigned. That throws the
per-character probabilities away whenever the reads agree: three reads that
all say ``A`` at 10% confidence produced a support of

    0.1 / 0.1 = 1.0

because the only character with any mass was the only character considered.
Unanimous guessing scored the same as unanimous certainty, and the positional
vote — the mechanism specifically meant to catch one bad character — reported
full confidence on a plate the model had no idea about.

The fix is to give the probability the recognizer withheld somewhere to go.
Each read contributes ``w * p`` to the character it saw and ``w * (1 - p)`` to
an explicit UNKNOWN bucket, and the posterior divides by the total including
that bucket. Unanimous-at-0.10 now scores 0.10, which is what it always
deserved.

GRAMMAR AS INDEPENDENT EVIDENCE
-------------------------------
The plate format knows things the recognizer does not. When position 5 must be
a letter, the model's first choice is ``B`` and its runner-up is ``8``, the
format has already eliminated the runner-up — and the position is far more
settled than its raw posterior suggests. Corroborating those positions is what
makes the weakest-character rule strict where it should be (a genuinely
ambiguous character) without being harsh where it should not (a character the
grammar has decided).

Deliberately conservative, per the "keep grammar repair conservative"
constraint: the boost applies ONLY where a concrete runner-up is excluded by
the format mask. A position where every read agrees but at low confidence gets
nothing, because "must be a digit" narrows 36 options to 10 — real information,
but not enough to call a coin flip settled.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

from ..ai.plate_recognizer import postprocess

#: Label for the probability mass a read did NOT commit to the character it
#: reported. Never a real character, so it can never win a position.
UNKNOWN = ""

# --- defaults ---------------------------------------------------------------
#
# A position is "unstable" when the evidence does not clearly prefer one
# character. Two independent ways that happens, so two thresholds:
#
#   posterior  how much of the total mass the winner holds. 0.55 is just over
#              half: below it, the characters we did NOT pick collectively
#              outweigh the one we did.
#   margin     how far ahead of the runner-up it is. 0.20 keeps a position
#              honest when two characters split the mass nearly evenly, which
#              a posterior test alone can miss (0.45 vs 0.44 passes neither).
DEFAULT_MIN_POSTERIOR = 0.55
DEFAULT_MIN_MARGIN = 0.20

#: Posterior a grammar-corroborated position is credited with. Set at the
#: CONFIRMED-grade level rather than 1.0: the format excluding the runner-up is
#: strong evidence, not proof, and a position should never become
#: unimpeachable on grammar alone.
DEFAULT_GRAMMAR_FLOOR = 0.85

#: How much the weakest position dominates the aggregate. See
#: ``FusionResult.char_support``.
DEFAULT_MIN_WEIGHT = 0.70


#: Hamming distance below which two reads count as the same view of the plate.
#: See ``_independence_scale``.
DEFAULT_CLUSTER_DISTANCE = 6
#: Frame gap within which two similar-looking reads count as the same view.
#: Beyond it the vehicle has had time to move even if the crop looks alike.
DEFAULT_CLUSTER_FRAME_GAP = 4


class _CharRead(Protocol):
    """What fusion needs from a read. Structural, so this module stays
    testable without constructing a WeightedRead or a TrackState."""

    text: str
    weight: float

    def char_confidence(self, i: int) -> float: ...


@dataclass(frozen=True)
class PositionEvidence:
    """The fused verdict for one character position."""

    index: int
    char: str
    #: Winner's share of the total mass, INCLUDING the unknown bucket.
    posterior: float
    #: Best competing real character, "" when the reads never disagreed here.
    runner_up: str
    runner_up_posterior: float
    #: True when the plate format excluded the runner-up.
    grammar_locked: bool = False
    #: Distinct characters the reads reported at this position.
    variants: int = 1
    #: Posterior after grammar corroboration — what scoring actually uses.
    adjusted_posterior: float = 0.0

    @property
    def margin(self) -> float:
        return self.adjusted_posterior - self.runner_up_posterior

    def is_unstable(
        self, min_posterior: float = DEFAULT_MIN_POSTERIOR, min_margin: float = DEFAULT_MIN_MARGIN
    ) -> bool:
        if self.adjusted_posterior < min_posterior:
            return True
        # An uncontested position has no margin to measure and is exactly as
        # settled as its posterior says.
        return bool(self.runner_up) and self.margin < min_margin

    def describe(self) -> str:
        """One audit-trail line. Kept terse — it is stored per event."""
        detail = f"pos{self.index}={self.char}@{self.adjusted_posterior:.2f}"
        if self.runner_up:
            detail += f" vs {self.runner_up}@{self.runner_up_posterior:.2f}"
        if self.grammar_locked:
            detail += " (grammar)"
        return detail


@dataclass(frozen=True)
class FusionResult:
    text: str
    modal_length: int
    pool_size: int
    positions: list[PositionEvidence] = field(default_factory=list)
    #: Total read weight that did NOT match the modal length, as a fraction.
    #: High values mean the reads disagree about how long the plate is, which
    #: is a different and worse problem than disagreeing about a character.
    off_length_weight: float = 0.0
    #: Effective independent observations behind this result, after redundant
    #: near-identical views are discounted. Lower than ``pool_size`` whenever
    #: the vehicle sat still. See ``_independence_scale``.
    independent_views: float = 0.0

    def position(self, index: int) -> Optional[PositionEvidence]:
        for evidence in self.positions:
            if evidence.index == index:
                return evidence
        return None

    def unstable(
        self, min_posterior: float = DEFAULT_MIN_POSTERIOR, min_margin: float = DEFAULT_MIN_MARGIN
    ) -> list[PositionEvidence]:
        return [p for p in self.positions if p.is_unstable(min_posterior, min_margin)]

    def char_support(self, min_weight: float = DEFAULT_MIN_WEIGHT) -> float:
        """Aggregate per-character confidence, 0..1.

        A blend of the weakest position and the mean, weighted toward the
        weakest. Pure ``min`` — what this replaced — is unnecessarily harsh:
        it scores a plate with nine positions at 0.99 and one at 0.50 exactly
        the same as one with ten positions at 0.50, though the first is one
        character from settled and the second is not a reading at all.

        Weighting the minimum at 0.70 keeps the weakest character firmly in
        charge, so a genuinely ambiguous plate still fails, while letting
        broad agreement elsewhere count for something. Set min_weight to 1.0
        to restore the old pure-minimum behaviour.
        """
        if not self.positions:
            return 0.0
        posteriors = [p.adjusted_posterior for p in self.positions]
        weakest = min(posteriors)
        mean = sum(posteriors) / len(posteriors)
        blend = max(0.0, min(1.0, min_weight))
        return blend * weakest + (1.0 - blend) * mean

    @property
    def weakest_posterior(self) -> float:
        """The old metric, retained for the audit trail and for comparison."""
        if not self.positions:
            return 0.0
        return min(p.adjusted_posterior for p in self.positions)


def _independence_scale(reads: list) -> list[float]:
    """Per-read weight multipliers that discount correlated observations.

    Four consecutive frames of a vehicle stopped at the boom are not four
    independent looks at its plate. They are ONE look, sampled four times: the
    same crop, the same angle, the same glare, the same blur — so the same
    error, four times over, and a fusion that sums their weights treats a
    single mistake as quadruple confirmation. That is the mechanism behind a
    confident wrong answer on a stationary vehicle, and it gets worse the
    longer the vehicle waits.

    So reads are grouped into clusters of near-identical views (appearance
    hash within ``cluster_distance`` AND captured within ``cluster_frame_gap``
    frames), and a cluster of n members contributes **sqrt(n)** rather than n
    times its weight — each member scaled by 1/sqrt(n). That is the standard
    correction for effectively-correlated samples: four identical frames count
    as two independent observations, nine as three. Genuinely different views
    are untouched, which is the point — the discount penalises redundancy, not
    evidence.

    Both signals are required for a cluster. Appearance alone would merge two
    genuinely separate passes of a similar-looking vehicle; the frame gap
    alone would merge frames across a real change.

    Reads without an appearance hash (EasyOCR path, or a crop that could not
    be hashed) are never clustered — treated as fully independent, which is
    the old behaviour and the conservative choice when we cannot tell.
    """
    scale = [1.0] * len(reads)
    cluster_of: list[int] = [-1] * len(reads)
    clusters: list[list[int]] = []

    for i, read in enumerate(reads):
        appearance = getattr(read, "appearance", 0) or 0
        if not appearance:
            continue
        frame_idx = getattr(read, "frame_idx", None)
        for c, members in enumerate(clusters):
            head = reads[members[0]]
            head_appearance = getattr(head, "appearance", 0) or 0
            if not head_appearance:
                continue
            if hamming(appearance, head_appearance) >= DEFAULT_CLUSTER_DISTANCE:
                continue
            head_idx = getattr(head, "frame_idx", None)
            if (
                frame_idx is not None
                and head_idx is not None
                and abs(frame_idx - head_idx) > DEFAULT_CLUSTER_FRAME_GAP
            ):
                continue
            members.append(i)
            cluster_of[i] = c
            break
        else:
            cluster_of[i] = len(clusters)
            clusters.append([i])

    for members in clusters:
        if len(members) < 2:
            continue
        factor = 1.0 / math.sqrt(len(members))
        for i in members:
            scale[i] = factor
    return scale


def hamming(a: int, b: int) -> int:
    """Bits that differ between two appearance hashes."""
    return int(bin(a ^ b).count("1"))


def fuse(
    reads: Iterable[_CharRead],
    grammar_floor: float = DEFAULT_GRAMMAR_FLOOR,
) -> Optional[FusionResult]:
    """Fuse per-character evidence across reads of the modal length.

    Only same-length reads participate, as before: a differing length almost
    always means a truncated or merged read rather than a substitution, and
    letting it vote shifts every position after the difference. What is new is
    that the weight of the excluded reads is reported rather than discarded,
    so a caller can tell "one odd read out of ten" from "the reads cannot
    agree how long this plate is".

    Returns None when no length has at least two reads behind it — there is
    nothing to fuse, and inventing a per-character posterior from a single
    read would just relabel that read's own confidence as agreement.
    """
    reads = list(reads)
    # Discount redundant views BEFORE anything is counted, so the modal-length
    # vote and every per-position posterior all see the same corrected weights.
    scale = _independence_scale(reads)
    effective = {id(r): max(0.0, r.weight) * scale[i] for i, r in enumerate(reads)}

    pool_by_length: dict[int, list[_CharRead]] = defaultdict(list)
    length_weight: dict[int, float] = defaultdict(float)
    total_weight = 0.0
    for read in reads:
        pool_by_length[len(read.text)].append(read)
        length_weight[len(read.text)] += effective[id(read)]
        total_weight += effective[id(read)]

    if not length_weight:
        return None
    modal_length = max(length_weight.items(), key=lambda kv: kv[1])[0]
    pool = pool_by_length[modal_length]
    if len(pool) < 2 or modal_length <= 0:
        return None

    off_length = 0.0
    if total_weight > 0:
        off_length = (total_weight - length_weight[modal_length]) / total_weight

    raw: list[PositionEvidence] = []
    for i in range(modal_length):
        assigned: dict[str, float] = defaultdict(float)
        unknown = 0.0
        for read in pool:
            weight = effective[id(read)]
            probability = max(0.0, min(1.0, read.char_confidence(i)))
            assigned[read.text[i]] += weight * probability
            # The mass the recognizer withheld. Without this the posteriors
            # ignore per-character confidence entirely whenever reads agree.
            unknown += weight * (1.0 - probability)

        total = sum(assigned.values()) + unknown
        if total <= 0:
            return None

        ranked = sorted(assigned.items(), key=lambda kv: kv[1], reverse=True)
        char, mass = ranked[0]
        runner_char, runner_mass = ranked[1] if len(ranked) > 1 else (UNKNOWN, 0.0)
        raw.append(
            PositionEvidence(
                index=i,
                char=char,
                posterior=mass / total,
                runner_up=runner_char,
                runner_up_posterior=runner_mass / total,
                variants=len(assigned),
                adjusted_posterior=mass / total,
            )
        )

    text = "".join(p.char for p in raw)
    positions = _apply_grammar(text, raw, grammar_floor)
    independent = sum(scale[i] for i, r in enumerate(reads) if len(r.text) == modal_length)
    return FusionResult(
        text=text,
        modal_length=modal_length,
        pool_size=len(pool),
        positions=positions,
        off_length_weight=off_length,
        independent_views=independent,
    )


def _apply_grammar(
    text: str, positions: list[PositionEvidence], grammar_floor: float
) -> list[PositionEvidence]:
    """Credit positions where the plate format excludes the runner-up.

    Conservative on purpose. Three conditions, all required:

      * the assembled string matches a known plate shape at all — otherwise
        the mask is not evidence about anything;
      * the winning character satisfies the mask at that position;
      * there is a concrete runner-up and it VIOLATES the mask.

    The third is what keeps this honest. Without it, every position of every
    format-matching plate would be credited, and the grammar factor already
    accounts for format validity once in the score. Here it is only allowed to
    resolve a specific competition between two characters.
    """
    fmt = postprocess.match_format(text)
    if fmt is None or len(fmt.mask) != len(positions):
        return positions

    adjusted: list[PositionEvidence] = []
    for evidence in positions:
        want = fmt.mask[evidence.index]
        winner_fits = _fits(evidence.char, want)
        runner_fits = _fits(evidence.runner_up, want) if evidence.runner_up else True
        if winner_fits and evidence.runner_up and not runner_fits:
            adjusted.append(
                PositionEvidence(
                    index=evidence.index,
                    char=evidence.char,
                    posterior=evidence.posterior,
                    runner_up=evidence.runner_up,
                    runner_up_posterior=evidence.runner_up_posterior,
                    grammar_locked=True,
                    variants=evidence.variants,
                    adjusted_posterior=max(evidence.posterior, grammar_floor),
                )
            )
        else:
            adjusted.append(evidence)
    return adjusted


def _fits(char: str, want: str) -> bool:
    if not char:
        return False
    if want == postprocess.ALPHA:
        return char.isalpha()
    if want == postprocess.DIGIT:
        return char.isdigit()
    return True
