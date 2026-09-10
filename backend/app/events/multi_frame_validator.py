"""Multi-frame validation.

Given N noisy reads of one track, emit one plate with a calibrated confidence,
or emit nothing. The stages, and the reason each exists:

1. Whole-string weighted vote — settles the common case, where most frames
   already agree and one disagrees.
2. Per-character temporal fusion — settles the hard case, where no single
   string has a majority but each character position does. See
   ``char_fusion``, which also says how sure it is of every position.
3. Grammar repair — fixes a confusion the recognizer could not know was wrong,
   using the plate format as the constraint.
4. Registry snapping — optional; resolves a one-character gap to a plate the
   society has actually registered. Never onto a blacklisted plate, and never
   over a character position the frames had already settled.
5. Score, then classify — CONFIRMED, PROBABLE or UNRESOLVED.

OCR CONFIDENCE IS NOT ANPR CONFIDENCE
-------------------------------------
The recognizer answers "how sure am I of the characters I emitted?". That is a
statement about pixels and glyphs. It is not, and cannot be, a statement about
whether the system has identified a vehicle — the model has no idea whether it
saw a whole plate, whether the string is a registration that could exist, or
whether the other nine frames agreed with it.

A recognizer that reports ``UP1606`` at 94% is not wrong. It really is 94%
sure of those six glyphs. But ``UP1606`` is not a plate: it is too short to be
a whole Indian registration, so what the 94% actually means is "I am very sure
about the part of the plate I could see". Publishing 94% as the ANPR
confidence would convert a confident partial read into a confident wrong
answer, which is the worst failure this system can produce — an operator
trusts a high number.

So the two are computed and reported separately, and the path from one to the
other is deliberately lossy:

  * ``ocr_confidence`` — the recognizer's own, weight-averaged, untouched.
    Reported for diagnosis; never the headline.
  * ``confidence`` — the ANPR confidence. OCR confidence is one of four terms
    and the smallest of them, and the result is then CAPPED by what could not
    be established: completeness first, then format, then state code.

Caps rather than multipliers. A multiplier says "this is somewhat less likely";
a cap says "whatever the other evidence, the system will not claim more than
this about a string it cannot show is a whole, well-formed plate". For an
incomplete read that is the correct semantics, and 0.94 * 0.7 = 0.66 — which
the old multiplier produced, and which is still high enough to look
trustworthy — is not.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol

from ..ai.plate_recognizer import postprocess
from . import char_fusion
from .char_fusion import FusionResult
from .track_state import TrackState, WeightedRead

logger = logging.getLogger("anpr.events.validator")


class RecognitionState(str, Enum):
    """How much the system is willing to claim about a plate.

    Three states rather than a bare confidence number because the three call
    for different handling downstream, and a threshold buried in a UI is not
    the place for that decision to live.
    """

    #: Settled. Well-formed, agreed across frames, no shaky character. Safe to
    #: act on automatically — open the boom, match the resident, count the
    #: entry.
    CONFIRMED = "confirmed"
    #: A best reading that is probably right but not established. Worth
    #: logging and showing, not worth acting on unattended.
    PROBABLE = "probable"
    #: The system did not identify the plate. Either incomplete, or
    #: ill-formed, or the frames could not agree. An event may still be stored
    #: (a vehicle WAS there) but the plate must not be presented as known.
    UNRESOLVED = "unresolved"


@dataclass
class ValidationConfig:
    min_reads: int = 3
    min_read_weight: float = 0.05
    strong_support: float = 0.60
    margin: float = 0.15
    #: PROBABLE floor. Below this the plate is UNRESOLVED.
    min_final_confidence: float = 0.55
    #: CONFIRMED floor. A plate at or above this, well-formed and with no
    #: unstable character position, is settled.
    confirm_confidence: float = 0.80
    #: Below this support, try to snap to a registered plate one confusable
    #: character away. Above it, trust the vote.
    snap_below: float = 0.85
    registry_snap: bool = True
    #: A character position the frames settled at or above this posterior is
    #: not up for revision by the registry. See _snap_to_registry.
    snap_protect_posterior: float = 0.80
    #: When False, a low-confidence result is still emitted but flagged, so an
    #: operator can correct it. When True it is dropped entirely. Sites that
    #: care more about a complete log than a clean one set this False.
    drop_low_confidence: bool = True
    require_grammar: bool = False

    # -- ANPR confidence weights (must sum to 1.0) --------------------------
    #
    # OCR confidence is deliberately the SMALLEST term. It was 0.30 when it
    # was the only per-character signal available; now that fusion produces a
    # real per-position posterior, the recognizer's own aggregate opinion is
    # the weakest of the four and is weighted accordingly.
    #: Whole-string agreement across frames.
    weight_support: float = 0.35
    #: Fused per-character confidence — see char_fusion.char_support.
    weight_char_support: float = 0.35
    #: The recognizer's own confidence.
    weight_ocr: float = 0.15
    #: Best plate-crop quality seen.
    weight_quality: float = 0.15

    # -- confidence caps ----------------------------------------------------
    #
    # Each cap is placed so the result lands in the state it deserves given
    # the default thresholds above.
    #: Too short to be a whole registration. Well below min_final_confidence,
    #: so an incomplete read can never be anything but UNRESOLVED however
    #: confident the recognizer was about the fragment it saw.
    cap_incomplete: float = 0.35
    #: Matches no known plate shape. Also below min_final_confidence.
    cap_no_format: float = 0.50
    #: Right shape, state code not in the RTO list. Below
    #: confirm_confidence, so such a plate can be PROBABLE but never
    #: CONFIRMED — it may be a valid new series, or a misread prefix, and the
    #: system cannot tell which.
    cap_unknown_state: float = 0.75

    # -- per-character stability -------------------------------------------
    min_char_posterior: float = char_fusion.DEFAULT_MIN_POSTERIOR
    min_char_margin: float = char_fusion.DEFAULT_MIN_MARGIN
    grammar_floor: float = char_fusion.DEFAULT_GRAMMAR_FLOOR
    #: How much the weakest character dominates char_support. 1.0 restores the
    #: old pure-minimum rule.
    char_min_weight: float = char_fusion.DEFAULT_MIN_WEIGHT
    #: Correlated-evidence clustering. See char_fusion._independence_scale.
    cluster_hamming: int = char_fusion.DEFAULT_CLUSTER_DISTANCE
    cluster_frame_gap: int = char_fusion.DEFAULT_CLUSTER_FRAME_GAP
    #: Fraction of character positions that may be unsettled before the plate
    #: is UNRESOLVED outright, whatever the aggregate score says.
    #:
    #: Whole-string agreement is a RATIO and therefore scale-free: five reads
    #: that all say the same thing produce support 1.0 whether each was 99%
    #: sure or 20% sure. Agreement plus a good crop can then carry the
    #: aggregate over the PROBABLE floor while not one character is actually
    #: settled — five confident-looking frames of unanimous guessing. If the
    #: system cannot pin most of the characters, it has not read the plate,
    #: and no amount of consistency about that should say otherwise.
    #:
    #: A third: up to 3 shaky positions in a 10-character plate stays
    #: PROBABLE (a best guess worth logging), 4 or more is UNRESOLVED.
    max_unstable_fraction: float = 0.34

    def __post_init__(self) -> None:
        total = (
            self.weight_support + self.weight_char_support
            + self.weight_ocr + self.weight_quality
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ANPR confidence weights must sum to 1.0, got {total:.4f}")
        if self.confirm_confidence < self.min_final_confidence:
            raise ValueError("confirm_confidence must be at or above min_final_confidence")


@dataclass
class FinalPlate:
    text: str
    #: ANPR confidence: how sure the SYSTEM is that this is the vehicle's
    #: registration. Capped by completeness and format — see the module
    #: docstring. This is the number a UI should show.
    confidence: float
    support: float
    read_count: int
    distinct_variants: int
    grammar_valid: bool
    corrections: list[str] = field(default_factory=list)
    best_read_confidence: float = 0.0
    best_quality: float = 0.0

    #: What the system is willing to claim. See RecognitionState.
    state: RecognitionState = RecognitionState.PROBABLE
    #: The RECOGNIZER's own confidence in the glyphs, weight-averaged across
    #: reads. Diagnostic only: never compare this to a threshold that gates
    #: an action, and never show it as "the" confidence.
    ocr_confidence: float = 0.0
    #: Fused per-character confidence, aggregated toward the weakest position.
    char_support: float = 0.0
    #: The old pure-weakest-character metric, kept for comparison and audit.
    weakest_char_posterior: float = 0.0
    #: Positions the frames could not settle, as "pos4=B@0.52 vs 8@0.44".
    unstable_positions: list[str] = field(default_factory=list)
    #: Why the ANPR confidence was capped, if it was.
    cap_reason: str = ""

    @property
    def is_confirmed(self) -> bool:
        return self.state is RecognitionState.CONFIRMED

    @property
    def review_required(self) -> bool:
        """Anything not settled wants a human eye."""
        return self.state is not RecognitionState.CONFIRMED


class RegistryLookup(Protocol):
    """What the validator needs from the vehicle registry.

    A Protocol rather than an import: `events/` must stay testable without a
    database, and the worker passes in a small cached snapshot rather than a
    live session.
    """

    def unique_confusable_neighbour(self, plate: str) -> Optional[tuple[str, str]]:
        """Returns (plate_number, status) when exactly one registered plate is
        a single confusable character away, else None."""


class MultiFrameValidator:
    def __init__(
        self,
        cfg: ValidationConfig | None = None,
        registry: RegistryLookup | None = None,
    ):
        self.cfg = cfg or ValidationConfig()
        self.registry = registry

    # -- public ------------------------------------------------------------
    def validate(self, state: TrackState) -> Optional[FinalPlate]:
        cfg = self.cfg
        reads = [r for r in state.reads if r.weight >= cfg.min_read_weight]
        # Fragments are removed from the ballot entirely rather than merely
        # down-weighted. A half-seen plate carries NO information about the
        # whole one, and left in the pool it does real damage: it is read
        # cleanly every frame the occlusion lasts, so it accumulates support,
        # and worse, its length can win the modal-length test and hand the
        # positional vote a pool that excludes the frames that saw the whole
        # plate. Down-weighting alone let three occluded frames beat one clear
        # one.
        reads = [r for r in reads if not postprocess.is_fragment(r.text)]
        if len(reads) < cfg.min_reads:
            # A track that produced reads but not enough usable ones is a
            # different situation from one that produced none at all: the
            # vehicle was there and the system could not read it. Flag it so it
            # shows up for review instead of vanishing.
            if not state.reads:
                return None
            state.mark_disputed()
            logger.info(
                "track %s: only %d/%d usable reads, no plate emitted",
                state.track_id, len(reads), len(state.reads),
            )
            if cfg.drop_low_confidence:
                return None
            # The site asked for a complete log. Report WHAT was read and that
            # the system does not believe it — an operator looking at a gap in
            # the log cannot tell "no vehicle" from "a vehicle we could only
            # half see", and those need different responses.
            return self._unresolved(state)

        # Fused per-character evidence. Computed once, up front, and used for
        # three different things: assembling a candidate when the string vote
        # is inconclusive, scoring how sure we are of each position, and
        # protecting settled positions from the registry.
        fusion = char_fusion.fuse(reads, grammar_floor=cfg.grammar_floor)

        candidate, support, votes = self._vote(reads, fusion)
        if candidate is None:
            return None

        # Reads are repaired individually at ingest (see TrackState.add_read),
        # so the winner is normally already valid. Carry the corrections that
        # the winning reads applied into the event, so the audit trail shows
        # why the stored plate differs from what the recognizer emitted.
        corrections: list[str] = []
        for read in reads:
            if read.text == candidate:
                for fix in read.corrections:
                    if fix not in corrections:
                        corrections.append(fix)

        # A candidate assembled character-by-character by fusion has not been
        # through repair, so try it here.
        repair = postprocess.repair(candidate)
        if repair.repaired:
            candidate = repair.text
            corrections.extend(repair.corrections)

        if cfg.registry_snap and self.registry is not None and support < cfg.snap_below:
            snapped = self._snap_to_registry(candidate, fusion)
            if snapped:
                corrections.append(f"registry-snap: {candidate}->{snapped}")
                candidate = snapped

        grammar_valid = postprocess.is_valid(candidate)
        incomplete = postprocess.is_fragment(candidate)

        if cfg.require_grammar and not grammar_valid:
            state.mark_disputed()
            return None

        ocr_confidence = self._ocr_confidence(reads)
        char_support = self._char_support(reads, candidate, fusion)
        unstable = fusion.unstable(cfg.min_char_posterior, cfg.min_char_margin) if fusion else []
        confidence, cap_reason = self._anpr_confidence(
            reads, support, char_support, ocr_confidence, candidate, incomplete, grammar_valid
        )

        recognition = self._classify(
            confidence, grammar_valid, incomplete, len(unstable), len(candidate)
        )

        if recognition is not RecognitionState.CONFIRMED:
            state.mark_disputed()

        if recognition is RecognitionState.UNRESOLVED:
            logger.info(
                "track %s: %s UNRESOLVED (anpr %.2f, ocr %.2f, support %.2f, chars %.2f, "
                "%d reads, cap=%s, unstable=%s)",
                state.track_id, candidate, confidence, ocr_confidence, support, char_support,
                len(reads), cap_reason or "none", [p.index for p in unstable],
            )
            # An incomplete read is not merely uncertain, it is a plate no
            # vehicle carries. It is dropped under the same setting that
            # governs low-confidence results, so a site that wants a complete
            # log still gets the row — flagged, UNRESOLVED, and capped well
            # below anything that reads as trustworthy.
            if cfg.drop_low_confidence:
                return None

        return FinalPlate(
            text=candidate,
            confidence=round(confidence, 4),
            support=round(support, 4),
            read_count=len(reads),
            distinct_variants=len(votes),
            grammar_valid=grammar_valid,
            corrections=corrections,
            best_read_confidence=max(r.rec_confidence for r in reads),
            best_quality=max(r.quality for r in reads),
            state=recognition,
            ocr_confidence=round(ocr_confidence, 4),
            char_support=round(char_support, 4),
            weakest_char_posterior=round(fusion.weakest_posterior, 4) if fusion else 0.0,
            unstable_positions=[p.describe() for p in unstable],
            cap_reason=cap_reason,
        )

    def _unresolved(self, state: TrackState) -> FinalPlate:
        """A best-effort record for a track whose reads were all unusable.

        This is the ``UP1606`` case, and it is the clearest illustration of
        why the two confidences are separate numbers. The recognizer really
        was 94% sure of those six glyphs, and that number is reported
        unchanged as ``ocr_confidence`` because it is true and diagnostic. The
        ANPR confidence is capped at ``cap_incomplete`` and the state is
        UNRESOLVED, because six characters is not a registration and the
        system has not identified the vehicle. Same evidence, two honest
        answers to two different questions.
        """
        reads = state.reads
        heaviest = max(reads, key=lambda r: r.weight)
        ocr_confidence = self._ocr_confidence(reads)
        return FinalPlate(
            text=heaviest.text,
            confidence=round(min(ocr_confidence, self.cfg.cap_incomplete), 4),
            support=0.0,
            read_count=len(reads),
            distinct_variants=len({r.text for r in reads}),
            grammar_valid=False,
            best_read_confidence=max(r.rec_confidence for r in reads),
            best_quality=max(r.quality for r in reads),
            state=RecognitionState.UNRESOLVED,
            ocr_confidence=round(ocr_confidence, 4),
            cap_reason="incomplete",
        )

    # -- confidence --------------------------------------------------------
    @staticmethod
    def _ocr_confidence(reads: list[WeightedRead]) -> float:
        """The RECOGNIZER's confidence, weight-averaged. Nothing else.

        No grammar, no cross-frame agreement, no completeness. This is what
        the model said about the glyphs, and it is reported so an operator can
        tell "the model could not read it" from "the model read it fine but it
        was not a whole plate" — two failures with identical symptoms in the
        old single-number scheme.
        """
        total = sum(r.weight for r in reads) or 1.0
        return sum(r.rec_confidence * r.weight for r in reads) / total

    def _char_support(
        self, reads: list[WeightedRead], candidate: str, fusion: Optional[FusionResult]
    ) -> float:
        if fusion is not None:
            return fusion.char_support(self.cfg.char_min_weight)
        # No fusable pool: fall back to the mean per-character confidence of
        # the reads that actually agree with the candidate. Weaker evidence
        # than a fused posterior, and it should not masquerade as one, but it
        # is still per-character and still beats reusing the aggregate.
        matching = [r for r in reads if r.text == candidate]
        pool = matching or reads
        scores = [
            r.char_confidence(i)
            for r in pool
            for i in range(len(r.text))
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def _anpr_confidence(
        self,
        reads: list[WeightedRead],
        support: float,
        char_support: float,
        ocr_confidence: float,
        candidate: str,
        incomplete: bool,
        grammar_valid: bool,
    ) -> tuple[float, str]:
        """The system's confidence that this is the vehicle's registration.

        Returns (confidence, cap_reason). See the module docstring for why the
        caps are caps and not multipliers.
        """
        cfg = self.cfg
        best_quality = max(r.quality for r in reads)
        confidence = (
            cfg.weight_support * support
            + cfg.weight_char_support * char_support
            + cfg.weight_ocr * ocr_confidence
            + cfg.weight_quality * best_quality
        )
        confidence = max(0.0, min(1.0, confidence))

        # Ordered most-fundamental first, and only the first that applies is
        # reported: being incomplete makes the format question moot.
        cap, reason = 1.0, ""
        if incomplete:
            cap, reason = cfg.cap_incomplete, "incomplete"
        elif postprocess.match_format(candidate) is None:
            cap, reason = cfg.cap_no_format, "no_known_format"
        elif not grammar_valid:
            # Shape is right, so the only way is_valid failed is the state code.
            cap, reason = cfg.cap_unknown_state, "unknown_state_code"

        if confidence > cap:
            return cap, reason
        return confidence, reason if reason else ""

    def _classify(
        self,
        confidence: float,
        grammar_valid: bool,
        incomplete: bool,
        unstable_count: int,
        length: int,
    ) -> RecognitionState:
        """Confidence plus the reasons a number alone should not decide.

        CONFIRMED needs more than a high score: it needs a well-formed plate,
        no incompleteness, and no character position the frames left shaky. A
        plate can score well on agreement and quality while one character
        remains a coin flip, and that plate is not settled — it is a good
        guess about a specific vehicle, which is exactly what PROBABLE is for.

        UNRESOLVED likewise is not purely a threshold. A plate whose
        characters are mostly unsettled has not been read, however consistent
        the frames were about not reading it — see max_unstable_fraction.
        """
        cfg = self.cfg
        if incomplete or confidence < cfg.min_final_confidence:
            return RecognitionState.UNRESOLVED
        if length and (unstable_count / length) > cfg.max_unstable_fraction:
            return RecognitionState.UNRESOLVED
        if confidence >= cfg.confirm_confidence and grammar_valid and not unstable_count:
            return RecognitionState.CONFIRMED
        return RecognitionState.PROBABLE

    # -- stages ------------------------------------------------------------
    def _vote(
        self, reads: list[WeightedRead], fusion: Optional[FusionResult]
    ) -> tuple[Optional[str], float, dict[str, float]]:
        votes: dict[str, float] = defaultdict(float)
        for read in reads:
            votes[read.text] += read.weight
        total = sum(votes.values())
        if total <= 0:
            return None, 0.0, {}

        best, best_weight = max(votes.items(), key=lambda kv: kv[1])
        support = best_weight / total
        runner_up = max((w for t, w in votes.items() if t != best), default=0.0) / total

        if support >= self.cfg.strong_support and (support - runner_up) >= self.cfg.margin:
            return best, support, dict(votes)

        if fusion is None:
            # No length commands a plurality — fall back to the string vote
            # rather than returning nothing, and let the confidence score
            # reflect how weak the agreement was.
            return best, support, dict(votes)

        # The string vote was inconclusive, so the character-fused assembly
        # decides. Its support is the fused per-character aggregate, which now
        # accounts for the probability the recognizer withheld — see
        # char_fusion for why the old normalization could not.
        return fusion.text, fusion.char_support(self.cfg.char_min_weight), dict(votes)

    def _snap_to_registry(self, candidate: str, fusion: Optional[FusionResult]) -> Optional[str]:
        """Resolve a near-miss onto a registered plate, conservatively.

        Three refusals, in increasing subtlety:

        1. more than one registered plate fits — handled by the registry's
           ``unique_confusable_neighbour``, which returns nothing rather than
           picking. A closed set makes the question "closer to ONE plate than
           to any other", and an ambiguous answer is not an answer.
        2. the target is blacklisted. A blacklisted vehicle must be recognised
           on its own evidence, never manufactured from an uncertain read.
        3. NEW — the snap would overwrite a character position the frames had
           already settled. Snapping exists to resolve characters we were
           unsure about; a position the evidence pinned at high posterior is
           not up for revision by a list of plates that happens to contain a
           near-neighbour. Without this guard the registry can quietly turn a
           correctly-read visitor's plate into a resident's.
        """
        assert self.registry is not None
        try:
            hit = self.registry.unique_confusable_neighbour(candidate)
        except Exception:
            logger.warning("registry lookup failed during snapping", exc_info=True)
            return None
        if not hit:
            return None
        plate, status = hit
        if status == "blacklist":
            logger.info("declined registry-snap onto blacklisted plate %s", plate)
            return None

        if fusion is not None and len(plate) == len(candidate):
            for index, (mine, theirs) in enumerate(zip(candidate, plate)):
                if mine == theirs:
                    continue
                evidence = fusion.position(index)
                if evidence is not None and evidence.adjusted_posterior >= self.cfg.snap_protect_posterior:
                    logger.info(
                        "declined registry-snap %s->%s: position %d was settled (%s@%.2f)",
                        candidate, plate, index, evidence.char, evidence.adjusted_posterior,
                    )
                    return None
        return plate


class SnapshotRegistry:
    """A cached plate list for confusable lookup inside a worker process.

    Workers must not hold a database session in the pipeline loop, so the
    supervisor hands each worker a periodically refreshed snapshot. A few
    minutes of staleness is acceptable: a vehicle registered a moment ago
    resolves correctly on the API side at ingest time regardless.
    """

    def __init__(self, plates: dict[str, str] | None = None):
        self._plates: dict[str, str] = dict(plates or {})
        self._by_length: dict[int, list[str]] = defaultdict(list)
        self._reindex()

    def _reindex(self) -> None:
        self._by_length.clear()
        for plate in self._plates:
            self._by_length[len(plate)].append(plate)

    def replace(self, plates: dict[str, str]) -> None:
        self._plates = dict(plates)
        self._reindex()

    def __len__(self) -> int:
        return len(self._plates)

    def status_of(self, plate: str) -> Optional[str]:
        return self._plates.get(plate)

    def unique_confusable_neighbour(self, plate: str) -> Optional[tuple[str, str]]:
        if plate in self._plates:
            return None  # already an exact hit; nothing to snap
        matches = [
            other for other in self._by_length.get(len(plate), ())
            if postprocess.confusable(plate, other)
        ]
        if len(matches) != 1:
            return None
        return matches[0], self._plates[matches[0]]
