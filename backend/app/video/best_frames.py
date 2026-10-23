"""Which banked plate crops are worth sending to the VLM.

Pure: observations in, observations out. No model, no I/O, no state — the same
property ``ocr_scheduler`` has, for the same reason: it is the one decision in
deferred mode whose every threshold is a site-calibration knob, and it has to
be testable without a frame.

The VLM is the most expensive call in the system by two orders of magnitude, so
the set it is given is chosen, not sampled. Three rules, in priority order:

  1. USABLE: has pixels, and clears the quality floor. A crop below the floor
     is not "a worse read", it is no read — and a vehicle whose every crop is
     below it is a class C result, not a class B one.
  2. BEST FIRST: highest quality wins a slot. The best view of a plate is
     usually mid-approach, so this is not "the last K frames".
  3. DIVERSE: a candidate that looks like one already chosen (appearance hash)
     or was captured on a neighbouring processed frame is skipped, even if it
     scores higher than the next-best different view. Seven copies of one
     stationary view are one observation, and a fluent VLM will read all seven
     identically and the validator will count that as seven-fold agreement.

``TrackState.note_observation`` already collapses near-identical views at
ingest; this repeats the test at selection time because ingest only compares
against what it currently holds, and selection can add the frame-gap rule,
which ingest cannot apply to a hash alone.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from ..ai.quality import crop_hash
from ..ai.types import PlateObservation


@dataclass(frozen=True)
class SelectionConfig:
    #: Most crops sent per vehicle. 5-7 is the design range: fewer starves the
    #: fusion of independent looks, more is CPU spent re-reading a plate the
    #: first six already settled.
    best_k: int = 6
    #: Quality floor for an observation to count as usable at all.
    min_quality: float = 0.35
    #: Width floor in pixels; 0 disables it. Below this a plate is detectable
    #: but not legible, however sharp the few pixels it has.
    min_width: int = 0
    #: Appearance-hash distance (of 64) below which two crops are one view.
    min_hamming: int = crop_hash.DEFAULT_MIN_DISTANCE
    #: Minimum processed-frame distance between two chosen crops.
    min_frame_gap: int = 2

    def __post_init__(self) -> None:
        if self.best_k < 1:
            raise ValueError("best_k must be at least 1")


def is_usable(observation: PlateObservation, cfg: SelectionConfig) -> bool:
    if observation.crop is None or observation.crop.size == 0:
        return False
    if observation.quality < cfg.min_quality:
        return False
    return observation.width >= cfg.min_width


def usable_observations(
    observations: Iterable[PlateObservation], cfg: SelectionConfig
) -> list[PlateObservation]:
    return [o for o in observations if is_usable(o, cfg)]


def select_best(
    observations: Iterable[PlateObservation], cfg: Optional[SelectionConfig] = None
) -> list[PlateObservation]:
    """Up to ``cfg.best_k`` usable, mutually distinct observations, best first.

    Fewer than ``best_k`` come back when fewer distinct usable views exist. The
    set is NOT padded with near-duplicates to reach the target: a vehicle that
    sat still produced one view, and sending it once is the right number.
    """
    cfg = cfg or SelectionConfig()
    pool = sorted(
        usable_observations(observations, cfg),
        key=lambda o: (o.quality, o.width),
        reverse=True,
    )
    pool = _dominant_chain(pool, cfg)
    chosen: list[PlateObservation] = []
    for candidate in pool:
        if len(chosen) >= cfg.best_k:
            break
        if any(_same_view(candidate, kept, cfg) for kept in chosen):
            continue
        chosen.append(candidate)
    return chosen


def _dominant_chain(pool: list[PlateObservation], cfg: SelectionConfig) -> list[PlateObservation]:
    """Restrict the pool to ONE physical plate.

    Observations carry a chain id (see FrameProcessor._pick_candidate): a run of
    looks at what is physically the same plate. A track in dense traffic can
    hold several chains, i.e. several different vehicles' plates, and the
    diversity rule below would otherwise pick them ON PURPOSE (a different
    car's plate looks maximally different from this one's). So pick the chain
    with the most evidence, scored as the sum of its best ``best_k`` qualities,
    which rewards both more views and better ones, and select only from it.

    Chain 0 means "unknown" and every observation made without chains is in
    it, so a pool with no chain information passes through untouched.
    """
    chains: dict[int, list[PlateObservation]] = defaultdict(list)
    for observation in pool:
        chains[observation.group].append(observation)
    if len(chains) <= 1:
        return pool

    def evidence(members: list[PlateObservation]) -> float:
        return sum(sorted((o.quality for o in members), reverse=True)[: cfg.best_k])

    return max(chains.values(), key=evidence)


def _same_view(a: PlateObservation, b: PlateObservation, cfg: SelectionConfig) -> bool:
    if abs(a.frame_idx - b.frame_idx) < cfg.min_frame_gap:
        return True
    return not crop_hash.is_distinct(a.appearance, b.appearance, cfg.min_hamming)
