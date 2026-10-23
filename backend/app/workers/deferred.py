"""Deferred finalization: a vehicle's pass through the gate, resolved later.

In ``deferred_vlm`` mode the frame loop reads nothing. By the time a track is
finalized it has banked a handful of plate observations and nothing else. This
class turns those into one result, over however many seconds the VLM takes:

    begin()    track finalized -> pick the best crops -> submit a job
    poll()     results arrive  -> reads join the track -> validate
                               -> maybe verify -> classify -> emit
    drain()    shutdown        -> wait a bounded time, then emit what is left

It owns ORCHESTRATION only. Recognition is the VLM client's, scoring is the
validator's, the emitted event is the worker's. Taking the three as arguments
is what makes the whole flow testable with a fake client and a capturing
callback, no model, no camera and no database.

THE GUARANTEES THIS CLASS EXISTS TO KEEP
----------------------------------------
* Every finalized vehicle that reached the read zone produces EXACTLY ONE
  emission: A, B or C. Nothing is silently dropped. A refused submission, a
  dead service, a timeout, an unreadable result and an inconclusive vote all
  end in a class B event that says which.
* Nothing here blocks. ``begin`` and ``poll`` do bounded in-memory work; every
  wait happens elsewhere, so the RTSP loop's latency does not depend on how
  long the model takes.
* Memory is bounded. A pending vehicle holds at most ``best_k`` crops, and the
  client refuses submissions past its bound, so a slow model produces class B
  results rather than an ever-growing pile of crops.
* A late answer for a vehicle already given up on is discarded: its event has
  been emitted (as B / timeout), and emitting it again would make two events
  for one vehicle.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..ai.plate_recognizer.grammar_decode import GrammarDecodeConfig
from ..ai.types import PlateObservation, PlateRead
from ..events import verification as verification_policy
from ..events.multi_frame_validator import FinalPlate, MultiFrameValidator
from ..events.result_class import (
    REASON_NO_CONSENSUS,
    REASON_QUEUE_OVERFLOW,
    REASON_VLM_TIMEOUT,
    REASON_VLM_UNAVAILABLE,
    REASON_VLM_UNREADABLE,
    ResultInfo,
    classify,
    no_plate,
    review,
)
from ..events.track_state import TrackState
from ..events.verification import VerificationOutcome
from ..video.best_frames import select_best
from ..video.frame_processor import quality_of_observation
from ..vlm.client import VlmClient
from ..vlm.config import VlmConfig
from ..vlm.protocol import (
    KIND_RECOGNIZE,
    KIND_VERIFY,
    STATUS_EXPIRED,
    STATUS_OVERFLOW,
    STATUS_TIMEOUT,
    VlmJob,
    VlmResult,
)

logger = logging.getLogger("anpr.worker.deferred")

#: (state, final plate or None, result, crops to save for a reviewer, wall time)
EmitCallback = Callable[[TrackState, Optional[FinalPlate], ResultInfo, list, float], None]

_STAGE_RECOGNIZE = "recognize"
_STAGE_VERIFY = "verify"

_FAILURE_REASONS = {
    STATUS_OVERFLOW: REASON_QUEUE_OVERFLOW,
    STATUS_EXPIRED: REASON_VLM_TIMEOUT,
    STATUS_TIMEOUT: REASON_VLM_TIMEOUT,
}


@dataclass
class _Pending:
    state: TrackState
    stage: str
    job_id: str
    selected: list[PlateObservation]
    wall: float
    deadline: float
    final: Optional[FinalPlate] = None
    why: list[str] = field(default_factory=list)


@dataclass
class DeferredStats:
    submitted: int = 0
    overflowed: int = 0
    completed: int = 0
    timed_out: int = 0
    unavailable: int = 0
    verifications: int = 0
    downgrades: int = 0
    class_a: int = 0
    class_b: int = 0
    class_c: int = 0

    def as_dict(self) -> dict:
        return dict(vars(self))


class DeferredFinalizer:
    def __init__(
        self,
        camera_id: int,
        client: VlmClient,
        validator: MultiFrameValidator,
        cfg: VlmConfig,
        grammar_cfg: GrammarDecodeConfig,
        on_result: EmitCallback,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.camera_id = camera_id
        self.client = client
        self.validator = validator
        self.cfg = cfg
        self.grammar_cfg = grammar_cfg
        self.on_result = on_result
        self.clock = clock
        self.stats = DeferredStats()
        self._selection = cfg.selection()
        self._pending: dict[str, _Pending] = {}

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # -- entry: a track was finalized --------------------------------------
    def begin(self, state: TrackState, wall: float) -> None:
        # A vehicle whose plate was never even looked for (it never reached
        # the read zone) is not an event: nothing was expected of the system
        # there. This is the same rule the inline pipeline has always had.
        if state.plate_attempts == 0 and state.observation_count == 0 and not state.reads:
            return

        selected = select_best(state.plate_observations, self._selection)
        if not selected:
            # The system looked, where a plate should be legible, and got
            # nothing usable. That is a vehicle WITHOUT a visible plate, which
            # is a different fact from there being no vehicle: class C.
            self._emit(
                state, None,
                no_plate(
                    observations=state.observation_count,
                    best_quality=round(state.best_quality, 3),
                    plate_attempts=state.plate_attempts,
                ),
                [], wall,
            )
            return

        job = self._job(KIND_RECOGNIZE, state, [o.crop for o in selected], [o.frame_idx for o in selected])
        job.stop_after_agree = self.cfg.early_stop_agree
        if not self.client.submit(job):
            self.stats.overflowed += 1
            logger.warning(
                "camera %s: VLM queue full; track %s goes to review with %d saved crops",
                self.camera_id, state.track_id, len(selected),
            )
            self._emit(state, None, review(REASON_QUEUE_OVERFLOW, selected=len(selected)), _crops(selected), wall)
            return

        self.stats.submitted += 1
        self._pending[job.job_id] = _Pending(
            state=state, stage=_STAGE_RECOGNIZE, job_id=job.job_id, selected=selected,
            wall=wall, deadline=self.clock() + self.cfg.job_timeout_s,
        )

    # -- called every loop iteration ---------------------------------------
    def poll(self) -> None:
        for result in self.client.poll():
            pending = self._pending.pop(result.job_id, None)
            if pending is None:
                # Answer for a vehicle we already gave up on and emitted.
                logger.debug("camera %s: discarding late result %s", self.camera_id, result.job_id)
                continue
            try:
                self._advance(pending, result)
            except Exception:
                logger.exception("camera %s: finalizing track %s failed", self.camera_id, pending.state.track_id)
                self._emit(
                    pending.state, None, review(REASON_VLM_UNAVAILABLE, error="finalization_error"),
                    _crops(pending.selected), pending.wall,
                )

        now = self.clock()
        for job_id in [j for j, p in self._pending.items() if p.deadline <= now]:
            pending = self._pending.pop(job_id)
            self.stats.timed_out += 1
            logger.warning(
                "camera %s: VLM did not answer track %s within %.0fs",
                self.camera_id, pending.state.track_id, self.cfg.job_timeout_s,
            )
            self._give_up(pending, REASON_VLM_TIMEOUT)

    def drain(self, wait_s: float, sleep: Callable[[float], None] = time.sleep) -> None:
        """Shutdown: give in-flight vehicles a bounded time, then emit the rest as B."""
        deadline = self.clock() + max(0.0, wait_s)
        while self._pending and self.clock() < deadline:
            self.poll()
            if self._pending:
                sleep(0.1)
        self.poll()
        for job_id in list(self._pending):
            pending = self._pending.pop(job_id)
            self._give_up(pending, REASON_VLM_TIMEOUT)

    # -- stages ------------------------------------------------------------
    def _advance(self, pending: _Pending, result: VlmResult) -> None:
        if pending.stage == _STAGE_RECOGNIZE:
            self._on_recognized(pending, result)
        else:
            self._on_verified(pending, result)

    def _on_recognized(self, pending: _Pending, result: VlmResult) -> None:
        state = pending.state
        if not result.ok:
            self._count_failure(result.status)
            self._give_up(pending, _FAILURE_REASONS.get(result.status, REASON_VLM_UNAVAILABLE), result.error)
            return

        by_id = {o.frame_idx: o for o in pending.selected}
        accepted = 0
        for read in result.reads:
            observation = by_id.get(read.crop_id)
            if observation is None or not read.readable:
                continue
            state.add_read(
                PlateRead(
                    text=read.text,
                    confidence=read.confidence,
                    per_char_confidence=list(read.per_char_confidence),
                    raw_text=read.raw_text,
                    per_char_alternatives=[tuple((c, p) for c, p in pos) for pos in read.per_char_alternatives],
                ),
                plate_det_confidence=observation.det_confidence,
                quality=quality_of_observation(observation),
                frame_ts=observation.ts,
                plate_crop=observation.crop,
                grammar_cfg=self.grammar_cfg,
                frame_idx=observation.frame_idx,
                appearance=observation.appearance,
            )
            observation.ocr_ran = True
            state.ocr_calls += 1
            accepted += 1

        self.stats.completed += 1
        if accepted == 0:
            # The model looked at every crop and declined each one. The
            # vehicle and its plate are in the pictures; nobody has read them.
            self._emit(
                state, None,
                review(REASON_VLM_UNREADABLE, crops=len(pending.selected), run_ms=round(result.run_ms)),
                _crops(pending.selected), pending.wall,
            )
            return

        final = self.validator.validate(state)
        if final is None:
            self._emit(state, None, review(REASON_NO_CONSENSUS), _crops(pending.selected), pending.wall)
            return

        pending.final = final
        why = verification_policy.should_verify(final, self.cfg.verification)
        if not why:
            self._finish(pending, final)
            return
        self._request_verification(pending, final, why)

    def _request_verification(self, pending: _Pending, final: FinalPlate, why: list[str]) -> None:
        crops = self._verification_crops(pending, final)
        job = self._job(KIND_VERIFY, pending.state, crops, list(range(len(crops))), candidate=final.text)
        if not self.client.submit(job):
            # Could not ask. For a triggered CONFIRMED result "not checked" is
            # handled by policy (downgrade by default), not guessed here.
            outcome = VerificationOutcome(ran=False, unavailable_reason="queue_full")
            self._apply_verification(pending, final, outcome, why)
            return
        self.stats.verifications += 1
        pending.stage = _STAGE_VERIFY
        pending.job_id = job.job_id
        pending.why = why
        pending.deadline = self.clock() + self.cfg.job_timeout_s
        self._pending[job.job_id] = pending

    def _on_verified(self, pending: _Pending, result: VlmResult) -> None:
        final = pending.final
        if not result.ok:
            outcome = VerificationOutcome(ran=False, unavailable_reason=result.status)
        else:
            answers = [a for a in result.answers if a is not None]
            outcome = VerificationOutcome(
                ran=True,
                p_yes=(sum(answers) / len(answers)) if answers else None,
                answers=list(result.answers),
                unavailable_reason="" if answers else "no_answer",
            )
        self._apply_verification(pending, final, outcome, pending.why)

    def _apply_verification(
        self, pending: _Pending, final: FinalPlate, outcome: VerificationOutcome, why: list[str]
    ) -> None:
        verification_policy.apply(
            final, outcome, self.cfg.verification, why,
            confirm_confidence=self.validator.cfg.confirm_confidence,
        )
        if final.verification.get("downgraded"):
            self.stats.downgrades += 1
        self._finish(pending, final)

    def _finish(self, pending: _Pending, final: FinalPlate) -> None:
        result = classify(final)
        crops = [] if result.value == "A" else _crops(pending.selected)
        self._emit(pending.state, final, result, crops, pending.wall)

    def _give_up(self, pending: _Pending, reason: str, error: str = "") -> None:
        """No answer will come. The vehicle still goes through as class B.

        If the track already has a validated plate (a verification that never
        answered), that plate is kept and merely cannot be class A; otherwise
        there is no string and the event carries the saved crops alone.
        """
        final = pending.final
        if final is not None:
            outcome = VerificationOutcome(ran=False, unavailable_reason=reason)
            self._apply_verification(pending, final, outcome, pending.why)
            return
        self._emit(
            pending.state, None, review(reason, error=error) if error else review(reason),
            _crops(pending.selected), pending.wall,
        )

    # -- helpers -----------------------------------------------------------
    def _verification_crops(self, pending: _Pending, final: FinalPlate) -> list[np.ndarray]:
        """The crop(s) the winning string was actually read from, best first,
        topped up from the other selected crops."""
        crops: list[np.ndarray] = []
        found = pending.state.evidence_for(final.text)
        if found is not None and found.plate_crop is not None:
            crops.append(found.plate_crop)
        for observation in sorted(pending.selected, key=lambda o: o.quality, reverse=True):
            if len(crops) >= self.cfg.verification.crops:
                break
            if observation.crop is not None and not any(observation.crop is c for c in crops):
                crops.append(observation.crop)
        return crops[: self.cfg.verification.crops]

    def _job(self, kind: str, state: TrackState, crops: list, crop_ids: list, candidate: str = "") -> VlmJob:
        return VlmJob(
            job_id=uuid.uuid4().hex,
            kind=kind,
            camera_id=self.camera_id,
            track_id=state.track_id,
            crops=list(crops),
            crop_ids=list(crop_ids),
            candidate=candidate,
        )

    def _count_failure(self, status: str) -> None:
        if status == STATUS_OVERFLOW:
            self.stats.overflowed += 1
        elif status in (STATUS_EXPIRED, STATUS_TIMEOUT):
            self.stats.timed_out += 1
        else:
            self.stats.unavailable += 1

    def _emit(
        self, state: TrackState, final: Optional[FinalPlate], result: ResultInfo, crops: list, wall: float
    ) -> None:
        if result.value == "A":
            self.stats.class_a += 1
        elif result.value == "B":
            self.stats.class_b += 1
        else:
            self.stats.class_c += 1
        self.on_result(state, final, result, crops, wall)


def _crops(observations: list[PlateObservation]) -> list[np.ndarray]:
    return [o.crop for o in observations if o.crop is not None]
