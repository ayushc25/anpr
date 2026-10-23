"""The deferred-VLM flow, end to end, with a fake VLM.

Covers what the pipeline promises rather than what the model can read:

  * deferred mode never recognizes inside the frame loop
  * a finalized vehicle gets its best, distinct crops submitted
  * the result is attached to the pending vehicle when it arrives, and fused,
    validated and classified by the EXISTING validator
  * every vehicle ends as exactly one A, B or C — through overflow, timeouts,
    a dead service, an unreadable plate and a verification that says no
  * verification can only downgrade
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.app.ai.plate_recognizer.grammar_decode import GrammarDecodeConfig
from backend.app.ai.types import PlateObservation
from backend.app.ai.vehicle_tracker.bytetrack import ByteTracker
from backend.app.events.multi_frame_validator import MultiFrameValidator, RecognitionState, ValidationConfig
from backend.app.events.track_state import TrackState
from backend.app.events.verification import VerificationConfig
from backend.app.video.frame_processor import FrameProcessor, PipelineConfig
from backend.app.video.roi import RoiPolygon
from backend.app.vlm.client import VlmClient
from backend.app.vlm.config import VlmConfig
from backend.app.vlm.protocol import (
    KIND_RECOGNIZE,
    KIND_VERIFY,
    STATUS_ERROR,
    STATUS_OVERFLOW,
    STATUS_UNAVAILABLE,
    VlmRead,
    VlmResult,
)
from backend.app.workers.deferred import DeferredFinalizer
from tests.integration.test_pipeline_cascade import StubPlateDetector, StubVehicleDetector, drive


# ---------------------------------------------------------------- fakes -----
class FakeVlmClient(VlmClient):
    """Answers jobs from a responder on the next poll.

    ``responder(job)`` returns a VlmResult, or None to withhold the answer
    (a hung service). ``accept`` False refuses every submission.
    """

    def __init__(self, responder=None, accept=True):
        self.responder = responder
        self.accept = accept
        self.jobs = []
        self._outstanding = []

    def submit(self, job):
        if not self.accept:
            return False
        self.jobs.append(job)
        self._outstanding.append(job)
        return True

    def poll(self):
        out, keep = [], []
        for job in self._outstanding:
            result = self.responder(job) if self.responder else None
            (out if result is not None else keep).append(result if result is not None else job)
        self._outstanding = keep
        return out

    @property
    def depth(self):
        return len(self._outstanding)

    def of_kind(self, kind):
        return [j for j in self.jobs if j.kind == kind]


def reads_for(job, texts, confidence=0.99):
    out = []
    for index, crop_id in enumerate(job.crop_ids):
        text = texts[index % len(texts)]
        out.append(
            VlmRead(
                crop_id=crop_id, text=text or "", raw_text=text or "",
                confidence=confidence if text else 0.0,
                per_char_confidence=[confidence] * len(text or ""),
            )
        )
    return out


def recognizer_says(texts, verify=None, confidence=0.99):
    """Responder: recognize jobs read `texts` round-robin; verify jobs answer `verify`."""

    def respond(job):
        if job.kind == KIND_RECOGNIZE:
            return VlmResult(job.job_id, job.kind, reads=reads_for(job, texts, confidence))
        return VlmResult(job.job_id, job.kind, answers=[verify] * len(job.crops))

    return respond


class Emissions:
    def __init__(self):
        self.items = []

    def __call__(self, state, final, result, crops, wall):
        self.items.append((state, final, result, crops, wall))

    @property
    def classes(self):
        return [r.value for _, _, r, _, _ in self.items]

    @property
    def last(self):
        return self.items[-1]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_state(n=7, quality=0.9, attempts=6, track_id=1):
    """A finalized track with `n` mutually-distinct retained observations."""
    state = TrackState(track_id=track_id, camera_id=1)
    state.plate_attempts = attempts
    for i in range(n):
        crop = np.full((40, 140, 3), 20 + i * 25, dtype=np.uint8)
        state.note_observation(
            PlateObservation(
                frame_idx=10 + i * 4, ts=100.0 + i, width=140, height=40, quality=quality - i * 0.01,
                det_confidence=0.9, crop=crop, appearance=_distinct_hash(i),
            ),
            keep=8,
        )
    state.best_vehicle_crop = np.full((200, 300, 3), 90, dtype=np.uint8)
    return state


def _distinct_hash(i: int) -> int:
    return 0xFF << (8 * (i % 8)) | (i // 8)


def finalizer_for(client, emissions, clock=None, **cfg_overrides):
    cfg_overrides.setdefault("min_quality", 0.0)
    cfg_overrides.setdefault("min_width", 0)
    cfg = VlmConfig(**cfg_overrides)
    validator = MultiFrameValidator(ValidationConfig(drop_low_confidence=False))
    return DeferredFinalizer(
        camera_id=1, client=client, validator=validator, cfg=cfg,
        grammar_cfg=GrammarDecodeConfig(), on_result=emissions, clock=clock or Clock(),
    )


# ------------------------------------------------- deferred frame loop ------
class TestDeferredFrameLoop:
    def _processor(self, positions):
        return FrameProcessor(
            detector=StubVehicleDetector(positions),
            tracker=ByteTracker(min_hits=2, track_buffer=4),
            plate_detector=StubPlateDetector(),
            recognizer=None,                                   # no model in this process
            roi=RoiPolygon.from_config([]),
            cfg=PipelineConfig(
                detect_interval=1, plate_interval=1, min_plate_quality=0.0,
                recognition_mode="deferred_vlm",
            ),
            camera_id=1,
        )

    def test_observations_are_banked_and_nothing_is_read(self):
        processor = self._processor([(300, 200 + i * 25) for i in range(8)])
        drive(processor, 8)
        state = next(iter(processor.states.values()))
        assert processor.plate_detector.calls > 0              # plates were localized...
        assert state.observation_count > 0 and state.plate_observations
        assert state.ocr_calls == 0 and state.reads == []      # ...and never read
        assert state.plate_attempts > 0

    def test_a_retired_track_is_handed_over_with_its_crops(self):
        processor = self._processor([(300, 200 + i * 25) for i in range(6)])
        finalized = drive(processor, 20)
        assert len(finalized) == 1
        assert finalized[0].finalize_reason == "track_retired"
        assert any(o.crop is not None for o in finalized[0].plate_observations)

    def test_retains_enough_observations_for_the_selector(self):
        cfg = PipelineConfig(recognition_mode="deferred_vlm")
        assert max(cfg.ocr.keep_observations, cfg.deferred_keep_observations) >= VlmConfig().best_k

    def test_an_inline_pipeline_still_requires_a_recognizer(self):
        with pytest.raises(ValueError):
            FrameProcessor(
                detector=StubVehicleDetector([]), tracker=ByteTracker(), plate_detector=StubPlateDetector(),
                recognizer=None, roi=RoiPolygon.from_config([]),
            )

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            FrameProcessor(
                detector=StubVehicleDetector([]), tracker=ByteTracker(), plate_detector=StubPlateDetector(),
                recognizer=None, roi=RoiPolygon.from_config([]),
                cfg=PipelineConfig(recognition_mode="telepathy"),
            )


# --------------------------------------------- selection and submission -----
class TestSubmission:
    def test_submits_the_best_distinct_crops_only(self):
        client, out = FakeVlmClient(), Emissions()
        finalizer_for(client, out, best_k=5).begin(make_state(n=8), wall=1.0)
        (job,) = client.of_kind(KIND_RECOGNIZE)
        assert len(job.crops) == 5
        assert out.items == []                                  # nothing emitted until the VLM answers

    def test_the_early_stop_setting_reaches_the_job(self):
        client, out = FakeVlmClient(), Emissions()
        finalizer_for(client, out, early_stop_agree=3).begin(make_state(), wall=1.0)
        assert client.of_kind(KIND_RECOGNIZE)[0].stop_after_agree == 3

    def test_a_vehicle_that_never_reached_the_read_zone_is_not_an_event(self):
        client, out = FakeVlmClient(), Emissions()
        state = TrackState(track_id=3, camera_id=1)             # no plate attempt, no observation
        finalizer_for(client, out).begin(state, wall=1.0)
        assert client.jobs == [] and out.items == []

    def test_polling_is_a_noop_with_nothing_pending(self):
        finalizer_for(FakeVlmClient(), Emissions()).poll()


# ---------------------------------------- pending finalization + fusion -----
class TestPendingFinalization:
    def test_result_attaches_to_the_pending_vehicle_and_is_class_a(self):
        client = FakeVlmClient(recognizer_says(["UP32AB1234"]))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        state = make_state()
        finalizer.begin(state, wall=5.0)
        assert finalizer.pending_count == 1 and out.items == []

        finalizer.poll()
        assert finalizer.pending_count == 0
        assert out.classes == ["A"]
        emitted_state, final, result, crops, wall = out.last
        assert emitted_state is state and wall == 5.0
        assert final.text == "UP32AB1234" and final.state is RecognitionState.CONFIRMED
        assert len(state.reads) == 6                            # every read joined the track
        assert crops == []                                      # an A needs no review crops
        assert client.of_kind(KIND_VERIFY) == []                # a clean A is not second-guessed

    def test_one_misread_among_agreeing_frames_is_outvoted(self):
        """UP32A81234 on one frame, UP32AB1234 on the rest — the existing
        fusion and grammar logic settles it; the VLM path adds no new rules."""
        texts = ["UP32AB1234", "UP32AB1234", "UP32A81234", "UP32AB1234", "UP32AB1234", "UP32AB1234"]
        client = FakeVlmClient(recognizer_says(texts, verify=0.97))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(), wall=1.0)
        finalizer.poll()
        finalizer.poll()
        _, final, _, _, _ = out.last
        assert final.text == "UP32AB1234"
        assert out.classes == ["A"]
        # The misread was repaired by the existing per-read grammar cleanup, and
        # that correction is what earned the result its verification.
        assert "grammar_correction" in final.verification["triggers"]
        assert final.verification["p_yes"] == pytest.approx(0.97)

    def test_too_few_distinct_views_cannot_reach_class_a(self):
        """A bar, knowingly kept: `validation.min_reads` (3) counts DISTINCT
        usable reads, and the selector refuses to pad with near-duplicates. A
        vehicle that produced one or two different-looking crops therefore ends
        as class B (UNRESOLVED), however confident the model was about them.
        If real footage shows this firing on vehicles that should be A, the
        knob is `validation.min_reads`, not a bug here."""
        client = FakeVlmClient(recognizer_says(["UP32AB1234"]))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(n=2), wall=1.0)
        finalizer.poll()
        _, final, result, crops, _ = out.last
        assert out.classes == ["B"] and final.state is RecognitionState.UNRESOLVED
        assert final.text == "UP32AB1234"                      # the best guess is still reported
        assert len(crops) == 2                                  # and the crops are kept for a human

    def test_two_vehicles_are_kept_apart(self):
        client = FakeVlmClient(lambda job: VlmResult(
            job.job_id, job.kind,
            reads=reads_for(job, ["UP32AB1234" if job.track_id == 1 else "DL8CAF5010"]),
        ))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(track_id=1), wall=1.0)
        finalizer.begin(make_state(track_id=2), wall=2.0)
        assert finalizer.pending_count == 2
        finalizer.poll()
        by_track = {s.track_id: f.text for s, f, *_ in out.items}
        assert by_track == {1: "UP32AB1234", 2: "DL8CAF5010"}

    def test_a_held_answer_leaves_the_vehicle_pending_not_lost(self):
        client = FakeVlmClient(responder=lambda job: None)      # the model is thinking
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(), wall=1.0)
        for _ in range(5):
            finalizer.poll()
        assert finalizer.pending_count == 1 and out.items == []


# ----------------------------------------------------- queue overflow -------
class TestOverflow:
    def test_refused_submission_becomes_class_b_with_the_best_crops(self):
        out = Emissions()
        finalizer = finalizer_for(FakeVlmClient(accept=False), out, best_k=5)
        finalizer.begin(make_state(n=8), wall=1.0)
        assert out.classes == ["B"]
        _, final, result, crops, _ = out.last
        assert final is None and result.reason == "vlm_queue_overflow"
        assert len(crops) == 5                                  # saved for a human, not discarded
        assert finalizer.stats.overflowed == 1

    def test_service_side_overflow_becomes_class_b(self):
        client = FakeVlmClient(lambda job: VlmResult(job.job_id, job.kind, status=STATUS_OVERFLOW))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(), wall=1.0)
        finalizer.poll()
        assert out.classes == ["B"] and out.last[2].reason == "vlm_queue_overflow"
        assert len(out.last[3]) > 0

    def test_no_silent_fallback_to_another_recognizer(self):
        """Overflow must be visible as review, not quietly read some other way."""
        out = Emissions()
        finalizer_for(FakeVlmClient(accept=False), out).begin(make_state(), wall=1.0)
        assert out.last[1] is None                              # no plate string was invented


# ------------------------------------------------ failures and timeouts -----
class TestFailures:
    def test_unavailable_service_is_class_b(self):
        client = FakeVlmClient(lambda job: VlmResult(job.job_id, job.kind, status=STATUS_UNAVAILABLE))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(), wall=1.0)
        finalizer.poll()
        assert out.classes == ["B"] and out.last[2].reason == "vlm_unavailable"

    def test_service_error_is_class_b(self):
        client = FakeVlmClient(lambda job: VlmResult(job.job_id, job.kind, status=STATUS_ERROR, error="boom"))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(), wall=1.0)
        finalizer.poll()
        assert out.classes == ["B"]

    def test_a_hung_service_times_out_to_class_b(self):
        clock, out = Clock(), Emissions()
        finalizer = finalizer_for(FakeVlmClient(lambda job: None), out, clock=clock, job_timeout_s=30)
        finalizer.begin(make_state(), wall=1.0)
        clock.now += 29
        finalizer.poll()
        assert out.items == []
        clock.now += 2
        finalizer.poll()
        assert out.classes == ["B"] and out.last[2].reason == "vlm_timeout"
        assert len(out.last[3]) > 0

    def test_a_late_answer_does_not_emit_a_second_event(self):
        clock, out = Clock(), Emissions()
        answers = {}
        client = FakeVlmClient(lambda job: answers.get(job.job_id))
        finalizer = finalizer_for(client, out, clock=clock, job_timeout_s=10)
        finalizer.begin(make_state(), wall=1.0)
        (job,) = client.jobs
        clock.now += 11
        finalizer.poll()
        assert out.classes == ["B"]
        # ...and now the model finally answers.
        client._outstanding.append(job)
        answers[job.job_id] = VlmResult(job.job_id, job.kind, reads=reads_for(job, ["UP32AB1234"]))
        finalizer.poll()
        assert out.classes == ["B"]                             # still exactly one emission

    def test_every_crop_unreadable_is_class_b_not_c(self):
        """The plate IS in the pictures; nobody could read it. That is a human's job."""
        client = FakeVlmClient(recognizer_says([None]))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(), wall=1.0)
        finalizer.poll()
        assert out.classes == ["B"] and out.last[2].reason == "vlm_unreadable"
        assert out.last[1] is None and len(out.last[3]) > 0

    def test_a_bug_while_finalizing_still_yields_one_event(self):
        client = FakeVlmClient(recognizer_says(["UP32AB1234"]))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.validator.validate = lambda state: (_ for _ in ()).throw(RuntimeError("bug"))
        finalizer.begin(make_state(), wall=1.0)
        finalizer.poll()
        assert out.classes == ["B"]

    def test_shutdown_drain_emits_whatever_is_still_pending(self):
        clock, out = Clock(), Emissions()
        finalizer = finalizer_for(FakeVlmClient(lambda job: None), out, clock=clock)

        def tick(seconds):
            clock.now += seconds

        finalizer.begin(make_state(), wall=1.0)
        finalizer.drain(wait_s=5, sleep=tick)
        assert out.classes == ["B"] and finalizer.pending_count == 0


# ------------------------------------------------------- class C ------------
class TestClassC:
    def test_vehicle_in_the_read_zone_with_no_observation_is_class_c(self):
        out = Emissions()
        state = TrackState(track_id=4, camera_id=1)
        state.plate_attempts = 9                                # looked nine times, found nothing
        finalizer_for(FakeVlmClient(), out).begin(state, wall=1.0)
        assert out.classes == ["C"]
        assert out.last[1] is None and out.last[2].reason == "no_usable_plate_observation"

    def test_only_unusable_observations_is_class_c(self):
        out = Emissions()
        state = make_state(n=4, quality=0.2)
        finalizer_for(FakeVlmClient(), out, min_quality=0.35).begin(state, wall=1.0)
        assert out.classes == ["C"]

    def test_class_c_does_not_call_the_vlm(self):
        client = FakeVlmClient()
        state = TrackState(track_id=4, camera_id=1)
        state.plate_attempts = 3
        finalizer_for(client, Emissions()).begin(state, wall=1.0)
        assert client.jobs == []


# -------------------------------------------------- verification -----------
class TestVerification:
    # Quality 0.45 is below verification.low_quality_below (0.50), so a result
    # that would otherwise be a clean A is triggered for a second opinion.
    def _run(self, verify, **cfg):
        client = FakeVlmClient(recognizer_says(["UP32AB1234"], verify=verify))
        out = Emissions()
        finalizer = finalizer_for(client, out, **cfg)
        finalizer.begin(make_state(quality=0.45), wall=1.0)
        finalizer.poll()          # recognition result -> verification submitted
        finalizer.poll()          # verification result -> emitted
        return client, out, finalizer

    def test_ambiguous_result_is_verified_and_a_yes_keeps_class_a(self):
        client, out, _ = self._run(verify=0.98)
        assert len(client.of_kind(KIND_VERIFY)) == 1
        assert client.of_kind(KIND_VERIFY)[0].candidate == "UP32AB1234"
        assert out.classes == ["A"]
        assert out.last[1].verification["p_yes"] == pytest.approx(0.98)

    def test_a_no_downgrades_a_to_b_and_saves_the_crops(self):
        _, out, finalizer = self._run(verify=0.04)
        assert out.classes == ["B"]
        _, final, result, crops, _ = out.last
        assert final.state is RecognitionState.PROBABLE
        assert result.reason == "verification_rejected"
        assert len(crops) > 0
        assert finalizer.stats.downgrades == 1

    def test_a_clean_result_skips_verification_entirely(self):
        client = FakeVlmClient(recognizer_says(["UP32AB1234"], verify=0.0))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(quality=0.9), wall=1.0)
        finalizer.poll()
        assert client.of_kind(KIND_VERIFY) == [] and out.classes == ["A"]

    def test_verification_disabled_means_no_second_call(self):
        client = FakeVlmClient(recognizer_says(["UP32AB1234"], verify=0.0))
        out = Emissions()
        finalizer = finalizer_for(client, out, verification=VerificationConfig(enabled=False))
        finalizer.begin(make_state(quality=0.45), wall=1.0)
        finalizer.poll()
        assert client.of_kind(KIND_VERIFY) == [] and out.classes == ["A"]

    def test_a_yes_can_never_turn_a_b_into_an_a(self):
        """Two competing readings -> PROBABLE -> class B. The verifier is asked,
        answers an emphatic yes, and the result must stay B."""
        texts = ["UP32AB1234"] * 4 + ["UP32AB1284"] * 2
        client = FakeVlmClient(recognizer_says(texts, verify=1.0, confidence=0.9))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(), wall=1.0)
        for _ in range(3):
            finalizer.poll()
        assert len(client.of_kind(KIND_VERIFY)) == 1            # it WAS asked
        _, final, result, _, _ = out.last
        assert out.classes == ["B"]
        assert final.state is RecognitionState.PROBABLE
        assert final.verification["p_yes"] == 1.0               # recorded for the reviewer...
        assert final.verification["downgraded"] is False        # ...and changed nothing
        assert result.reason == "ambiguous"

    def test_an_unresolved_result_is_not_even_asked(self):
        texts = ["UP32AB1234", "UP32AB6234", "UP32AB1284", "UP32AB1239", "UP32AB1734", "UP32AB1244"]
        client = FakeVlmClient(recognizer_says(texts, verify=1.0, confidence=0.55))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(), wall=1.0)
        for _ in range(3):
            finalizer.poll()
        assert out.last[1].state is RecognitionState.UNRESOLVED
        assert client.of_kind(KIND_VERIFY) == [] and out.classes == ["B"]

    def test_unavailable_verification_downgrades_a_triggered_a(self):
        client = FakeVlmClient(recognizer_says(["UP32AB1234"]))
        out = Emissions()
        finalizer = finalizer_for(client, out)
        finalizer.begin(make_state(quality=0.45), wall=1.0)
        client.accept = False                                   # the queue fills before the verification
        finalizer.poll()
        assert out.classes == ["B"] and out.last[2].reason == "verification_unavailable"

    def test_unavailable_verification_can_keep_a(self):
        client = FakeVlmClient(recognizer_says(["UP32AB1234"]))
        out = Emissions()
        finalizer = finalizer_for(client, out, verification=VerificationConfig(on_unavailable="keep"))
        finalizer.begin(make_state(quality=0.45), wall=1.0)
        client.accept = False
        finalizer.poll()
        assert out.classes == ["A"]

    def test_verification_that_never_answers_still_emits_once(self):
        clock, out = Clock(), Emissions()

        def respond(job):
            if job.kind == KIND_RECOGNIZE:
                return VlmResult(job.job_id, job.kind, reads=reads_for(job, ["UP32AB1234"]))
            return None                                         # verification hangs

        finalizer = finalizer_for(FakeVlmClient(respond), out, clock=clock, job_timeout_s=10)
        finalizer.begin(make_state(quality=0.45), wall=1.0)
        finalizer.poll()
        assert out.items == []
        clock.now += 11
        finalizer.poll()
        assert out.classes == ["B"]


# ------------------------------------------- plate continuity (one car) -----
class TwoPlateDetector:
    """Two plates inside one tracked box, as in dense traffic. Their
    confidences alternate frame to frame, so 'most confident wins' would
    flip between two different cars."""

    name = "two_plate"
    input_size = (320, 320)

    def __init__(self, jump_after=None):
        self.calls = 0
        self.jump_after = jump_after

    def detect(self, image, offset=(0, 0)):
        from backend.app.ai.types import PlateCandidate

        self.calls += 1
        h, w = image.shape[:2]
        y = offset[1] + int(h * 0.62)
        a = PlateCandidate(bbox=(offset[0] + int(w * 0.04), y, offset[0] + int(w * 0.04) + 70, y + 20),
                           confidence=0.95 if self.calls % 2 else 0.60)
        b = PlateCandidate(bbox=(offset[0] + int(w * 0.62), y, offset[0] + int(w * 0.62) + 70, y + 20),
                           confidence=0.60 if self.calls % 2 else 0.95)
        return [a, b]

    def warmup(self, n=2):
        return 0.0

    def close(self):
        pass


class TestPlateContinuity:
    def _drive(self, detector, frames=10):
        processor = FrameProcessor(
            detector=StubVehicleDetector([(300, 200 + i * 5) for i in range(frames)]),
            tracker=ByteTracker(min_hits=2, track_buffer=4),
            plate_detector=detector,
            recognizer=None,
            roi=RoiPolygon.from_config([]),
            cfg=PipelineConfig(
                detect_interval=1, plate_interval=1, min_plate_quality=0.0,
                recognition_mode="deferred_vlm",
            ),
            camera_id=1,
        )
        drive(processor, frames)
        return next(iter(processor.states.values()))

    def test_one_track_follows_one_plate_even_when_another_scores_higher(self):
        state = self._drive(TwoPlateDetector())
        assert state.plate_group == 1                           # a single chain, never flipped
        assert {o.group for o in state.plate_observations} == {1}
        xs = {o.bbox[0] for o in state.plate_observations}
        assert max(xs) - min(xs) < 40                           # all on the same plate, not both

    def test_a_plate_that_jumps_far_starts_a_new_chain(self):
        class Jumper(TwoPlateDetector):
            def detect(self, image, offset=(0, 0)):
                candidates = super().detect(image, offset)
                # From the 6th call only the OTHER plate is visible: the first car has left.
                return [candidates[1]] if self.calls > 5 else [candidates[0]]

        state = self._drive(Jumper(), frames=12)
        assert state.plate_group == 2
        assert {o.group for o in state.plate_observations} <= {1, 2}


class TestCropMargin:
    def _crop_width(self, mode, **cfg):
        processor = FrameProcessor(
            detector=StubVehicleDetector([(300, 200 + i * 5) for i in range(6)]),
            tracker=ByteTracker(min_hits=2, track_buffer=4),
            plate_detector=StubPlateDetector(),
            recognizer=None if mode == "deferred_vlm" else type("R", (), {"close": lambda s: None})(),
            roi=RoiPolygon.from_config([]),
            cfg=PipelineConfig(detect_interval=1, plate_interval=1, min_plate_quality=0.0,
                               recognition_mode=mode, **cfg),
            camera_id=1,
        )
        drive(processor, 6)
        state = next(iter(processor.states.values()))
        return max(o.width for o in state.plate_observations)

    def test_deferred_mode_crops_with_a_wider_margin_than_inline(self):
        # The stub plate is 130 px wide: 0.06 pad -> ~145 px, 0.15 pad -> ~169 px.
        deferred = self._crop_width("deferred_vlm")
        inline_pad = self._crop_width("deferred_vlm", deferred_plate_crop_pad=0.06)
        assert deferred > inline_pad + 15
        assert deferred >= 130 * 1.25

    def test_inline_mode_keeps_its_own_pad(self):
        assert PipelineConfig().plate_crop_pad == 0.06
