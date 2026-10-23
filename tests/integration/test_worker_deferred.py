"""The real CameraWorker in deferred_vlm mode, with stub models and a fake VLM.

test_deferred_vlm.py covers the finalizer in isolation. This covers the wiring
around it that only the worker owns: that no recognizer is ever built in the
camera process, that a finalized track reaches the VLM, that the answer comes
back through the worker's own emit path into a draft, and that health says so.
"""
from __future__ import annotations

import time

import pytest

from backend.app.ai.inference.threading import ThreadBudget
from backend.app.ai.vehicle_tracker.bytetrack import ByteTracker
from backend.app.events.dedupe import DedupeConfig
from backend.app.events.multi_frame_validator import ValidationConfig
from backend.app.video.frame_processor import PipelineConfig
from backend.app.vlm.config import VlmConfig
from backend.app.workers import camera_worker as cw
from backend.app.workers.camera_worker import CameraConfig, CameraWorker
from backend.app.workers.ingest_client import NullIngestClient
from tests.integration.test_deferred_vlm import FakeVlmClient, recognizer_says
from tests.integration.test_pipeline_cascade import StubPlateDetector, StubVehicleDetector, blank_frame


class PlateStub(StubPlateDetector):
    def warmup(self, n=2):
        return 0.0

    def close(self):
        pass


class StubReader:
    def __init__(self, *args, **kwargs):
        self.stats = type("S", (), {"connected": True, "fps": 6.0, "last_frame_at": 0.0,
                                    "reconnects": 0, "last_error": ""})()

    def start(self): ...
    def stop(self): ...
    def latest_with_timestamp(self):
        return None, 0.0


@pytest.fixture
def build(monkeypatch, tmp_path):
    built = []

    def _build(client, mode="deferred_vlm", positions=None):
        positions = positions or [(300, 200 + i * 25) for i in range(6)]
        monkeypatch.setattr(cw, "build_vehicle_detector", lambda *a, **k: StubVehicleDetector(positions))
        monkeypatch.setattr(cw, "build_tracker", lambda *a, **k: ByteTracker(min_hits=2, track_buffer=4))
        monkeypatch.setattr(cw, "build_plate_detector", lambda *a, **k: PlateStub())

        def no_recognizer(*a, **k):
            built.append("recognizer")
            raise AssertionError("a camera process must not load a recognizer in deferred mode")

        monkeypatch.setattr(cw, "build_plate_recognizer", no_recognizer)
        monkeypatch.setattr(cw, "RtspReader", StubReader)

        ingest = NullIngestClient(tmp_path / "spool")
        worker = CameraWorker(
            camera=CameraConfig(id=1, code="cam1", name="Gate", rtsp_url="rtsp://x", direction="in"),
            pipeline_cfg=PipelineConfig(
                detect_interval=1, plate_interval=1, min_plate_quality=0.0, recognition_mode=mode,
            ),
            # The stub frames are near-identical, so only a view or two survives
            # the diversity filter. This test is about the WIRING; the bar for
            # class A (min_reads) has its own test in test_deferred_vlm.py.
            validation_cfg=ValidationConfig(min_reads=1),
            dedupe_cfg=DedupeConfig(),
            model_cfgs={k: {} for k in ("vehicle_detector", "tracker", "plate_detector", "plate_recognizer")},
            threads=ThreadBudget(intra_op=1),
            ingest=ingest,
            vlm_cfg=VlmConfig(min_quality=0.0, min_width=0),
            vlm_client=client,
        )
        return worker, ingest, built

    return _build


def run_vehicle(worker, frames=20):
    frame = blank_frame()
    t0 = time.time()
    for i in range(frames):
        worker._process(frame, captured_at=t0 + i * 0.16)


def test_no_recognizer_is_loaded_in_the_camera_process(build):
    worker, _, built = build(FakeVlmClient())
    assert worker.recognizer is None and built == []
    assert worker.processor.recognizer is None and worker.processor.deferred


def test_a_finalized_vehicle_is_handed_to_the_vlm_not_emitted(build):
    client = FakeVlmClient(responder=lambda job: None)        # the model has not answered yet
    worker, ingest, _ = build(client)
    run_vehicle(worker)
    assert len(client.jobs) == 1 and ingest.events == []
    assert worker.finalizer.pending_count == 1
    assert worker.health.events_emitted == 0


def test_the_answer_comes_back_as_a_class_a_event(build):
    client = FakeVlmClient(recognizer_says(["UP32AB1234"]))
    worker, ingest, _ = build(client)
    run_vehicle(worker)
    worker.finalizer.poll()

    (draft,) = ingest.events
    assert draft["plate_number"] == "UP32AB1234"
    assert draft["result_class"] == "A"
    assert draft["direction"] == "in"                          # the camera's declared role
    assert draft["vehicle_image_b64"] and draft["plate_image_b64"]
    assert draft["review_crops_b64"] == []
    assert worker.health.events_emitted == 1


def test_overflow_reaches_the_event_log_as_class_b_with_crops(build):
    worker, ingest, _ = build(FakeVlmClient(accept=False))
    run_vehicle(worker)
    (draft,) = ingest.events                                   # immediately: nothing to wait for
    assert draft["plate_number"] is None
    assert (draft["result_class"], draft["result_reason"]) == ("B", "vlm_queue_overflow")
    assert len(draft["review_crops_b64"]) >= 1
    assert draft["disputed"] is True


def test_the_deferred_path_records_into_the_dedupe_window(build):
    client = FakeVlmClient(recognizer_says(["UP32AB1234"]))
    worker, ingest, _ = build(client, positions=[(300, 200 + i * 25) for i in range(6)])
    run_vehicle(worker)
    worker.finalizer.poll()
    assert len(ingest.events) == 1
    # Same plate, same direction, moments later: the event above recorded it in
    # the existing dedupe window, so a second vehicle would be suppressed.
    assert worker.dedupe.check_and_record("UP32AB1234", cw.CrossDirection.IN, time.time()) is False


def test_health_reports_the_vlm_state(build):
    client = FakeVlmClient(responder=lambda job: None)
    worker, _, _ = build(client)
    run_vehicle(worker)
    worker._sync_health()
    vlm = worker.health.as_dict()["vlm"]
    assert vlm["pending"] == 1 and vlm["submitted"] == 1 and vlm["queue_depth"] == 1


def test_shutdown_emits_pending_vehicles_as_class_b(build):
    client = FakeVlmClient(responder=lambda job: None)
    worker, ingest, _ = build(client)
    worker._vlm_cfg = VlmConfig(shutdown_wait_s=0.0, min_quality=0.0, min_width=0)
    run_vehicle(worker)
    assert ingest.events == []
    worker._shutdown()
    assert [e["result_class"] for e in ingest.events] == ["B"]
    assert ingest.events[0]["result_reason"] == "vlm_timeout"


def test_inline_mode_is_untouched(build, monkeypatch):
    """The original cascade must still build its recognizer and ignore the VLM."""

    class Rec:
        name = "stub"
        def warmup(self, n=2): return 0.0
        def recognize(self, image): return None
        def close(self): ...

    monkeypatch.setattr(cw, "build_vehicle_detector", lambda *a, **k: StubVehicleDetector([]))
    monkeypatch.setattr(cw, "build_tracker", lambda *a, **k: ByteTracker())
    monkeypatch.setattr(cw, "build_plate_detector", lambda *a, **k: PlateStub())
    monkeypatch.setattr(cw, "build_plate_recognizer", lambda *a, **k: Rec())
    monkeypatch.setattr(cw, "RtspReader", StubReader)
    worker = CameraWorker(
        camera=CameraConfig(id=1, code="cam1", name="Gate", rtsp_url="rtsp://x"),
        pipeline_cfg=PipelineConfig(recognition_mode="inline"),
        validation_cfg=ValidationConfig(), dedupe_cfg=DedupeConfig(),
        model_cfgs={k: {} for k in ("vehicle_detector", "tracker", "plate_detector", "plate_recognizer")},
        threads=ThreadBudget(intra_op=1),
        ingest=NullIngestClient(),
    )
    assert worker.finalizer is None and worker.recognizer is not None and not worker.deferred
