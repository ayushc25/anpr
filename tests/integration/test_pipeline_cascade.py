"""End-to-end cascade with stub models.

Proves the wiring — gating, tracking, accumulation, line crossing, validation,
dedupe, draft building — without needing model artifacts or a database. The
stubs return scripted results so the assertions are about the PIPELINE, not
about recognition accuracy.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.app.ai.types import Detection, PlateCandidate, PlateRead
from backend.app.ai.vehicle_tracker.bytetrack import ByteTracker
from backend.app.events.dedupe import DedupeConfig, EventDeduplicator
from backend.app.events.event_builder import EventBuilder
from backend.app.events.multi_frame_validator import MultiFrameValidator, ValidationConfig
from backend.app.video.frame_processor import FrameProcessor, PipelineConfig
from backend.app.video.line_crossing import CrossDirection, VirtualLine
from backend.app.video.roi import RoiPolygon

FRAME_W, FRAME_H = 1280, 720


def blank_frame():
    # Mid-grey with texture, so the quality scorer sees plausible contrast
    # rather than a degenerate all-zero crop.
    rng = np.random.default_rng(7)
    return rng.integers(90, 170, (FRAME_H, FRAME_W, 3), dtype=np.uint8)


class StubVehicleDetector:
    """Reports one car whose box is driven by a scripted track of positions."""

    name = "stub_vehicle"
    input_size = (640, 640)

    def __init__(self, positions):
        self.positions = positions
        self.calls = 0

    def detect(self, frame, roi_offset=(0, 0)):
        if self.calls >= len(self.positions):
            self.calls += 1
            return []
        x, y = self.positions[self.calls]
        self.calls += 1
        return [Detection(bbox=(x, y, x + 260, y + 200), confidence=0.92, class_id=2, class_name="car")]

    def warmup(self, n=2):
        return 0.0

    def close(self):
        pass


class StubPlateDetector:
    name = "stub_plate"
    input_size = (320, 320)

    def __init__(self, confidence=0.88):
        self.confidence = confidence
        self.calls = 0

    def detect(self, image, offset=(0, 0)):
        self.calls += 1
        h, w = image.shape[:2]
        # A plate-shaped box in the lower middle of the vehicle crop.
        x1, y1 = offset[0] + int(w * 0.25), offset[1] + int(h * 0.62)
        return [PlateCandidate(bbox=(x1, y1, x1 + 130, y1 + 34), confidence=self.confidence)]


class StubRecognizer:
    name = "stub_rec"
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    expects_grayscale = True

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def recognize(self, plate_image):
        if not self.script:
            return None
        text, confidence = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if text is None:
            return None
        return PlateRead(
            text=text, confidence=confidence,
            per_char_confidence=[confidence] * len(text), raw_text=text,
        )

    def close(self):
        pass


def build_processor(positions, script, roi=None, line=None, cfg=None):
    return FrameProcessor(
        detector=StubVehicleDetector(positions),
        tracker=ByteTracker(min_hits=2, track_buffer=4),
        plate_detector=StubPlateDetector(),
        recognizer=StubRecognizer(script),
        roi=roi if roi is not None else RoiPolygon.from_config([]),
        line=line,
        cfg=cfg or PipelineConfig(detect_interval=1, plate_interval=1, min_plate_quality=0.0),
        camera_id=1,
    )


def drive(processor, n_frames, start_ts=1000.0, step=0.16):
    """Run n frames and collect everything finalized along the way."""
    frame = blank_frame()
    finalized = []
    for i in range(n_frames):
        result = processor.process(frame, frame_idx=i + 1, ts=start_ts + i * step)
        finalized.extend(result.finalized)
    return finalized


class TestCascade:
    def test_reads_accumulate_on_one_track(self):
        positions = [(300, 200 + i * 30) for i in range(8)]
        processor = build_processor(positions, [("UP32AB1234", 0.9)])
        drive(processor, 8)

        states = list(processor.states.values())
        assert len(states) == 1, "one vehicle must produce exactly one track state"
        assert states[0].read_count >= 4
        assert {r.text for r in states[0].reads} == {"UP32AB1234"}

    def test_track_locking_stops_the_plate_stages(self):
        positions = [(300, 200 + i * 25) for i in range(14)]
        processor = build_processor(
            positions, [("UP32AB1234", 0.95)],
            cfg=PipelineConfig(detect_interval=1, plate_interval=1, min_plate_quality=0.0,
                               lock_min_reads=4, lock_min_support=0.9),
        )
        drive(processor, 14)

        state = next(iter(processor.states.values()))
        assert state.locked
        # Once locked, the plate detector must stop being called for it.
        assert processor.plate_detector.calls < 14

    def test_a_vehicle_outside_the_roi_is_never_read(self):
        roi = RoiPolygon.from_config([(0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.0, 0.3)])
        positions = [(900, 500 + i * 10) for i in range(8)]   # far outside the ROI
        processor = build_processor(positions, [("UP32AB1234", 0.9)], roi=roi)
        drive(processor, 8)
        assert processor.plate_detector.calls == 0

    def test_small_vehicles_are_skipped(self):
        cfg = PipelineConfig(detect_interval=1, plate_interval=1, min_vehicle_area_ratio=0.9)
        processor = build_processor([(300, 200 + i * 20) for i in range(6)], [("UP32AB1234", 0.9)], cfg=cfg)
        drive(processor, 6)
        assert processor.plate_detector.calls == 0

    def test_detect_interval_skips_the_detector_but_not_the_tracker(self):
        cfg = PipelineConfig(detect_interval=3, plate_interval=1, min_plate_quality=0.0)
        positions = [(300, 200 + i * 20) for i in range(12)]
        processor = build_processor(positions, [("UP32AB1234", 0.9)], cfg=cfg)
        drive(processor, 12)
        assert processor.detector.calls == 4          # 12 / 3
        assert processor.tracker is not None


class TestFinalization:
    def test_line_crossing_finalizes_immediately(self):
        line = VirtualLine.from_config([(0.0, 0.5), (1.0, 0.5)], forward="in")
        positions = [(500, 100 + i * 90) for i in range(9)]   # drives down through y=360
        processor = build_processor(positions, [("UP32AB1234", 0.92)], line=line)
        finalized = drive(processor, 9)

        assert len(finalized) == 1
        assert finalized[0].finalize_reason == "line_crossing"
        assert finalized[0].direction in (CrossDirection.IN, CrossDirection.OUT)

    def test_track_retirement_finalizes(self):
        positions = [(300, 200 + i * 25) for i in range(6)]
        processor = build_processor(positions, [("UP32AB1234", 0.9)])
        finalized = drive(processor, 20)   # detector runs dry, the track retires

        assert len(finalized) == 1
        assert finalized[0].finalize_reason == "track_retired"

    def test_flush_emits_in_flight_tracks(self):
        positions = [(300, 200 + i * 25) for i in range(6)]
        processor = build_processor(positions, [("UP32AB1234", 0.9)])
        drive(processor, 6)
        pending = processor.flush()
        assert len(pending) == 1
        assert pending[0].finalize_reason == "shutdown"
        assert processor.states == {}


class TestFullPathToDraft:
    def test_noisy_reads_produce_one_correct_event(self):
        """The whole point of the system, in one test: several frames, one of
        them misread, one event with the right plate.

        The trigger line sits at y=0.85 rather than mid-frame. That is where a
        real gate puts it — at the boom, near the camera — and it matters here
        because line crossing freezes the ballot: a line across the middle of
        the frame finalizes the track 0.8 s after it first appears, which is
        fewer reads than a gate ever actually gets and fewer than
        ``min_reads`` requires once ``min_track_age`` has excluded the
        unstable first frames.
        """
        line = VirtualLine.from_config([(0.0, 0.85), (1.0, 0.85)], forward="in")
        positions = [(500, 100 + i * 90) for i in range(9)]
        script = [
            ("UP32AB1234", 0.91),
            ("UP32AB1234", 0.88),
            ("UP32A81234", 0.62),   # structurally impossible: pos 5 must be alpha
            ("UP32AB1234", 0.93),
            ("UP32AB1234", 0.90),
        ]
        processor = build_processor(positions, script, line=line)
        finalized = drive(processor, 9)
        assert finalized

        state = finalized[0]
        final = MultiFrameValidator(ValidationConfig(min_reads=3)).validate(state)
        assert final is not None
        assert final.text == "UP32AB1234"
        assert final.grammar_valid

        draft = EventBuilder(camera_id=1).build(state, final, state.direction)
        assert draft.plate_number == "UP32AB1234"
        assert draft.direction in ("in", "out")
        assert draft.event_uid
        assert draft.read_count >= 3
        assert draft.reads, "evidence trail must be attached"
        assert draft.vehicle_image_b64, "an event needs a vehicle image"

    def test_dedupe_suppresses_a_re_detected_vehicle(self):
        dedupe = EventDeduplicator(DedupeConfig(same_direction_seconds=25.0))
        assert dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1000.0)
        assert not dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1004.0)

    def test_unreadable_vehicle_produces_no_event_but_keeps_an_image(self):
        positions = [(300, 200 + i * 25) for i in range(6)]
        processor = build_processor(positions, [(None, 0.0)])
        finalized = drive(processor, 20)

        assert len(finalized) == 1
        state = finalized[0]
        assert state.read_count == 0
        assert MultiFrameValidator().validate(state) is None
        assert state.best_vehicle_crop is not None


class TestStats:
    def test_timings_are_reported(self):
        processor = build_processor([(300, 200 + i * 25) for i in range(4)], [("UP32AB1234", 0.9)])
        frame = blank_frame()
        result = processor.process(frame, frame_idx=1, ts=1000.0)
        assert result.stats.timing.total >= 0
        assert result.stats.detector_ran
