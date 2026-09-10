"""Read zone, OCR budget and observation retention, driven through the cascade.

Companion to test_ocr_scheduler.py: that file proves the decision table, this
one proves FrameProcessor honours it — that a WAIT really does cost no
recognizer call, that the budget really does stop the plate detector too, and
that nothing accumulates a backlog along the way.

The stubs count calls. Every assertion here is ultimately about a number of
model invocations, because that is what this phase was for.
"""
from __future__ import annotations

import numpy as np

from backend.app.ai.types import Detection, PlateCandidate, PlateRead
from backend.app.ai.vehicle_tracker.bytetrack import ByteTracker
from backend.app.video.frame_processor import FrameProcessor, PipelineConfig
from backend.app.video.ocr_scheduler import OcrPolicy
from backend.app.video.roi import RoiPolygon

FRAME_W, FRAME_H = 1280, 720


def blank_frame():
    """Sharp, well-exposed texture: every crop scores 'excellent', which
    isolates the gating rules from the quality scorer."""
    rng = np.random.default_rng(11)
    return rng.integers(90, 170, (FRAME_H, FRAME_W, 3), dtype=np.uint8)


def soft_frame():
    """A smooth gradient: zero Laplacian variance, so quality is driven almost
    entirely by the resolution term and therefore by plate WIDTH.

    That is what a real approach looks like to the scorer — a distant plate is
    marginal and becomes good as it grows — and it is the only way to exercise
    the approach/quality interaction without a real clip. Measured: ~0.42 at
    80 px (marginal) and ~0.61 at 120 px (good).
    """
    ramp = np.linspace(60, 190, FRAME_W, dtype=np.float32).astype(np.uint8)
    return np.dstack([np.repeat(ramp[None, :], FRAME_H, axis=0)] * 3)


class ScriptedVehicleDetector:
    """Reports vehicles from a per-frame script of boxes."""

    name = "stub_vehicle"
    input_size = (640, 640)

    def __init__(self, frames):
        self.frames = frames
        self.calls = 0

    def detect(self, frame, roi_offset=(0, 0)):
        index = self.calls
        self.calls += 1
        if index >= len(self.frames):
            return []
        return [
            Detection(bbox=box, confidence=0.92, class_id=2, class_name="car")
            for box in self.frames[index]
        ]

    def warmup(self, n=2):
        return 0.0

    def close(self):
        pass


class SizedPlateDetector:
    """Emits a plate whose width is a fixed fraction of the vehicle box, so a
    test can make a plate 'approach' just by growing the vehicle."""

    name = "stub_plate"
    input_size = (320, 320)

    def __init__(self, fraction=0.5):
        self.fraction = fraction
        self.calls = 0

    def detect(self, image, offset=(0, 0)):
        self.calls += 1
        h, w = image.shape[:2]
        plate_w = max(8, int(w * self.fraction))
        plate_h = max(6, int(plate_w / 4.5))
        x1 = offset[0] + (w - plate_w) // 2
        y1 = offset[1] + int(h * 0.65)
        return [PlateCandidate(bbox=(x1, y1, x1 + plate_w, y1 + plate_h), confidence=0.9)]


class CountingRecognizer:
    name = "stub_rec"
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    expects_grayscale = True

    def __init__(self, text="UP32AB1234", confidence=0.93):
        self.text = text
        self.confidence = confidence
        self.calls = 0
        self.widths = []

    def recognize(self, plate_image):
        self.calls += 1
        self.widths.append(plate_image.shape[1])
        return PlateRead(
            text=self.text, confidence=self.confidence,
            per_char_confidence=[self.confidence] * len(self.text), raw_text=self.text,
        )

    def close(self):
        pass


def build(frames, cfg=None, read_zone=None, plate_fraction=0.5):
    return FrameProcessor(
        detector=ScriptedVehicleDetector(frames),
        tracker=ByteTracker(min_hits=2, track_buffer=6),
        plate_detector=SizedPlateDetector(plate_fraction),
        recognizer=CountingRecognizer(),
        roi=RoiPolygon.from_config([]),
        cfg=cfg or PipelineConfig(detect_interval=1, plate_interval=1),
        camera_id=1,
        read_zone=read_zone,
    )


def drive(processor, n, ts0=1000.0, step=0.16, frame_age_ms=None, frame=None):
    frame = blank_frame() if frame is None else frame
    for i in range(n):
        processor.process(frame, frame_idx=i + 1, ts=ts0 + i * step, frame_age_ms=frame_age_ms)


def approaching(n, x=500, y0=80, step=45, w0=120, grow=26):
    """A vehicle driving toward a low-mounted camera: the box grows and moves
    down the frame, exactly as it does at a real gate."""
    frames = []
    for i in range(n):
        w = w0 + i * grow
        h = int(w * 0.8)
        y = y0 + i * step
        frames.append([(x - w // 2, y, x + w // 2, y + h)])
    return frames


class TestReadZone:
    def test_no_read_zone_keeps_the_old_behaviour(self):
        """The rollout guarantee: an uncalibrated camera behaves as before."""
        processor = build(approaching(8))
        drive(processor, 8)
        assert processor.plate_detector.calls > 0
        assert processor.recognizer.calls > 0

    def test_a_vehicle_short_of_the_read_zone_costs_no_plate_call(self):
        """Gating on geometry BEFORE inference: not one plate-detector call,
        let alone an OCR one."""
        # Zone is the bottom fifth of the frame; the vehicle never gets there.
        zone = RoiPolygon.from_config([(0.0, 0.8), (1.0, 0.8), (1.0, 1.0), (0.0, 1.0)])
        frames = [[(500, 60, 700, 220)] for _ in range(8)]
        processor = build(frames, read_zone=zone)
        drive(processor, 8)
        assert processor.plate_detector.calls == 0
        assert processor.recognizer.calls == 0

    def test_entering_the_read_zone_starts_the_plate_stages(self):
        zone = RoiPolygon.from_config([(0.0, 0.55), (1.0, 0.55), (1.0, 1.0), (0.0, 1.0)])
        processor = build(approaching(10), read_zone=zone)
        drive(processor, 10)
        assert processor.plate_detector.calls > 0
        assert processor.recognizer.calls > 0

    def test_the_anchor_is_the_plate_end_not_the_whole_box(self):
        """A tall vehicle overlaps a near-field band with its roof long before
        its plate arrives. bottom_center is what makes that not count."""
        zone = RoiPolygon.from_config([(0.0, 0.75), (1.0, 0.75), (1.0, 1.0), (0.0, 1.0)])
        # Box spans y=400..560; bottom edge (560) is above the zone (y>=540)?
        # 0.75*720 = 540, so bottom 560 IS inside; use a taller box that
        # overlaps the band while its bottom stays out.
        frames = [[(500, 300, 700, 520)] for _ in range(8)]
        processor = build(frames, read_zone=zone)
        drive(processor, 8)
        assert processor.plate_detector.calls == 0, "roof overlap must not count as arrival"

    def test_overlap_anchor_is_available_for_awkward_mountings(self):
        zone = RoiPolygon.from_config([(0.0, 0.3), (1.0, 0.3), (1.0, 1.0), (0.0, 1.0)])
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            read_zone_anchor="overlap", min_read_zone_overlap=0.2,
        )
        processor = build(approaching(8), cfg=cfg, read_zone=zone)
        drive(processor, 8)
        assert processor.plate_detector.calls > 0


class TestOcrBudget:
    def test_the_budget_caps_recognizer_calls(self):
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(budget_per_track=3, reserve=1, interval_excellent=1),
        )
        processor = build(approaching(30, step=12, grow=6), cfg=cfg)
        drive(processor, 30)
        assert processor.recognizer.calls <= 3

    def test_an_exhausted_budget_stops_the_plate_detector_too(self):
        """Localization exists only to feed recognition. Once the budget is
        gone there is nothing for a candidate to become."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(budget_per_track=2, reserve=1, interval_excellent=1),
        )
        processor = build(approaching(30, step=12, grow=6), cfg=cfg)
        drive(processor, 30)
        assert processor.recognizer.calls <= 2
        # A few detector calls before the budget ran out, nowhere near 30.
        assert processor.plate_detector.calls <= 6

    def test_locking_still_stops_the_plate_stages_early(self):
        """Requirement 7: early lock survives the new budget machinery, and
        must still bite BEFORE the budget does."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            lock_min_reads=4, lock_min_support=0.9,
            ocr=OcrPolicy(budget_per_track=12, reserve=3, interval_excellent=1),
        )
        processor = build(approaching(20, step=20, grow=10), cfg=cfg)
        drive(processor, 20)
        state = next(iter(processor.states.values()))
        assert state.locked
        assert state.ocr_calls < cfg.ocr.budget_per_track, "lock must beat the budget to it"
        assert processor.plate_detector.calls < 20


class TestPerFrameCap:
    def test_a_queue_of_vehicles_cannot_multiply_ocr_calls_in_one_frame(self):
        """The no-backlog guarantee, stated as a cost ceiling: whatever is
        happening at the gate, one frame costs at most max_per_frame calls."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(max_per_frame=1, interval_excellent=1, budget_per_track=99, reserve=1),
        )
        # Four vehicles abreast, all large, all readable, every frame.
        frames = [
            [(80 + lane * 300, 400, 80 + lane * 300 + 220, 620) for lane in range(4)]
            for _ in range(6)
        ]
        processor = build(frames, cfg=cfg)
        drive(processor, 6)
        assert processor.recognizer.calls <= 6, "at most one call per frame"

    def test_the_best_crop_wins_a_contested_frame(self):
        """Two vehicles, one much closer. The near one's plate is wider, so
        it should be the one the frame's single call is spent on."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(max_per_frame=1, interval_excellent=1, budget_per_track=99, reserve=1),
        )
        frames = [[(100, 420, 700, 700), (900, 300, 1010, 400)] for _ in range(6)]
        processor = build(frames, cfg=cfg)
        drive(processor, 6)
        assert processor.recognizer.calls > 0
        # The near vehicle's plate is ~300 px; the far one's ~55 px.
        assert min(processor.recognizer.widths) > 100


class TestWaitBackoff:
    def test_waiting_also_backs_off_the_plate_detector(self):
        """Declining to READ a crop while still paying to PRODUCE it three
        times a second is the same waste one stage earlier."""
        # Big enough to clear min_vehicle_area_ratio, but the readable-width
        # floor is set so high that no crop is ever worth reading.
        far = [[(500, 300, 760, 500)] for _ in range(24)]

        eager = build(far, cfg=PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(min_plate_width_read=100000, wait_backoff=1)))
        drive(eager, 24)

        backed_off = build(far, cfg=PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(min_plate_width_read=100000, wait_backoff=3)))
        drive(backed_off, 24)

        assert backed_off.plate_detector.calls < eager.plate_detector.calls
        assert eager.recognizer.calls == backed_off.recognizer.calls == 0

    def test_backoff_still_notices_the_plate_becoming_readable(self):
        """The backoff must not be a one-way door: a vehicle that arrives has
        to get read, just checked for less often on the way in."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1, lock_min_reads=999,
            ocr=OcrPolicy(wait_backoff=3, min_plate_width_read=90, budget_per_track=99, reserve=1),
        )
        processor = build(approaching(24, y0=40, step=22, w0=100, grow=18), cfg=cfg, plate_fraction=0.3)
        drive(processor, 24)
        assert processor.recognizer.calls > 0
        assert min(processor.recognizer.widths) >= 90


class TestStaleFrames:
    def test_a_stale_frame_is_never_recognized(self):
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1, ocr=OcrPolicy(max_frame_age_ms=400.0)
        )
        processor = build(approaching(10), cfg=cfg)
        drive(processor, 10, frame_age_ms=900.0)
        assert processor.recognizer.calls == 0

    def test_stale_frames_still_detect_and_track(self):
        """Dropping the frame entirely would break tracks, and a broken track
        scatters a vehicle's reads across two events."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1, ocr=OcrPolicy(max_frame_age_ms=400.0)
        )
        processor = build(approaching(10), cfg=cfg)
        drive(processor, 10, frame_age_ms=900.0)
        assert processor.detector.calls == 10
        assert len(processor.states) == 1

    def test_no_age_supplied_disables_the_check(self):
        processor = build(approaching(10))
        drive(processor, 10, frame_age_ms=None)
        assert processor.recognizer.calls > 0


class TestObservations:
    def test_observations_are_recorded_even_without_recognition(self):
        """A too-small plate is still evidence — it is what tells the
        scheduler the vehicle is approaching."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(min_plate_width_read=100000),  # nothing is ever readable
        )
        processor = build(approaching(10), cfg=cfg)
        drive(processor, 10)
        state = next(iter(processor.states.values()))
        assert processor.recognizer.calls == 0
        assert state.observation_count > 0
        assert state.peak_plate_width > 0

    def test_top_k_retention_is_bounded_and_keeps_the_best(self):
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(keep_observations=3, budget_per_track=99, reserve=1),
        )
        processor = build(approaching(14, step=18, grow=14), cfg=cfg)
        drive(processor, 14)
        state = next(iter(processor.states.values()))
        assert state.observation_count > 3, "more looks than we retain"
        assert len(state.plate_observations) <= 3
        # Retention is by quality, not recency.
        kept = [o.quality for o in state.plate_observations]
        assert min(kept) >= 0.0
        assert state.best_observation.quality == max(kept)

    def test_an_event_image_survives_a_vehicle_that_was_never_read(self):
        """Requirement: a vehicle we could not read is still an event worth
        showing. Observations give it a plate crop where reads gave none."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(min_plate_width_read=100000),
        )
        processor = build(approaching(10), cfg=cfg)
        drive(processor, 10)
        state = next(iter(processor.states.values()))
        assert state.reads == []
        assert state.best_plate_crop is not None

    def test_the_vehicle_image_is_the_best_view_not_the_first(self):
        """Regression guard. best_quality now moves with observations, so the
        vehicle image needs its own high-water mark or it sticks on frame one."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1, lock_min_reads=999,
            ocr=OcrPolicy(budget_per_track=99, reserve=1, min_plate_width_read=40),
        )
        processor = build(approaching(12, y0=60, step=40, w0=150, grow=40), cfg=cfg, plate_fraction=0.3)
        drive(processor, 12, frame=soft_frame())
        state = next(iter(processor.states.values()))
        assert state.best_vehicle_crop is not None
        # The vehicle grows all clip, so the best view is a late, large crop —
        # never the small first one.
        first_width = 150
        assert state.best_vehicle_crop.shape[1] > first_width

    def test_nothing_accumulates_across_a_long_track(self):
        """The no-backlog guarantee in memory terms: a vehicle that parks in
        the read zone must not grow an unbounded buffer of pending crops."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(keep_observations=5, budget_per_track=99, reserve=1),
        )
        frames = [[(400, 380, 800, 700)] for _ in range(60)]
        processor = build(frames, cfg=cfg)
        drive(processor, 60)
        state = next(iter(processor.states.values()))
        assert len(state.plate_observations) <= 5
        assert len(state.recent_plate_widths) <= cfg.ocr.growth_window + 1


class TestApproachBehaviour:
    def test_a_growing_plate_is_read_less_often_than_a_stopped_one(self):
        """The saving, measured. Same frame count, same models, same final
        plate size; the only difference is whether the vehicle is still
        approaching. Uses the soft frame so quality tracks plate width the way
        it does on a real approach."""
        # Locking is disabled here ONLY so the comparison measures the
        # scheduler. With it on, the stub recognizer's identical reads lock
        # both runs at four reads and the two numbers are equal by
        # construction — which is the lock doing its job, tested separately.
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1, lock_min_reads=999,
            ocr=OcrPolicy(budget_per_track=99, reserve=1, min_plate_width_read=40),
        )
        boxes = approaching(12, y0=60, step=40, w0=150, grow=40)

        # Approaching: plate grows from ~45 px to ~200 px across the clip.
        moving = build(boxes, cfg=cfg, plate_fraction=0.3)
        drive(moving, 12, frame=soft_frame())

        # Stopped at the boom, at the size the approach ends on.
        parked = build([boxes[-1] for _ in range(12)], cfg=cfg, plate_fraction=0.3)
        drive(parked, 12, frame=soft_frame())

        assert moving.recognizer.calls < parked.recognizer.calls, (
            f"approach spent {moving.recognizer.calls} calls, "
            f"stopped spent {parked.recognizer.calls}"
        )

    def test_the_approach_wait_is_recorded_as_such(self):
        """Not just fewer calls — fewer calls for the documented reason."""
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1, lock_min_reads=999,
            ocr=OcrPolicy(budget_per_track=99, reserve=1, min_plate_width_read=40),
        )
        processor = build(approaching(6, y0=60, step=40, w0=150, grow=40), cfg=cfg, plate_fraction=0.3)
        drive(processor, 6, frame=soft_frame())
        state = next(iter(processor.states.values()))
        assert state.observation_count > state.ocr_calls
        assert state.approach_growth(cfg.ocr.growth_window) > 1.0

    def test_frames_since_ocr_only_resets_on_a_real_call(self):
        cfg = PipelineConfig(
            detect_interval=1, plate_interval=1,
            ocr=OcrPolicy(min_plate_width_read=100000),
        )
        processor = build(approaching(8), cfg=cfg)
        drive(processor, 8)
        state = next(iter(processor.states.values()))
        assert state.ocr_calls == 0
        assert state.frames_since_ocr > 5
        assert state.consecutive_ocr_waits > 0
        assert state.last_ocr_reason == "plate_too_small"
