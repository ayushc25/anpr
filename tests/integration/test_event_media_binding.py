"""Regression tests for two data-association bugs found in production.

BUG 1 — event media was chosen independently of the winning plate.
    `TrackState` kept ONE global best crop per track (highest quality seen)
    while the plate came from a weighted vote across all reads. When a
    track's identity drifted across vehicles, the two selections landed on
    different ones. Real event 2692: plate `HR29BG7381`, stored photograph a
    grey Honda City carrying `UP14FU2031`, because that later frame scored
    q=1.00 against the winning frames' q=0.99. `is_disputed` was False.

BUG 2 — `max_duration` finalization popped a LIVE track's state.
    The next frame rebuilt a fresh `TrackState` for the same live track_id,
    which accumulated new reads and emitted again when the timer expired.
    Real tracks: 56 emitted `DL5CU1624` (car) then `UP16BA8695` (motorcycle)
    42 s apart; 19 emitted twice 101 s apart; 57 twice 45 s apart.

Both are verified here by construction rather than by inspection, because
both are invisible in a log — each individual value looks plausible.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from backend.app.ai.plate_recognizer import postprocess
from backend.app.ai.quality.plate_quality import QualityBreakdown
from backend.app.ai.types import Detection, PlateCandidate, PlateRead
from backend.app.ai.vehicle_tracker.bytetrack import ByteTracker
from backend.app.events.event_builder import EventBuilder
from backend.app.events.multi_frame_validator import MultiFrameValidator, ValidationConfig
from backend.app.events.track_state import TrackState
from backend.app.video.frame_processor import FrameProcessor, PipelineConfig
from backend.app.video.roi import RoiPolygon

# Colour-coded crops, so a decoded JPEG says which vehicle it came from.
BLUE = (200, 40, 40)   # BGR — the winning plate's vehicle
RED = (40, 40, 200)    # BGR — the other vehicle in the same drifted track


def solid(colour, w=140, h=40):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = colour
    return img


def dominant(b64: str) -> str:
    """Decode an event image and say which colour-coded vehicle it shows."""
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    b, g, r = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    return "blue" if b > r else "red"


def quality(score: float) -> QualityBreakdown:
    return QualityBreakdown(score, 140, 0.0, 0.0, 0.0, 0.0, 0.0)


def add(state, text, *, q, ts, frame_idx, colour, rec=0.80, det=0.80):
    """One read, with plate and vehicle crops colour-coded to its vehicle."""
    return state.add_read(
        PlateRead(text=text, confidence=rec, per_char_confidence=[rec] * len(text), raw_text=text),
        plate_det_confidence=det,
        quality=quality(q),
        frame_ts=ts,
        plate_crop=solid(colour),
        vehicle_crop=solid(colour, 200, 160),
        frame_idx=frame_idx,
    )


#: The real strings from production event 2692, track 209.
WINNER = "HR29BG7381"   # 10 chars, four reads over t+0..4.3s
DRIFTED = "UP4FU2031"   # 9 chars, one read at t+12.25s, q=1.00

def drifted_track() -> TrackState:
    """A track whose identity drifted, reproducing event 2692 exactly.

    Four reads of the winning plate on mediocre crops, then ONE read of a
    DIFFERENT vehicle's plate on the best crop in the track. The old code
    took its images from that last frame.

    The drifted read is a different LENGTH, which is what let it through in
    production: the positional vote only pools reads of the modal length, so
    a 9-character read contests none of the 10 winning characters and the
    plate emits with support 0.78. Two same-length plates would contest every
    position and the validator would — correctly — reject the whole track as
    UNRESOLVED, which is what it did when this test first used them.
    """
    state = TrackState(track_id=209, camera_id=2)
    for i in range(4):
        add(state, WINNER, q=0.60, ts=1000.0 + i, frame_idx=i * 2, colour=BLUE)
    add(state, DRIFTED, q=1.00, ts=1012.0, frame_idx=60, colour=RED, rec=0.95)
    return state


class TestBug1MediaFollowsTheWinningPlate:
    def test_the_winning_plate_is_the_one_with_more_reads(self):
        """Precondition: the vote must pick UP32AB1234, not the single
        higher-quality read. Otherwise the test proves nothing."""
        final = MultiFrameValidator(ValidationConfig()).validate(drifted_track())
        assert final is not None
        assert final.text == WINNER

    def test_images_come_from_the_winning_plate_not_the_best_frame(self):
        """THE regression. The other vehicle's frame has strictly higher
        quality, so the old global-best selection chose it."""
        state = drifted_track()
        final = MultiFrameValidator(ValidationConfig()).validate(state)
        draft = EventBuilder(camera_id=2).build(state, final)

        assert dominant(draft.plate_image_b64) == "blue", "plate crop must be the winner's"
        assert dominant(draft.vehicle_image_b64) == "blue", "vehicle crop must be the winner's"

    def test_the_old_global_best_would_have_chosen_the_other_vehicle(self):
        """Documents the bug: the fallback the fix bypasses still points at
        the wrong vehicle, which is exactly what production was storing."""
        state = drifted_track()
        assert dominant(base64.b64encode(cv2.imencode('.jpg', state.best_plate_crop)[1]).decode()) == "red"
        assert dominant(base64.b64encode(cv2.imencode('.jpg', state.best_vehicle_crop)[1]).decode()) == "red"

    def test_media_provenance_is_recorded(self):
        state = drifted_track()
        final = MultiFrameValidator(ValidationConfig()).validate(state)
        draft = EventBuilder(camera_id=2).build(state, final)

        assert draft.media_source == "hypothesis"
        assert draft.winning_read_count == 4
        # Both images from within the winning plate's 3-second window, never
        # the drifted frame at t=1012.
        assert draft.winning_window == (1000.0, 1003.0)
        assert 1000.0 <= draft.plate_image_ts <= 1003.0
        assert 1000.0 <= draft.vehicle_image_ts <= 1003.0
        assert draft.plate_image_frame <= 6
        assert draft.vehicle_image_frame <= 6

    def test_a_read_repaired_at_ingest_still_binds_exactly(self):
        """Reads are repaired BEFORE they vote (phase 1), so the hypothesis is
        keyed on the repaired string and the winner matches it exactly. The
        raw text is still preserved in the audit trail."""
        state = TrackState(track_id=1, camera_id=2)
        for i in range(4):
            # pos5 must be alpha, so repair turns 6 into G at ingest.
            add(state, "HR29B67381", q=0.60, ts=1000.0 + i, frame_idx=i * 2, colour=BLUE)
        add(state, DRIFTED, q=1.00, ts=1012.0, frame_idx=60, colour=RED, rec=0.95)

        final = MultiFrameValidator(ValidationConfig()).validate(state)
        draft = EventBuilder(camera_id=2).build(state, final)
        assert final.text == WINNER
        assert draft.media_source == "hypothesis"
        assert dominant(draft.vehicle_image_b64) == "blue"
        assert {r.raw_text for r in draft.reads} >= {"HR29B67381"}, "raw text preserved"

    def test_a_registry_snapped_plate_takes_the_images_it_was_derived_from(self):
        """The validator can change the plate AFTER the vote — registry
        snapping and positional fusion both emit strings no read produced.
        The images must follow the read the winner was DERIVED from, never an
        unrelated frame.
        """
        from backend.app.events.multi_frame_validator import SnapshotRegistry

        state = TrackState(track_id=1, camera_id=2)
        for i in range(4):
            # B/D is a same-type confusion: repair never touches it (both are
            # letters and the mask is satisfied), so only the registry can.
            add(state, "HR29DG7381", q=0.60, ts=1000.0 + i, frame_idx=i * 2,
                colour=BLUE, rec=0.60)
        add(state, DRIFTED, q=1.00, ts=1012.0, frame_idx=60, colour=RED, rec=0.95)

        registry = SnapshotRegistry({WINNER: "resident"})
        final = MultiFrameValidator(ValidationConfig(), registry).validate(state)
        assert final is not None
        assert final.text == WINNER, "snapped, so no read has this exact text"
        assert any("registry-snap" in c for c in final.corrections)

        draft = EventBuilder(camera_id=2).build(state, final)
        assert draft.media_source == "derived"
        assert dominant(draft.vehicle_image_b64) == "blue", "the snapped-from read's vehicle"
        assert dominant(draft.plate_image_b64) == "blue"

    def test_an_unsupported_plate_falls_back_and_says_so(self):
        """If no read's imagery supports the plate, the pairing is not
        guaranteed and the event must admit it rather than hide it."""
        state = drifted_track()
        final = MultiFrameValidator(ValidationConfig()).validate(state)
        final.text = "KA05ZZ9999"  # unrelated to every hypothesis
        draft = EventBuilder(camera_id=2).build(state, final)
        assert draft.media_source == "fallback_global"

    def test_hypothesis_storage_is_bounded(self):
        from backend.app.events.track_state import MAX_HYPOTHESES

        state = TrackState(track_id=1, camera_id=2)
        for i in range(20):
            add(state, f"UP32AB{1000+i}", q=0.5 + i * 0.02, ts=1000.0 + i, frame_idx=i, colour=BLUE)
        assert len(state.hypothesis_evidence) <= MAX_HYPOTHESES

    def test_a_retry_read_without_a_vehicle_crop_still_yields_both_images(self):
        """A targeted retry is recognized from a banked plate crop and has no
        vehicle crop. The hypothesis must keep the vehicle image it already
        had rather than losing it."""
        state = TrackState(track_id=1, camera_id=2)
        for i in range(3):
            add(state, "UP32AB1234", q=0.60, ts=1000.0 + i, frame_idx=i, colour=BLUE)
        state.add_read(
            PlateRead(text="UP32AB1234", confidence=0.9, per_char_confidence=[0.9] * 10, raw_text="UP32AB1234"),
            plate_det_confidence=0.9, quality=quality(0.99), frame_ts=1020.0,
            plate_crop=solid(BLUE), frame_idx=99,   # no vehicle_crop
        )
        final = MultiFrameValidator(ValidationConfig()).validate(state)
        draft = EventBuilder(camera_id=2).build(state, final)
        assert draft.vehicle_image_b64 is not None
        assert draft.plate_image_b64 is not None
        assert dominant(draft.vehicle_image_b64) == "blue"

    def test_the_read_audit_trail_is_preserved(self):
        """Requirement: preserve the existing audit trail."""
        state = drifted_track()
        final = MultiFrameValidator(ValidationConfig()).validate(state)
        draft = EventBuilder(camera_id=2).build(state, final)
        assert len(draft.reads) == 5, "all five reads, including the drifted one"
        assert {r.normalized_text for r in draft.reads} == {WINNER, DRIFTED}


# --------------------------------------------------------------------------
# BUG 2 — max_duration must not let one live track emit repeatedly
# --------------------------------------------------------------------------

FRAME_W, FRAME_H = 1280, 720


def blank():
    rng = np.random.default_rng(4)
    return rng.integers(90, 170, (FRAME_H, FRAME_W, 3), dtype=np.uint8)


class ParkedDetector:
    """One vehicle, never moving, never leaving — a vehicle stopped at the
    boom, or a track whose identity has drifted onto a static object."""

    name = "stub"
    input_size = (640, 640)

    def __init__(self, frames):
        self.frames = frames
        self.calls = 0

    def detect(self, frame, roi_offset=(0, 0)):
        self.calls += 1
        if self.calls > self.frames:
            return []          # the vehicle finally leaves -> track retires
        return [Detection(bbox=(400, 380, 700, 640), confidence=0.93, class_id=2, class_name="car")]

    def warmup(self, n=2):
        return 0.0

    def close(self):
        pass


class StubPlate:
    name = "stub"
    input_size = (320, 320)

    def detect(self, image, offset=(0, 0)):
        h, w = image.shape[:2]
        pw = int(w * 0.45)
        x1 = offset[0] + (w - pw) // 2
        y1 = offset[1] + int(h * 0.70)
        return [PlateCandidate(bbox=(x1, y1, x1 + pw, y1 + int(pw / 4.5)), confidence=0.9)]


class StubRec:
    name = "stub"
    charset = ""
    expects_grayscale = True

    def __init__(self):
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        t = "UP32AB1234"
        return PlateRead(text=t, confidence=0.9, per_char_confidence=[0.9] * len(t), raw_text=t)

    def close(self):
        pass


def parked_processor(frames=400, max_track_seconds=2.0):
    return FrameProcessor(
        detector=ParkedDetector(frames),
        tracker=ByteTracker(min_hits=2, track_buffer=4),
        plate_detector=StubPlate(),
        recognizer=StubRec(),
        roi=RoiPolygon.from_config([]),
        cfg=PipelineConfig(
            detect_interval=1, plate_interval=1, max_track_seconds=max_track_seconds,
        ),
        camera_id=2,
    )


def drive(processor, n, step=0.5):
    """step=0.5s per frame, so max_track_seconds is crossed many times."""
    frame = blank()
    out = []
    for i in range(n):
        out.extend(processor.process(frame, frame_idx=i + 1, ts=1000.0 + i * step).finalized)
    return out


class TestBug2MaxDurationDoesNotLeak:
    def test_one_live_track_emits_exactly_once(self):
        """THE regression. 40 frames at 0.5 s = 20 s of track life against a
        2 s ceiling: the old code emitted roughly ten times."""
        processor = parked_processor(frames=400)
        finalized = drive(processor, 40)

        assert len(finalized) == 1, f"one live track emitted {len(finalized)} times"
        assert finalized[0].finalize_reason == "max_duration"
        assert len({s.track_id for s in finalized}) == 1

    def test_the_state_survives_as_a_tombstone_while_the_track_lives(self):
        processor = parked_processor(frames=400)
        drive(processor, 40)
        states = list(processor.states.values())
        assert len(states) == 1, "the state must NOT be popped while the track is alive"
        assert states[0].emitted

    def test_a_fresh_state_is_never_rebuilt_for_the_same_track_id(self):
        """The mechanism: popping the state let the next frame construct a new
        TrackState for the same live track_id, which then accumulated a
        different vehicle's reads and emitted again."""
        processor = parked_processor(frames=400)
        drive(processor, 40)
        state = next(iter(processor.states.values()))
        reads_at_emit = state.read_count
        drive(processor, 40)
        assert next(iter(processor.states.values())) is state, "same object, not a rebuild"
        assert state.read_count == reads_at_emit, "an emitted track takes no further reads"

    def test_no_plate_work_is_spent_after_emitting(self):
        processor = parked_processor(frames=400)
        drive(processor, 30)
        ocr_at_emit = processor.recognizer.calls
        drive(processor, 60)
        assert processor.recognizer.calls == ocr_at_emit

    def test_the_tracker_still_cleans_the_state_up_on_retirement(self):
        """`removed_track_ids` remains the owner of cleanup, so a tombstone
        is not a leak — it lasts exactly as long as the track."""
        processor = parked_processor(frames=30)
        drive(processor, 30)
        assert len(processor.states) == 1
        drive(processor, 20)          # detector runs dry, tracker retires it
        assert processor.states == {}

    def test_a_retired_track_does_not_emit_a_second_event(self):
        processor = parked_processor(frames=30)
        finalized = drive(processor, 60)
        assert len(finalized) == 1, "max_duration then retirement must not double-emit"

    def test_a_genuinely_new_track_still_emits(self):
        """The fix must not suppress real vehicles."""
        processor = parked_processor(frames=400)
        drive(processor, 40)
        processor.detector.calls = 0          # a new vehicle arrives
        processor.tracker.reset()
        finalized = drive(processor, 40)
        assert len(finalized) >= 1

    def test_flush_still_reports_unemitted_tracks_only(self):
        processor = parked_processor(frames=400)
        drive(processor, 40)
        assert processor.flush() == [], "already emitted, so nothing pending"
        assert processor.states == {}
