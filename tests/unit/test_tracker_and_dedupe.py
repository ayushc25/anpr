import pytest

from backend.app.ai.types import Detection
from backend.app.ai.vehicle_tracker.bytetrack import ByteTracker
from backend.app.ai.vehicle_tracker.iou_tracker import IouTracker
from backend.app.events.dedupe import DedupeConfig, EventDeduplicator
from backend.app.video.line_crossing import CrossDirection

FRAME = (720, 1280)


def car(x: int, y: int, w: int = 120, h: int = 90, conf: float = 0.9) -> Detection:
    return Detection(bbox=(x, y, x + w, y + h), confidence=conf, class_id=2, class_name="car")


class TestByteTracker:
    def test_assigns_a_stable_id_across_frames(self):
        tracker = ByteTracker(min_hits=2)
        ids = []
        for step in range(6):
            tracks = tracker.update([car(100 + step * 20, 300)], FRAME)
            if tracks:
                ids.append(tracks[0].track_id)
        assert ids and len(set(ids)) == 1

    def test_two_vehicles_keep_separate_ids(self):
        tracker = ByteTracker(min_hits=2)
        seen = set()
        for step in range(5):
            tracks = tracker.update([car(100 + step * 15, 200), car(700 - step * 15, 500)], FRAME)
            seen.update(t.track_id for t in tracks)
        assert len(seen) == 2

    def test_low_confidence_detection_continues_but_never_starts_a_track(self):
        """ByteTrack's second association pass: a vehicle whose score dips as
        it turns keeps its id, but a low-score blob on its own does not
        become a track."""
        tracker = ByteTracker(min_hits=2, track_thresh=0.5, low_thresh=0.1)

        assert tracker.update([car(100, 300, conf=0.2)], FRAME) == []
        assert tracker.update([car(120, 300, conf=0.2)], FRAME) == []

        tracker.reset()
        for step in range(3):
            tracker.update([car(100 + step * 20, 300, conf=0.9)], FRAME)
        established = tracker.update([car(160, 300, conf=0.9)], FRAME)
        assert established
        original_id = established[0].track_id

        # Score dips: the track must survive on the low-confidence detection.
        tracker.update([car(180, 300, conf=0.25)], FRAME)
        assert original_id not in tracker.removed_track_ids()

    def test_disappearing_vehicle_is_retired(self):
        tracker = ByteTracker(min_hits=2, track_buffer=3)
        for step in range(4):
            tracker.update([car(100 + step * 20, 300)], FRAME)
        for _ in range(6):
            tracker.update([], FRAME)
        assert tracker.removed_track_ids()

    def test_removed_ids_drain_once(self):
        tracker = ByteTracker(min_hits=2, track_buffer=2)
        for step in range(3):
            tracker.update([car(100 + step * 20, 300)], FRAME)
        for _ in range(5):
            tracker.update([], FRAME)
        assert tracker.removed_track_ids()
        assert tracker.removed_track_ids() == []

    def test_noise_never_reaches_the_event_builder(self):
        """A single-frame blob is not reported as a retired track, so it can
        never produce an event."""
        tracker = ByteTracker(min_hits=3, track_buffer=1)
        tracker.update([car(100, 300)], FRAME)
        for _ in range(4):
            tracker.update([], FRAME)
        assert tracker.removed_track_ids() == []

    def test_centroid_history_feeds_line_crossing(self):
        tracker = ByteTracker(min_hits=2)
        tracks = []
        for step in range(8):
            tracks = tracker.update([car(600, 100 + step * 70)], FRAME)
        assert tracks
        history = tracks[0].centroid_history
        assert len(history) >= 5
        assert history[-1][1] > history[0][1]

    def test_coasts_when_the_detector_did_not_run(self):
        """Frames where the detector is skipped must not break the track."""
        tracker = ByteTracker(min_hits=2, track_buffer=10)
        for step in range(3):
            tracker.update([car(100 + step * 20, 300)], FRAME)
        tracker.update([], FRAME)          # detector skipped this frame
        tracks = tracker.update([car(180, 300)], FRAME)
        assert len(tracks) == 1


class TestIouTracker:
    def test_is_bytetrack_without_the_second_pass(self):
        tracker = IouTracker(track_thresh=0.5)
        assert tracker.low_thresh == tracker.track_thresh

    def test_still_tracks(self):
        tracker = IouTracker()
        ids = set()
        for step in range(5):
            for track in tracker.update([car(100 + step * 20, 300)], FRAME):
                ids.add(track.track_id)
        assert len(ids) == 1


class TestDeduplicator:
    def test_first_event_passes(self):
        dedupe = EventDeduplicator()
        assert dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1000.0)

    def test_same_direction_within_window_is_suppressed(self):
        dedupe = EventDeduplicator(DedupeConfig(same_direction_seconds=25.0))
        dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1000.0)
        assert not dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1010.0)

    def test_same_direction_after_window_passes(self):
        dedupe = EventDeduplicator(DedupeConfig(same_direction_seconds=25.0))
        dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1000.0)
        assert dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1040.0)

    def test_bounce_is_suppressed(self):
        """Nose over the line, reverse, continue: one event, not three."""
        dedupe = EventDeduplicator(DedupeConfig(reversal_seconds=12.0))
        assert dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1000.0)
        assert not dedupe.check_and_record("UP32AB1234", CrossDirection.OUT, now=1003.0)
        assert not dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1006.0)

    def test_genuine_departure_passes(self):
        dedupe = EventDeduplicator(DedupeConfig(reversal_seconds=12.0))
        dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1000.0)
        assert dedupe.check_and_record("UP32AB1234", CrossDirection.OUT, now=4600.0)

    def test_different_plates_are_independent(self):
        dedupe = EventDeduplicator()
        assert dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1000.0)
        assert dedupe.check_and_record("DL8CAF5010", CrossDirection.IN, now=1001.0)

    def test_a_vehicle_sitting_in_the_roi_keeps_the_window_open(self):
        dedupe = EventDeduplicator(DedupeConfig(same_direction_seconds=25.0))
        dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=1000.0)
        for t in range(1010, 1200, 10):
            assert not dedupe.check_and_record("UP32AB1234", CrossDirection.IN, now=float(t))
