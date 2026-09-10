"""Debug artifact recorder.

Two properties matter more than the writing itself: it must be genuinely free
when disabled, and it must never be able to take down a camera worker.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from backend.app.debug.recorder import DebugConfig, DebugRecorder, NullRecorder


def image(w=64, h=32):
    rng = np.random.default_rng(5)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def payload(text="UP32AB1234"):
    return {
        "ocr": {
            "normalized_text": text,
            "per_char_confidence": [0.9] * len(text),
            "per_char_alternatives": [[["B", 0.1]] for _ in text],
        }
    }


@pytest.fixture
def recorder(tmp_path):
    return DebugRecorder(DebugConfig(enabled=True), camera_id=7, root=tmp_path)


class TestDisabledIsFree:
    def test_disabled_config_produces_no_directory(self, tmp_path):
        r = DebugRecorder(DebugConfig(enabled=False), camera_id=1, root=tmp_path)
        assert not r.enabled
        r.record_ocr(1, 1, vehicle_crop=image(), payload=payload())
        assert list(tmp_path.iterdir()) == []

    def test_null_recorder_accepts_everything_and_does_nothing(self):
        n = NullRecorder()
        assert not n.enabled
        n.record_ocr(1, 1, vehicle_crop=image(), payload=payload())
        n.record_resolution(1, {"a": 1})
        assert n.stats == {"enabled": False}


class TestArtifacts:
    def test_all_four_crops_are_written(self, recorder, tmp_path):
        """The four images separate causes that are indistinguishable in a
        log: a badly framed box, a bad deskew, and an enhancement that
        crushed a low-contrast plate all produce the same wrong string."""
        recorder.record_ocr(
            42, 7,
            vehicle_crop=image(200, 160),
            plate_raw=image(),
            plate_warped=image(),
            plate_enhanced=image(128, 64),
            payload=payload(),
        )
        target = tmp_path / "camera_7" / "track_000042" / "frame_000007"
        for name in ("vehicle", "plate_raw", "plate_warped", "plate_enhanced"):
            assert (target / f"{name}.jpg").exists(), name
        assert (target / "ocr.json").exists()

    def test_per_character_probabilities_reach_the_json(self, recorder, tmp_path):
        """The raw material for a real confusion matrix — counted from
        recorded evidence rather than guessed from single examples."""
        recorder.record_ocr(1, 1, plate_warped=image(), payload=payload("UP1606"))
        data = json.loads(
            (tmp_path / "camera_7" / "track_000001" / "frame_000001" / "ocr.json").read_text()
        )
        assert data["ocr"]["normalized_text"] == "UP1606"
        assert len(data["ocr"]["per_char_confidence"]) == 6
        assert data["ocr"]["per_char_alternatives"][0] == [["B", 0.1]]

    def test_missing_crops_are_simply_skipped(self, recorder, tmp_path):
        recorder.record_ocr(1, 1, plate_warped=image(), payload=payload())
        target = tmp_path / "camera_7" / "track_000001" / "frame_000001"
        assert (target / "plate_warped.jpg").exists()
        assert not (target / "vehicle.jpg").exists()

    def test_the_resolution_lands_beside_the_frames(self, recorder, tmp_path):
        recorder.record_ocr(9, 1, plate_warped=image(), payload=payload())
        recorder.record_resolution(9, {"plate_number": "UP32AB1234", "recognition_state": "confirmed"})
        resolved = tmp_path / "camera_7" / "track_000009" / "resolved.json"
        assert json.loads(resolved.read_text())["recognition_state"] == "confirmed"

    def test_a_resolution_without_frames_is_not_written(self, recorder, tmp_path):
        """A lone verdict has nothing to explain."""
        recorder.record_resolution(99, {"plate_number": "X"})
        assert not (tmp_path / "camera_7" / "track_000099").exists()

    def test_numpy_scalars_survive_json_encoding(self, recorder, tmp_path):
        recorder.record_ocr(
            1, 1, plate_warped=image(),
            payload={"q": np.float32(0.75), "n": np.int64(3), "arr": np.array([1, 2])},
        )
        data = json.loads(
            (tmp_path / "camera_7" / "track_000001" / "frame_000001" / "ocr.json").read_text()
        )
        assert data["q"] == pytest.approx(0.75)
        assert data["n"] == 3
        assert data["arr"] == [1, 2]


class TestBounds:
    def test_the_track_cap_is_enforced(self, tmp_path):
        r = DebugRecorder(DebugConfig(enabled=True, max_tracks=3), camera_id=7, root=tmp_path)
        for track in range(10):
            r.record_ocr(track, 1, plate_warped=image(), payload=payload())
        assert r.stats["tracks_recorded"] == 3

    def test_a_track_already_recording_keeps_recording(self, tmp_path):
        """No vehicle should be left with half its frames captured."""
        r = DebugRecorder(DebugConfig(enabled=True, max_tracks=1), camera_id=7, root=tmp_path)
        for frame in range(5):
            r.record_ocr(1, frame, plate_warped=image(), payload=payload())
        r.record_ocr(2, 1, plate_warped=image(), payload=payload())
        frames = list((tmp_path / "camera_7" / "track_000001").iterdir())
        assert len(frames) == 5
        assert not (tmp_path / "camera_7" / "track_000002").exists()

    def test_the_byte_budget_stops_recording(self, tmp_path):
        r = DebugRecorder(DebugConfig(enabled=True, max_bytes=2000, max_tracks=999), camera_id=7, root=tmp_path)
        for frame in range(50):
            r.record_ocr(1, frame, vehicle_crop=image(400, 300), plate_warped=image(), payload=payload())
        assert r.stats["bytes_written"] >= 2000
        assert len(list((tmp_path / "camera_7" / "track_000001").iterdir())) < 50

    def test_stats_report_what_was_written(self, recorder):
        recorder.record_ocr(1, 1, plate_warped=image(), payload=payload())
        stats = recorder.stats
        assert stats["enabled"]
        assert stats["tracks_recorded"] == 1
        assert stats["bytes_written"] > 0


class TestItNeverRaises:
    def test_a_bad_payload_does_not_propagate(self, recorder):
        class Unserializable:
            pass

        # default=_encode falls back to str(), so this must not raise.
        recorder.record_ocr(1, 1, plate_warped=image(), payload={"x": Unserializable()})

    def test_an_undirectable_root_disables_rather_than_crashes(self, tmp_path):
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        r = DebugRecorder(DebugConfig(enabled=True), camera_id=7, root=blocker)
        assert not r.enabled
        r.record_ocr(1, 1, plate_warped=image(), payload=payload())

    def test_a_broken_image_is_skipped_not_fatal(self, recorder):
        recorder.record_ocr(1, 1, plate_warped=np.zeros((0, 0, 3), dtype=np.uint8), payload=payload())
