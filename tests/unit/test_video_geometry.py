import numpy as np
import pytest

from backend.app.video.line_crossing import CrossDirection, VirtualLine
from backend.app.video.roi import RoiPolygon

FRAME = (720, 1280)  # height, width


class TestRoiPolygon:
    @pytest.fixture
    def roi(self):
        # Middle half of the frame.
        return RoiPolygon.from_config([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])

    def test_empty_roi_accepts_everything(self):
        empty = RoiPolygon.from_config([])
        assert empty.is_empty
        assert empty.contains((5, 5), FRAME)
        assert empty.overlap_ratio((0, 0, 10, 10), FRAME) == 1.0

    def test_contains(self, roi):
        assert roi.contains((640, 360), FRAME)
        assert not roi.contains((10, 10), FRAME)

    def test_overlap_ratio(self, roi):
        assert roi.overlap_ratio((600, 340, 680, 380), FRAME) == pytest.approx(1.0)
        assert roi.overlap_ratio((0, 0, 50, 50), FRAME) == pytest.approx(0.0)
        partial = roi.overlap_ratio((300, 100, 500, 300), FRAME)
        assert 0.0 < partial < 1.0

    def test_crop_bounds_are_padded_and_clamped(self, roi):
        x1, y1, x2, y2 = roi.crop_bounds(FRAME, pad=0.05)
        assert 0 <= x1 < 320 and 0 <= y1 < 180
        assert 960 < x2 <= 1280 and 540 < y2 <= 720

    def test_crop_returns_offset(self, roi):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cropped, (ox, oy) = roi.crop(frame)
        assert cropped.shape[0] < frame.shape[0]
        assert (ox, oy) == roi.crop_bounds(FRAME)[:2]

    def test_normalized_coords_survive_a_resolution_change(self, roi):
        """The whole point of storing 0..1: switching a camera from its main
        stream to its sub-stream must not invalidate the operator's polygon."""
        assert roi.contains((640, 360), (720, 1280))
        assert roi.contains((320, 180), (360, 640))

    def test_round_trip_config(self, roi):
        assert RoiPolygon.from_config(roi.to_config()).points == roi.points


class TestVirtualLine:
    @pytest.fixture
    def line(self):
        # Horizontal line across the middle; forward (left of A->B) means IN.
        return VirtualLine.from_config([(0.0, 0.5), (1.0, 0.5)], forward="in")

    def test_no_crossing_without_movement(self, line):
        history = [(640.0, 100.0)] * 5
        assert line.check(history, FRAME) == CrossDirection.NONE

    def test_crossing_downward(self, line):
        history = [(640.0, 100.0), (640.0, 300.0), (640.0, 500.0), (640.0, 620.0)]
        assert line.check(history, FRAME) in (CrossDirection.IN, CrossDirection.OUT)

    def test_direction_reverses_with_travel_direction(self, line):
        down = [(640.0, 100.0), (640.0, 360.0), (640.0, 620.0)]
        up = list(reversed(down))
        assert line.check(down, FRAME) != line.check(up, FRAME)
        assert CrossDirection.NONE not in (line.check(down, FRAME), line.check(up, FRAME))

    def test_forward_direction_is_configurable(self):
        history = [(640.0, 100.0), (640.0, 360.0), (640.0, 620.0)]
        as_in = VirtualLine.from_config([(0.0, 0.5), (1.0, 0.5)], forward="in")
        as_out = VirtualLine.from_config([(0.0, 0.5), (1.0, 0.5)], forward="out")
        assert as_in.check(history, FRAME) != as_out.check(history, FRAME)

    def test_jitter_across_the_line_is_not_a_crossing(self, line):
        """A vehicle stopped on the line whose box wobbles by a few pixels must
        not emit an event every frame."""
        history = [(640.0, 359.0), (640.0, 361.0), (640.0, 358.0), (640.0, 362.0)]
        assert line.check(history, FRAME) == CrossDirection.NONE

    def test_passing_beyond_the_segment_is_not_a_crossing(self):
        # Line only spans the left quarter; the vehicle passes on the right.
        short = VirtualLine.from_config([(0.0, 0.5), (0.25, 0.5)], forward="in")
        history = [(1100.0, 100.0), (1100.0, 360.0), (1100.0, 620.0)]
        assert short.check(history, FRAME) == CrossDirection.NONE

    def test_too_short_history(self, line):
        assert line.check([(1.0, 1.0)], FRAME) == CrossDirection.NONE
        assert line.check([], FRAME) == CrossDirection.NONE

    def test_from_config_rejects_incomplete_geometry(self):
        assert VirtualLine.from_config(None) is None
        assert VirtualLine.from_config([(0.0, 0.5)]) is None

    def test_accepts_dict_points_from_the_ui(self):
        line = VirtualLine.from_config([{"x": 0.0, "y": 0.5}, {"x": 1.0, "y": 0.5}])
        assert line is not None and line.a == (0.0, 0.5)
