import numpy as np
import pytest

from backend.app.ai.inference.ops import (
    crop, decode_yolo_output, letterbox, nms, to_blob, undo_letterbox, xywh_to_xyxy,
)


class TestLetterbox:
    def test_output_shape_and_aspect(self):
        image = np.zeros((534, 377, 3), dtype=np.uint8)
        padded, scale, (px, py) = letterbox(image, (640, 640))
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(640 / 534, rel=1e-3)
        assert px > 0 and py == 0        # tall image pads left/right

    def test_round_trip_maps_a_box_back(self):
        image = np.zeros((534, 377, 3), dtype=np.uint8)
        _, scale, pad = letterbox(image, (640, 640))
        original = np.array([[10.0, 20.0, 200.0, 300.0]])
        forward = original * scale + np.array([pad[0], pad[1], pad[0], pad[1]])
        back = undo_letterbox(forward, scale, pad, image.shape[:2])
        assert back == pytest.approx(original, abs=1.0)

    def test_reuses_a_provided_buffer(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        buffer = np.empty((320, 320, 3), dtype=np.uint8)
        padded, _, _ = letterbox(image, (320, 320), out=buffer)
        assert padded is buffer


class TestUndoLetterbox:
    def test_clips_to_the_frame(self):
        """Regression: fancy indexing returns a copy, so np.clip(..., out=...)
        on a sliced view silently does nothing and negative coordinates leak
        through into crops."""
        boxes = np.array([[-50.0, -30.0, 5000.0, 4000.0]])
        clipped = undo_letterbox(boxes, 1.0, (0, 0), (480, 640))
        assert clipped[0][0] >= 0
        assert clipped[0][1] >= 0
        assert clipped[0][2] <= 639
        assert clipped[0][3] <= 479

    def test_empty_input(self):
        assert undo_letterbox(np.empty((0, 4)), 1.0, (0, 0), (10, 10)).size == 0


class TestNms:
    def test_suppresses_overlapping_boxes(self):
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 600, 600]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        assert nms(boxes, scores, 0.5) == [0, 2]

    def test_keeps_distinct_boxes(self):
        boxes = np.array([[0, 0, 50, 50], [200, 200, 250, 250]], dtype=float)
        assert len(nms(boxes, np.array([0.9, 0.8]), 0.5)) == 2

    def test_empty(self):
        assert nms(np.empty((0, 4)), np.empty((0,)), 0.5) == []


class TestDecodeYoloOutput:
    def _fake_head(self, n_classes: int, cls: int = 0, n_anchors: int = 2100) -> np.ndarray:
        """Shaped like a real export: (1, 4 + n_classes, n_anchors), with the
        anchor count far larger than the feature count — which is what the
        auto-transpose in decode_yolo_output keys off."""
        raw = np.zeros((1, 4 + n_classes, n_anchors), dtype=np.float32)
        raw[0, :4, 0] = [100, 100, 40, 20]      # cx, cy, w, h
        raw[0, 4 + cls, 0] = 0.9
        return raw

    def test_decodes_80_class_head(self):
        """Matches the exported yolo26n head: (1, 84, 8400)."""
        boxes, scores, class_ids = decode_yolo_output(self._fake_head(80, cls=2, n_anchors=8400), 0.25)
        assert boxes.shape == (1, 4)
        assert class_ids[0] == 2
        assert scores[0] == pytest.approx(0.9)
        assert boxes[0] == pytest.approx([80, 90, 120, 110])

    def test_decodes_single_class_plate_head(self):
        """Matches the exported plate head: (1, 5, 2100)."""
        boxes, scores, class_ids = decode_yolo_output(self._fake_head(1, cls=0), 0.25)
        assert boxes.shape == (1, 4)
        assert class_ids.tolist() == [0]
        assert scores[0] == pytest.approx(0.9)

    def test_accepts_an_already_transposed_head(self):
        raw = self._fake_head(1, cls=0)[0].T          # (n_anchors, 5)
        boxes, _, _ = decode_yolo_output(raw, 0.25)
        assert boxes.shape == (1, 4)

    def test_threshold_filters(self):
        boxes, _, _ = decode_yolo_output(self._fake_head(80, cls=2, n_anchors=8400), 0.95)
        assert boxes.shape[0] == 0


def test_xywh_to_xyxy():
    result = xywh_to_xyxy(np.array([[100.0, 100.0, 40.0, 20.0]]))
    assert result[0] == pytest.approx([80, 90, 120, 110])


class TestCrop:
    def test_basic(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        assert crop(image, (10, 10, 60, 50)).shape == (40, 50, 3)

    def test_clamps_out_of_bounds(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        assert crop(image, (-20, -20, 300, 300)).shape == (100, 200, 3)

    def test_degenerate_returns_none(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        assert crop(image, (50, 50, 51, 51)) is None

    def test_padding_expands(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        padded = crop(image, (50, 40, 100, 60), pad=0.1)
        assert padded.shape[0] > 20 and padded.shape[1] > 50


def test_to_blob_shape_and_range():
    blob = to_blob(np.full((320, 320, 3), 255, dtype=np.uint8))
    assert blob.shape == (1, 3, 320, 320)
    assert blob.dtype == np.float32
    assert blob.max() <= 1.0 + 1e-6
