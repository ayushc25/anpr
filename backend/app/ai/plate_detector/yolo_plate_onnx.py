"""Dedicated license-plate detector (single-class YOLO) on ONNX."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..inference.backend import build_backend
from ..inference.ops import decode_yolo_output, letterbox, nms, to_blob, undo_letterbox
from ..inference.threading import ThreadBudget
from ..types import PlateCandidate
from .base import PlateDetector

# An Indian single-line plate is ~4.5:1; the stacked two-wheeler/commercial
# plate is ~2:1. Anything outside this band is not a plate, and rejecting it
# here is much cheaper than letting the recognizer produce noise the validator
# then has to outvote.
MIN_ASPECT = 1.2
MAX_ASPECT = 8.0


class YoloPlateDetector(PlateDetector):
    name = "yolo_plate_onnx"

    def __init__(
        self,
        artifact: str | Path,
        input_size: tuple[int, int] = (320, 320),
        conf_threshold: float = 0.30,
        nms_iou: float = 0.45,
        backend: str = "auto",
        threads: ThreadBudget | None = None,
        cache_dir: str | Path | None = None,
        max_plates: int = 3,
        min_width_px: int = 24,
        filter_aspect: bool = True,
    ):
        self._input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(conf_threshold)
        self.nms_iou = float(nms_iou)
        self.max_plates = int(max_plates)
        self.min_width_px = int(min_width_px)
        self.filter_aspect = bool(filter_aspect)
        self.backend = build_backend(artifact, backend, threads, cache_dir)
        self._canvas: np.ndarray | None = None

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    def detect(self, image: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[PlateCandidate]:
        if image is None or image.size == 0:
            return []

        padded, scale, pad = letterbox(image, self._input_size, out=self._canvas)
        self._canvas = padded
        outputs = self.backend.run_single(to_blob(padded))

        boxes, scores, _ = decode_yolo_output(outputs[0], self.conf_threshold)
        if boxes.shape[0] == 0:
            return []
        boxes = undo_letterbox(boxes, scale, pad, image.shape[:2])

        off_x, off_y = offset
        candidates: list[PlateCandidate] = []
        for k in nms(boxes, scores, self.nms_iou):
            x1, y1, x2, y2 = (float(v) for v in boxes[k])
            w, h = x2 - x1, y2 - y1
            if w < self.min_width_px or h < 6:
                continue
            if self.filter_aspect and not (MIN_ASPECT <= w / max(h, 1e-6) <= MAX_ASPECT):
                continue
            candidates.append(
                PlateCandidate(
                    bbox=(int(x1) + off_x, int(y1) + off_y, int(x2) + off_x, int(y2) + off_y),
                    confidence=float(scores[k]),
                )
            )
            if len(candidates) >= self.max_plates:
                break
        return candidates

    def warmup(self, n: int = 2) -> float:
        return self.backend.warmup(n)

    def close(self) -> None:
        self.backend.close()
