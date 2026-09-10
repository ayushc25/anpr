"""YOLOv8/v11-family vehicle detector on ONNX Runtime / OpenVINO."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..inference.backend import build_backend
from ..inference.ops import decode_yolo_output, letterbox, nms, to_blob, undo_letterbox
from ..inference.threading import ThreadBudget
from ..types import Detection
from .base import VehicleDetector

# COCO ids we care about. Bicycles are included because a society gate counts
# them, but they carry no plate, so the pipeline gates the plate stages on
# class as well as size.
DEFAULT_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloOnnxVehicleDetector(VehicleDetector):
    name = "yolo_onnx"

    def __init__(
        self,
        artifact: str | Path,
        input_size: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.35,
        nms_iou: float = 0.5,
        classes: dict[int, str] | None = None,
        backend: str = "auto",
        threads: ThreadBudget | None = None,
        cache_dir: str | Path | None = None,
        max_detections: int = 50,
    ):
        self._input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(conf_threshold)
        self.nms_iou = float(nms_iou)
        self.classes = {int(k): v for k, v in (classes or DEFAULT_CLASSES).items()}
        self.max_detections = int(max_detections)
        self.backend = build_backend(artifact, backend, threads, cache_dir)
        # Reused across frames so the hot path does not allocate a fresh
        # letterbox canvas 6 times a second per camera.
        self._canvas: np.ndarray | None = None

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    def detect(self, frame: np.ndarray, roi_offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []

        padded, scale, pad = letterbox(frame, self._input_size, out=self._canvas)
        self._canvas = padded
        outputs = self.backend.run_single(to_blob(padded))

        boxes, scores, class_ids = decode_yolo_output(outputs[0], self.conf_threshold)
        if boxes.shape[0] == 0:
            return []

        wanted = np.isin(class_ids, list(self.classes))
        if not wanted.any():
            return []
        boxes, scores, class_ids = boxes[wanted], scores[wanted], class_ids[wanted]

        boxes = undo_letterbox(boxes, scale, pad, frame.shape[:2])

        detections: list[Detection] = []
        off_x, off_y = roi_offset
        # NMS per class: a car boxed inside a truck box is a real situation at a
        # gate and cross-class suppression would drop one of them.
        for cls in np.unique(class_ids):
            idx = np.where(class_ids == cls)[0]
            for k in nms(boxes[idx], scores[idx], self.nms_iou):
                j = idx[k]
                x1, y1, x2, y2 = boxes[j]
                detections.append(
                    Detection(
                        bbox=(int(x1) + off_x, int(y1) + off_y, int(x2) + off_x, int(y2) + off_y),
                        confidence=float(scores[j]),
                        class_id=int(cls),
                        class_name=self.classes.get(int(cls), "vehicle"),
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_detections]

    def warmup(self, n: int = 2) -> float:
        return self.backend.warmup(n)

    def close(self) -> None:
        self.backend.close()


class NullVehicleDetector(VehicleDetector):
    """Plate-only mode: treat the whole frame as one vehicle.

    Useful for a tightly-framed gate camera where the vehicle always fills the
    ROI, and as an ablation when diagnosing whether the vehicle stage is
    costing accuracy rather than adding it.
    """

    name = "null"

    def __init__(self, class_name: str = "vehicle", **_ignored):
        self.class_name = class_name

    @property
    def input_size(self) -> tuple[int, int]:
        return (0, 0)

    def detect(self, frame: np.ndarray, roi_offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        off_x, off_y = roi_offset
        return [
            Detection(
                bbox=(off_x, off_y, off_x + w, off_y + h),
                confidence=1.0,
                class_id=-1,
                class_name=self.class_name,
            )
        ]
