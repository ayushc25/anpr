"""Pre/post-processing shared by the ONNX detectors.

Written in numpy rather than pulled from Ultralytics on purpose: the serving
path should not drag torch onto an edge box for a letterbox and an NMS.
"""
from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
    out: np.ndarray | None = None,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize preserving aspect ratio and pad to ``new_shape`` (w, h).

    Returns the padded image plus the scale and (left, top) padding needed to
    map boxes back to the original image.

    ``out`` lets a caller pass a pre-allocated buffer so the hot path does not
    allocate a new array per frame.
    """
    src_h, src_w = image.shape[:2]
    new_w, new_h = new_shape
    scale = min(new_w / src_w, new_h / src_h)
    resized_w, resized_h = int(round(src_w * scale)), int(round(src_h * scale))
    pad_w = (new_w - resized_w) // 2
    pad_h = (new_h - resized_h) // 2

    if out is None or out.shape[:2] != (new_h, new_w):
        out = np.empty((new_h, new_w, 3), dtype=np.uint8)
    out[:] = color

    interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    cv2.resize(image, (resized_w, resized_h), dst=out[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w], interpolation=interp)
    return out, scale, (pad_w, pad_h)


def to_blob(padded: np.ndarray) -> np.ndarray:
    """HWC BGR uint8 -> NCHW RGB float32 in [0,1], contiguous."""
    blob = cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0, swapRB=True, crop=False)
    return np.ascontiguousarray(blob, dtype=np.float32)


def undo_letterbox(
    boxes: np.ndarray, scale: float, pad: tuple[int, int], src_shape: tuple[int, int]
) -> np.ndarray:
    """Map xyxy boxes from letterboxed space back to source pixels, clipped."""
    if boxes.size == 0:
        return boxes
    pad_w, pad_h = pad
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_w) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_h) / scale
    h, w = src_shape
    # Assign the clipped result back explicitly: fancy indexing returns a
    # copy, so `np.clip(..., out=out[:, [0, 2]])` writes into a temporary and
    # silently leaves the boxes unclipped.
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h - 1)
    return out


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS on xyxy boxes. Returns kept indices, best score first."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def decode_yolo_output(
    raw: np.ndarray, conf_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLOv8/v11-style head into (xyxy, scores, class_ids).

    The exported head is (1, 4 + n_classes, n_anchors); some exporters emit it
    already transposed, so handle both rather than making the caller care.
    """
    pred = raw[0] if raw.ndim == 3 else raw
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # -> (n_anchors, 4 + n_classes)

    if pred.shape[1] < 5:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    class_scores = pred[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= conf_threshold
    if not mask.any():
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    return (
        xywh_to_xyxy(pred[mask, :4].astype(np.float32)),
        scores[mask].astype(np.float32),
        class_ids[mask].astype(int),
    )


def crop(image: np.ndarray, box, pad: float = 0.0) -> np.ndarray | None:
    """Safe crop with optional proportional padding. None if degenerate."""
    x1, y1, x2, y2 = (int(v) for v in box)
    if pad > 0:
        w, h = x2 - x1, y2 - y1
        x1 -= int(w * pad)
        x2 += int(w * pad)
        y1 -= int(h * pad)
        y2 += int(h * pad)
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image[y1:y2, x1:x2]


def warp_quad(image: np.ndarray, quad: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """Perspective-rectify a plate from its four corners.

    A plate seen at an angle is the normal case at a gate, and deskewing before
    recognition is worth several points of accuracy on its own.
    """
    dst_w, dst_h = out_size
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
