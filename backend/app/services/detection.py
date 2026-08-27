"""
Vehicle detection (YOLO, COCO classes) + license-plate localization
(dedicated fine-tuned YOLO model) + OCR.

yolo26n.pt finds vehicles. license-plate-finetune-v1l.pt is a model
fine-tuned specifically to find the plate rectangle within a vehicle crop,
used as the primary localization method. A contour/edge heuristic remains
as a fallback for on the rare frame the plate model misses.
"""
import logging
import re
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("anpr.detection")

from ..config import (
    YOLO_MODEL_PATH,
    VEHICLE_CLASS_IDS,
    VEHICLE_CONF_THRESHOLD,
    PLATE_MODEL_PATH,
    PLATE_CONF_THRESHOLD,
)

_model = None
_model_lock = threading.Lock()
_plate_model = None
_plate_model_lock = threading.Lock()
_ocr_reader = None
_ocr_lock = threading.Lock()

COCO_VEHICLE_NAMES = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

PLATE_RE = re.compile(r"[A-Z0-9]{5,12}")
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Hue band upper-bounds (OpenCV H range 0-179) used for nearest-match
# naming of a sampled dominant color, checked in order; a hue at or past
# the last bound wraps back to red (red spans both ends of the hue circle).
# Neutrals (white/black/gray/silver) are separated by saturation/value
# before falling back to this hue-based lookup for chromatic colors.
_HUE_NAMES = [
    (10, "red"), (25, "orange"), (35, "yellow"), (85, "green"),
    (100, "teal"), (130, "blue"), (150, "purple"), (170, "pink"),
]


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO
                _model = YOLO(YOLO_MODEL_PATH)
    return _model


def get_plate_model():
    global _plate_model
    if _plate_model is None:
        with _plate_model_lock:
            if _plate_model is None:
                from ultralytics import YOLO
                _plate_model = YOLO(PLATE_MODEL_PATH)
    return _plate_model


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                import easyocr
                _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def detect_vehicles(frame: np.ndarray):
    """Returns list of dicts: {bbox:(x1,y1,x2,y2), conf, vehicle_type}"""
    model = get_model()
    results = model.predict(frame, verbose=False, conf=VEHICLE_CONF_THRESHOLD)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "vehicle_type": COCO_VEHICLE_NAMES.get(cls_id, "vehicle"),
            })
    return detections


def _vehicle_crop(frame: np.ndarray, bbox):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _heuristic_plate_box(crop_shape):
    """Fallback plate location: lower-middle portion of the vehicle box,
    used when contour search finds no plate-shaped candidate."""
    h, w = crop_shape[:2]
    y1 = int(h * 0.55)
    return (0, y1, w, h - y1)


def _model_plate_candidates(crop: np.ndarray):
    """Primary plate localization: run the fine-tuned plate detector on the
    vehicle crop. Returns boxes (in crop coordinates) sorted by detector
    confidence, each padded slightly so OCR doesn't clip plate edge chars."""
    try:
        model = get_plate_model()
        results = model.predict(crop, verbose=False, conf=PLATE_CONF_THRESHOLD)
    except Exception:
        logger.exception("plate model inference failed")
        return []

    h, w = crop.shape[:2]
    found = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            pad_x = (x2 - x1) * 0.08
            pad_y = (y2 - y1) * 0.15
            bx1 = max(int(x1 - pad_x), 0)
            by1 = max(int(y1 - pad_y), 0)
            bx2 = min(int(x2 + pad_x), w)
            by2 = min(int(y2 + pad_y), h)
            found.append((conf, (bx1, by1, bx2, by2)))
    found.sort(key=lambda f: f[0], reverse=True)
    return [b for _, b in found[:3]]


def _contour_plate_candidates(gray: np.ndarray):
    """Heuristic plate localization via edge/contour search: look for
    rectangular regions with a plate-like aspect ratio in the lower half
    of the vehicle crop, sorted by area (largest first)."""
    h, w = gray.shape[:2]
    if h < 20 or w < 20:
        return []
    blur = cv2.bilateralFilter(gray, 9, 30, 30)
    edges = cv2.Canny(blur, 60, 180)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch <= 0 or cw <= 0:
            continue
        aspect = cw / ch
        area_ratio = (cw * ch) / (w * h)
        if 1.8 <= aspect <= 6.0 and 0.008 <= area_ratio <= 0.3 and y >= h * 0.25:
            candidates.append((x, y, x + cw, y + ch))
    candidates.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return candidates[:3]


def _preprocess_for_ocr(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    h, w = gray.shape[:2]
    if w < 220:
        scale = 220 / max(w, 1)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _ocr_roi(roi: np.ndarray):
    """Runs OCR on a single ROI, returns the best plate-shaped reading (text, conf)."""
    if roi is None or roi.size == 0:
        return None, 0.0
    try:
        processed = _preprocess_for_ocr(roi)
        reader = get_ocr_reader()
        results = reader.readtext(processed, allowlist=OCR_ALLOWLIST)
    except Exception:
        logger.exception("OCR read failed")
        return None, 0.0

    best_text, best_conf = None, 0.0
    for _, text, conf in results:
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        if PLATE_RE.fullmatch(cleaned) and conf > best_conf:
            best_text, best_conf = cleaned, conf
    if best_text is None:
        for _, text, conf in results:
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            if len(cleaned) >= 5 and conf > best_conf:
                best_text, best_conf = cleaned, conf
    return best_text, best_conf


def read_plate(frame: np.ndarray, bbox):
    """Best-effort plate OCR for one vehicle bbox in one frame.

    Localizes the plate with the fine-tuned plate-detector model first
    (accurate, but can occasionally miss a frame); if it finds nothing,
    falls back to a contour/edge heuristic. Returns whichever OCR reading
    scored the highest confidence, plus the winning plate box in the
    original frame's coordinates (for plate-color sampling), or None if no
    candidate box produced a reading. A single frame rarely captures the
    plate perfectly (motion blur, angle, glare) - callers should aggregate
    this across several frames of the same tracked vehicle and keep the
    best result rather than trusting any one frame's reading.
    """
    crop = _vehicle_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None, 0.0, None

    boxes = _model_plate_candidates(crop)
    if not boxes:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        boxes = _contour_plate_candidates(gray)
        if not boxes:
            boxes = [_heuristic_plate_box(crop.shape)]

    ox1, oy1 = max(bbox[0], 0), max(bbox[1], 0)
    best_text, best_conf, best_box_abs = None, 0.0, None
    for (bx1, by1, bx2, by2) in boxes:
        roi = crop[by1:by2, bx1:bx2]
        text, conf = _ocr_roi(roi)
        if text and conf > best_conf:
            best_text, best_conf = text, conf
            best_box_abs = (ox1 + bx1, oy1 + by1, ox1 + bx2, oy1 + by2)
    return best_text, best_conf, best_box_abs


def _hue_to_name(hue: float) -> str:
    for threshold, name in _HUE_NAMES:
        if hue < threshold:
            return name
    return "red"


def _classify_bgr(bgr, ref_v: Optional[float] = None) -> str:
    """Classifies a sampled color by hue/saturation/value, optionally
    normalizing value relative to `ref_v` - the brightest level actually
    seen in the same crop. Night/low-light or underexposed footage can sit
    entirely below 100/255 in absolute brightness, which without this
    would misclassify every color (including genuinely white/bright
    surfaces) as black; scaling relative to the crop's own brightest pixels
    approximates a white-balance correction."""
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if ref_v and ref_v > 0:
        v = min(255, int(round(v * 235.0 / ref_v)))
    if v < 50:
        return "black"
    if s < 40:
        if v > 190:
            return "white"
        if v > 120:
            return "silver"
        return "gray"
    return _hue_to_name(h)


def _value_percentile(bgr_region: np.ndarray, pct: float) -> float:
    hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    return float(np.percentile(hsv[..., 2], pct))


def _dominant_from_pixels(pixels: np.ndarray, k: int):
    """K-means dominant color over a flat (N, 3) BGR pixel array, returned
    as the largest cluster's center."""
    pixels = pixels.reshape(-1, 3).astype(np.float32)
    k = min(k, len(pixels))
    if k < 1:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    try:
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    except cv2.error:
        return None
    counts = np.bincount(labels.flatten(), minlength=k)
    return centers[int(np.argmax(counts))]


def _dominant_bgr(region: np.ndarray, k: int = 3):
    """K-means dominant color of an image region. Downsamples first since
    we only need an approximate color, not per-pixel precision."""
    if region is None or region.size == 0:
        return None
    small = cv2.resize(region, (40, 40), interpolation=cv2.INTER_LINEAR)
    return _dominant_from_pixels(small, k)


def get_vehicle_color(frame: np.ndarray, bbox) -> Optional[str]:
    """Dominant body-paint color of a vehicle, sampled from the central
    band of its box to avoid windshield glare, wheels/shadow at the
    bottom, and background creeping in at the edges. Brightness is
    normalized against this same band's own brightest pixels so dim/
    underexposed footage doesn't read every color as black."""
    crop = _vehicle_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    band = crop[int(h * 0.2):int(h * 0.55), int(w * 0.1):int(w * 0.9)]
    if band.size == 0:
        return None
    dominant = _dominant_bgr(band)
    if dominant is None:
        return None
    return _classify_bgr(dominant, ref_v=_value_percentile(band, 95))


def get_plate_color(frame: np.ndarray, plate_box_abs) -> Optional[str]:
    """Dominant background color of a plate (e.g. white/yellow/green for
    India's private/commercial/EV plate categories).

    Samples the brightest-value pixels in the plate ROI rather than a
    fixed region: plate backgrounds are almost always lighter than the
    printed characters, so this is more robust to imprecise plate-box
    localization than a fixed corner/border crop (which can land on
    surrounding bodywork instead of the plate). Brightness is then
    normalized against the ROI's own brightest pixels, since plates are
    lightly reflective and read far dimmer than their true color under
    low ambient light."""
    if plate_box_abs is None:
        return None
    x1, y1, x2, y2 = plate_box_abs
    h, w = frame.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] < 4 or roi.shape[1] < 4:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[..., 2]
    ref_v = float(np.percentile(v_channel, 95))
    bg_threshold = np.percentile(v_channel, 60)
    bg_mask = v_channel >= bg_threshold
    bg_pixels = roi[bg_mask]
    if bg_pixels.size == 0:
        bg_pixels = roi.reshape(-1, 3)

    dominant = _dominant_from_pixels(bg_pixels, k=2)
    return _classify_bgr(dominant, ref_v=ref_v) if dominant is not None else None
