"""Vehicle body colour and plate background colour.

Ported from the prototype's ``services/detection.py``, which had this right:
the logic here is genuinely hard-won and worth keeping. What changed is only
that it now lives in the pure ``ai/`` library, so it is testable without a
model, a camera or a database.

Plate background colour is not decoration on an Indian gate — it is the
vehicle category. White is private, yellow commercial, green electric, red a
temporary registration. An operator scanning the events list uses it to spot a
commercial vehicle entering a residential society at a glance.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Hue band upper-bounds (OpenCV H range 0-179) used for nearest-match naming of
# a sampled dominant colour, checked in order; a hue at or past the last bound
# wraps back to red (red spans both ends of the hue circle). Neutrals
# (white/black/gray/silver) are separated by saturation/value before falling
# back to this hue-based lookup for chromatic colours.
HUE_NAMES: tuple[tuple[int, str], ...] = (
    (10, "red"), (25, "orange"), (35, "yellow"), (85, "green"),
    (100, "teal"), (130, "blue"), (150, "purple"), (170, "pink"),
)


def _hue_to_name(hue: float) -> str:
    for threshold, name in HUE_NAMES:
        if hue < threshold:
            return name
    return "red"


def classify_bgr(bgr, ref_value: Optional[float] = None) -> str:
    """Name a sampled BGR colour by hue/saturation/value.

    ``ref_value`` is the brightest level actually seen in the same crop.
    Night or underexposed footage can sit entirely below 100/255 in absolute
    brightness, which without this normalisation would classify every colour —
    including genuinely white surfaces — as black. Scaling against the crop's
    own brightest pixels approximates a white balance correction.
    """
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hue, saturation, value = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if ref_value and ref_value > 0:
        value = min(255, int(round(value * 235.0 / ref_value)))
    if value < 50:
        return "black"
    if saturation < 40:
        if value > 190:
            return "white"
        if value > 120:
            return "silver"
        return "gray"
    return _hue_to_name(hue)


def _value_percentile(region: np.ndarray, pct: float) -> float:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return float(np.percentile(hsv[..., 2], pct))


def _dominant_from_pixels(pixels: np.ndarray, k: int):
    """K-means dominant colour over a flat (N, 3) BGR array: the largest
    cluster's centre."""
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
    """Downsampled first: an approximate colour is all we need, and k-means on
    a full 1080p crop would dominate the frame budget."""
    if region is None or region.size == 0:
        return None
    small = cv2.resize(region, (40, 40), interpolation=cv2.INTER_LINEAR)
    return _dominant_from_pixels(small, k)


def vehicle_color(vehicle_crop: np.ndarray) -> Optional[str]:
    """Dominant body-paint colour, sampled from the central band of the
    vehicle box to avoid windshield glare above, wheels and shadow below, and
    background creeping in at the edges."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None
    height, width = vehicle_crop.shape[:2]
    band = vehicle_crop[int(height * 0.2):int(height * 0.55), int(width * 0.1):int(width * 0.9)]
    if band.size == 0:
        return None
    dominant = _dominant_bgr(band)
    if dominant is None:
        return None
    return classify_bgr(dominant, ref_value=_value_percentile(band, 95))


def plate_color(plate_crop: np.ndarray) -> Optional[str]:
    """Dominant BACKGROUND colour of a plate — white / yellow / green / red,
    which is the vehicle's registration category in India.

    Samples the brightest pixels in the crop rather than a fixed region: plate
    backgrounds are almost always lighter than the printed characters, so this
    survives an imprecise plate box far better than sampling a corner (which
    can land on surrounding bodywork).
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
    if plate_crop.shape[0] < 4 or plate_crop.shape[1] < 4:
        return None

    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    value_channel = hsv[..., 2]
    ref_value = float(np.percentile(value_channel, 95))
    background_mask = value_channel >= np.percentile(value_channel, 60)
    background_pixels = plate_crop[background_mask]
    if background_pixels.size == 0:
        background_pixels = plate_crop.reshape(-1, 3)

    dominant = _dominant_from_pixels(background_pixels, k=2)
    if dominant is None:
        return None
    return _snap_to_plate_category(classify_bgr(dominant, ref_value=ref_value))


#: Indian plate backgrounds are a CLOSED set, and each one is a legal vehicle
#: category. Neighbouring hue names must collapse onto the nearest real
#: category rather than be reported literally: calling a commercial plate
#: "orange" instead of "yellow" loses the only fact the colour carries.
PLATE_CATEGORY = {
    "yellow": "yellow", "orange": "yellow",              # commercial
    "white": "white", "silver": "white", "gray": "white",  # private
    "green": "green", "teal": "green",                   # electric
    "red": "red",                                        # temporary registration
    "black": "black",                                    # commercial self-drive
    "blue": "blue",                                      # diplomatic
}


def _snap_to_plate_category(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return PLATE_CATEGORY.get(name, name)
