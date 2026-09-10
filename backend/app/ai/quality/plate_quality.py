"""Plate crop quality, 0..1.

Two jobs downstream: gate the recognizer (don't spend CPU on a crop nothing
can read) and weight the read when it votes. Both matter, which is why this
returns a continuous score and a component breakdown rather than a boolean —
the breakdown is what an operator sees when asking why a plate was missed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: Below this the character strokes are smaller than the sensor noise and no
#: recognizer, fine-tuned or not, recovers the plate. Surveys should target
#: >=100 px at the trigger line.
MIN_USABLE_WIDTH_PX = 45
GOOD_WIDTH_PX = 110

#: Variance of the Laplacian at which a plate crop is considered sharp.
SHARP_VARIANCE = 150.0

#: Single-line Indian plates sit near 4.5:1, stacked plates near 2:1.
ASPECT_LOW, ASPECT_HIGH = 2.0, 5.5
ASPECT_STACKED = 2.0


@dataclass(frozen=True)
class QualityBreakdown:
    score: float
    width_px: int
    sharpness: float
    resolution_score: float
    sharpness_score: float
    aspect_score: float
    exposure_score: float

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "width_px": self.width_px,
            "sharpness": round(self.sharpness, 2),
            "components": {
                "resolution": round(self.resolution_score, 3),
                "sharpness": round(self.sharpness_score, 3),
                "aspect": round(self.aspect_score, 3),
                "exposure": round(self.exposure_score, 3),
            },
        }


def _sat(value: float) -> float:
    return max(0.0, min(1.0, value))


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _aspect_score(width: int, height: int) -> float:
    if height <= 0:
        return 0.0
    ratio = width / height
    if ASPECT_LOW <= ratio <= ASPECT_HIGH:
        return 1.0
    # Stacked plates are legitimate (two-wheelers, many commercial vehicles),
    # so a squarer crop is penalised gently rather than rejected.
    if 1.2 <= ratio < ASPECT_LOW:
        return 0.75
    if ASPECT_HIGH < ratio <= 7.0:
        return 0.6
    return 0.2


def _exposure_score(gray: np.ndarray) -> float:
    """Penalise crops that are blown out (IR glare on a retroreflective plate,
    the classic night failure) or crushed to black."""
    mean = float(gray.mean())
    std = float(gray.std())
    if mean < 25 or mean > 235:
        return 0.15
    centred = 1.0 - abs(mean - 128.0) / 128.0
    contrast = _sat(std / 55.0)
    return _sat(0.45 * centred + 0.55 * contrast)


def score_plate(crop: np.ndarray) -> QualityBreakdown:
    """Composite score for a plate crop. Cheap enough to run on every
    candidate — a few hundred microseconds."""
    if crop is None or crop.size == 0:
        return QualityBreakdown(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    sharpness = variance_of_laplacian(gray)
    resolution_score = _sat((width - MIN_USABLE_WIDTH_PX) / (GOOD_WIDTH_PX - MIN_USABLE_WIDTH_PX))
    sharpness_score = _sat(sharpness / SHARP_VARIANCE)
    aspect = _aspect_score(width, height)
    exposure = _exposure_score(gray)

    score = _sat(
        0.40 * resolution_score
        + 0.30 * sharpness_score
        + 0.15 * aspect
        + 0.15 * exposure
    )
    return QualityBreakdown(
        score=score,
        width_px=int(width),
        sharpness=sharpness,
        resolution_score=resolution_score,
        sharpness_score=sharpness_score,
        aspect_score=aspect,
        exposure_score=exposure,
    )


def enhance_for_ocr(crop: np.ndarray, target_height: int = 64) -> np.ndarray:
    """Light, deterministic clean-up before recognition.

    Upscaling a small plate and equalising local contrast reliably helps; heavy
    denoise or thresholding does not, and costs accuracy on the crops that were
    already fine. Keep this conservative.
    """
    if crop is None or crop.size == 0:
        return crop
    height, width = crop.shape[:2]
    if height < target_height:
        scale = target_height / height
        crop = cv2.resize(crop, (int(width * scale), target_height), interpolation=cv2.INTER_CUBIC)

    if crop.ndim == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
