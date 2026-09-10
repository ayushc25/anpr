"""Shared vocabulary for the AI pipeline.

Every coordinate in these structures is in FULL-FRAME pixel space, even when
the model that produced it ran on a crop. Stages that run on crops take an
``offset`` and map their results back before returning, so no downstream code
ever has to remember which coordinate space it is holding.

This module must stay dependency-free apart from numpy: `ai/` is a pure
library that has to import cleanly inside a test or an offline eval script
with no database, no FastAPI and no config loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

BBox = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass(slots=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(slots=True)
class Track:
    """A vehicle followed across frames. Owned by the tracker, read by everyone."""

    track_id: int
    bbox: BBox
    class_name: str
    confidence: float
    age: int = 0
    centroid_history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass(slots=True)
class PlateCandidate:
    """A localized plate. ``quad`` is present only for models that emit corners;
    without it the caller falls back to an axis-aligned crop of ``bbox``."""

    bbox: BBox
    confidence: float
    quad: Optional[np.ndarray] = None  # (4, 2) float32, clockwise from top-left


@dataclass(slots=True)
class PlateObservation:
    """One *look* at a plate: localized, cropped and quality-scored, but not
    necessarily recognized.

    The distinction from ``PlateRead`` is the whole point. Localizing and
    scoring a plate costs a fraction of recognizing it, so the cascade takes
    many more observations than it takes reads, and uses the cheap ones to
    decide which expensive ones are worth paying for.

    ``crop`` is retained only for the top-K observations of a track (see
    ``TrackState.note_observation``) and is used for the event image. It is
    NOT a pending work item: nothing ever recognizes an observation after the
    frame it came from. See ``video/ocr_scheduler`` for why.
    """

    frame_idx: int
    ts: float
    width: int
    height: int
    quality: float
    det_confidence: float
    #: The plate crop as the detector framed it — deskewed if a quad was
    #: available, otherwise the padded axis-aligned crop. NEVER the enhanced
    #: OCR input: keeping the two apart is what lets a debug dump show whether
    #: a misread came from the crop or from the enhancement.
    crop: Optional[np.ndarray] = None
    #: The unenhanced, un-deskewed axis-aligned crop, retained separately so
    #: an operator can see what the camera actually saw.
    raw_crop: Optional[np.ndarray] = None
    #: 64-bit dHash of the crop. Tells "the vehicle moved" from "the sensor
    #: flipped a few bits", which is what makes diverse retention and
    #: correlated-evidence discounting possible. 0 means not computed.
    appearance: int = 0
    #: Whether this observation was the one an OCR call was spent on.
    ocr_ran: bool = False

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)

    def release(self) -> None:
        """Drop the pixel buffers, keeping the measurements.

        Called on eviction so a long track cannot accumulate crops. The
        numbers are what the approach signal and the audit trail need; the
        arrays are only needed while the observation is a retry candidate or
        the best image.
        """
        self.crop = None
        self.raw_crop = None


@dataclass(slots=True)
class PlateRead:
    """One recognizer output. ``raw_text`` is kept verbatim so an operator can
    always be shown what the model actually emitted, before normalization."""

    text: str
    confidence: float
    per_char_confidence: list[float] = field(default_factory=list)
    raw_text: str = ""
    #: Runner-up characters per position, best first, as (char, probability).
    #:
    #: Empty for recognizers that cannot produce it — EasyOCR reports one
    #: score per detected BOX, not per timestep, so there is no distribution
    #: to keep. Every consumer must therefore treat this as optional and
    #: degrade to ``per_char_confidence`` alone, which is what the pipeline
    #: did before this existed.
    per_char_alternatives: list[tuple[tuple[str, float], ...]] = field(default_factory=list)

    def char_confidence(self, i: int) -> float:
        """Per-character confidence with a safe fallback for recognizers that
        only report an aggregate score."""
        if 0 <= i < len(self.per_char_confidence):
            return self.per_char_confidence[i]
        return self.confidence

    def alternatives_at(self, i: int) -> tuple[tuple[str, float], ...]:
        if 0 <= i < len(self.per_char_alternatives):
            return self.per_char_alternatives[i]
        return ()

    @property
    def has_alternatives(self) -> bool:
        return any(self.per_char_alternatives)


@dataclass(slots=True)
class StageTiming:
    """Per-stage wall-clock, in milliseconds, for one processed frame."""

    decode: float = 0.0
    vehicle_detect: float = 0.0
    track: float = 0.0
    plate_detect: float = 0.0
    recognize: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class FrameStats:
    frame_idx: int = 0
    timing: StageTiming = field(default_factory=StageTiming)
    vehicles: int = 0
    plates_detected: int = 0
    plates_recognized: int = 0
    detector_ran: bool = False

    #: Plates localized and scored this frame — the denominator the OCR
    #: counters below are worth reading against. ``plates_observed`` minus
    #: ``ocr_run`` is the work the scheduler saved.
    plates_observed: int = 0
    ocr_run: int = 0
    ocr_waited: int = 0
    ocr_skipped_small: int = 0
    ocr_skipped_budget: int = 0
    ocr_skipped_stale: int = 0
    #: Plate candidates discarded as not geometrically on their vehicle.
    plates_rejected_geometry: int = 0
    #: Age of this frame at the moment processing started, in milliseconds.
    #: 0.0 when the caller did not supply one (offline replay, tests).
    frame_age_ms: float = 0.0
