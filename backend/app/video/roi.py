"""Region of interest for one camera.

Coordinates are normalized to 0..1 so a stored ROI survives a resolution
change — switching a camera from its main stream to its sub-stream must not
silently invalidate the polygon an operator drew.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

Point = tuple[float, float]


@dataclass
class RoiPolygon:
    """A gate area. An empty polygon means "the whole frame", which is the
    correct default for a camera nobody has configured yet."""

    points: list[Point]
    name: str = "roi"

    @classmethod
    def from_config(cls, geometry: Optional[Sequence], name: str = "roi") -> "RoiPolygon":
        points: list[Point] = []
        for point in geometry or []:
            if isinstance(point, dict):
                points.append((float(point["x"]), float(point["y"])))
            else:
                points.append((float(point[0]), float(point[1])))
        return cls(points=points, name=name)

    def to_config(self) -> list[dict]:
        return [{"x": round(x, 6), "y": round(y, 6)} for x, y in self.points]

    @property
    def is_empty(self) -> bool:
        return len(self.points) < 3

    # -- pixel-space cache -------------------------------------------------
    #
    # On a fixed camera the polygon and the frame shape never change, so the
    # projection to pixels is computed once and reused for the life of the
    # process. The cache is keyed on the points as well as the shape, so an
    # operator editing the polygon at runtime gets a correct answer rather
    # than a stale one.
    _pixel_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def _cache_key(self, frame_shape: tuple[int, int]) -> tuple:
        return (tuple(self.points), frame_shape[0], frame_shape[1])

    def pixels(self, frame_shape: tuple[int, int]) -> np.ndarray:
        key = self._cache_key(frame_shape)
        cached = self._pixel_cache.get(key)
        if cached is None:
            height, width = frame_shape[:2]
            cached = np.array(
                [(int(x * width), int(y * height)) for x, y in self.points], dtype=np.int32
            )
            cached.setflags(write=False)  # a shared array must not be mutated in place
            # Bounded: a camera sees one or two frame shapes in its lifetime,
            # and an edited polygon adds one entry per edit.
            if len(self._pixel_cache) > 8:
                self._pixel_cache.clear()
            self._pixel_cache[key] = cached
        return cached

    def contains(self, point: tuple[float, float], frame_shape: tuple[int, int]) -> bool:
        if self.is_empty:
            return True
        polygon = self.pixels(frame_shape)
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def overlap_ratio(self, bbox, frame_shape: tuple[int, int]) -> float:
        """Fraction of ``bbox`` inside the polygon.

        Used instead of a centroid test where a vehicle is large enough that
        its centre can sit outside a narrow gate ROI while most of the vehicle
        is inside it.
        """
        if self.is_empty:
            return 1.0
        x1, y1, x2, y2 = (int(v) for v in bbox)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.pixels(frame_shape)], 1)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        window = mask[y1:y2, x1:x2]
        return float(window.sum()) / float(window.size)

    def crop_bounds(self, frame_shape: tuple[int, int], pad: float = 0.05) -> tuple[int, int, int, int]:
        """Axis-aligned bounds of the polygon, padded.

        Running the detector on this crop rather than the full frame is one of
        the cheapest accuracy-and-speed wins available: a gate ROI is often
        ~40% of the frame, so the vehicle occupies far more of the letterboxed
        tensor.
        """
        height, width = frame_shape[:2]
        if self.is_empty:
            return (0, 0, width, height)
        polygon = self.pixels(frame_shape)
        x1, y1 = polygon.min(axis=0)
        x2, y2 = polygon.max(axis=0)
        pad_x, pad_y = int((x2 - x1) * pad), int((y2 - y1) * pad)
        return (
            max(0, int(x1) - pad_x),
            max(0, int(y1) - pad_y),
            min(width, int(x2) + pad_x),
            min(height, int(y2) + pad_y),
        )

    def crop(self, frame: np.ndarray, pad: float = 0.05) -> tuple[np.ndarray, tuple[int, int]]:
        """Returns (cropped frame, (offset_x, offset_y))."""
        if self.is_empty:
            return frame, (0, 0)
        x1, y1, x2, y2 = self.crop_bounds(frame.shape[:2], pad)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return frame, (0, 0)
        return frame[y1:y2, x1:x2], (x1, y1)

    def draw(self, frame: np.ndarray, color=(64, 190, 200), thickness: int = 2) -> None:
        if self.is_empty:
            return
        cv2.polylines(frame, [self.pixels(frame.shape[:2])], True, color, thickness)
