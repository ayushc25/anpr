from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Detection


class VehicleDetector(ABC):
    """Stateless. One frame in, detections out, always in full-frame coords."""

    name: str = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray, roi_offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        """``frame`` may be an ROI crop. ``roi_offset`` is the (x, y) of that
        crop within the full frame and is added back before returning, so
        callers never handle two coordinate spaces."""

    @property
    @abstractmethod
    def input_size(self) -> tuple[int, int]:
        """(width, height) the model expects."""

    def warmup(self, n: int = 2) -> float:
        return 0.0

    def close(self) -> None:
        pass
