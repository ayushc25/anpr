from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import PlateCandidate


class PlateDetector(ABC):
    """Finds plates. Normally called on a vehicle crop, not a full frame —
    that is both faster and more accurate, because the plate occupies far more
    of the letterboxed tensor."""

    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[PlateCandidate]:
        """Returns candidates sorted by confidence, best first, with ``offset``
        added so coordinates are full-frame."""

    @property
    @abstractmethod
    def input_size(self) -> tuple[int, int]: ...

    def warmup(self, n: int = 2) -> float:
        return 0.0

    def close(self) -> None:
        pass
