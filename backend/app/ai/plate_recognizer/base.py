from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from ..types import PlateRead
from . import postprocess


class PlateRecognizer(ABC):
    """The interchangeable stage.

    Implementations return what the model saw. They must NOT decide whether a
    read is trustworthy — that is the multi-frame validator's job, and moving
    the decision into a single-frame recognizer is exactly the mistake this
    architecture exists to avoid.
    """

    name: str = "base"
    charset: str = ""

    @abstractmethod
    def recognize(self, plate_image: np.ndarray) -> Optional[PlateRead]:
        """``plate_image`` is a tight, ideally deskewed BGR plate crop.
        Returns None when nothing legible came out."""

    def recognize_batch(self, plate_images: Sequence[np.ndarray]) -> list[Optional[PlateRead]]:
        """Override where the backend supports real batching; several plates
        pending in one frame is common at a gate with a queue."""
        return [self.recognize(image) for image in plate_images]

    @property
    def expects_grayscale(self) -> bool:
        return True

    def warmup(self, n: int = 2) -> float:
        return 0.0

    def close(self) -> None:
        pass

    # -- shared helper -----------------------------------------------------
    @staticmethod
    def finalize(
        raw_text: str,
        confidence: float,
        per_char: list[float],
        alternatives: Optional[list[tuple[tuple[str, float], ...]]] = None,
    ) -> Optional[PlateRead]:
        """Normalize a raw model string into a PlateRead, keeping per-character
        confidences — and the runner-up characters, where the recognizer
        produced them — aligned with the characters that survived
        normalization.

        The alignment is the delicate part and it is done by carrying both
        lists through the SAME filter as the text. Confidences and
        alternatives that drift out of step with their characters are worse
        than absent: the grammar decoder would then substitute at a position
        using another position's evidence.
        """
        if not raw_text:
            return None
        alternatives = alternatives or []
        kept: list[float] = []
        kept_alts: list[tuple[tuple[str, float], ...]] = []
        normalized_chars: list[str] = []
        for i, char in enumerate(raw_text.upper()):
            if char.isalnum():
                normalized_chars.append(char)
                kept.append(per_char[i] if i < len(per_char) else confidence)
                kept_alts.append(alternatives[i] if i < len(alternatives) else ())
        text = postprocess.normalize("".join(normalized_chars))
        if len(text) < postprocess.MIN_LEN:
            return None
        # normalize() can drop a leading hologram token; realign from the right,
        # where the characters are the ones we actually kept.
        per_char_conf = kept[-len(text):] if len(kept) >= len(text) else kept + [confidence] * (len(text) - len(kept))
        per_char_alts = (
            kept_alts[-len(text):]
            if len(kept_alts) >= len(text)
            else kept_alts + [()] * (len(text) - len(kept_alts))
        )
        return PlateRead(
            text=text,
            confidence=float(confidence),
            per_char_confidence=per_char_conf,
            raw_text=raw_text,
            per_char_alternatives=per_char_alts,
        )
