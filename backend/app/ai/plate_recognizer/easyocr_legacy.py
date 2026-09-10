"""EasyOCR, wrapped in the recognizer contract.

This is the prototype's recognizer. It is kept — not deleted — so the ONNX
recognizers can be A/B'd against it on the same clip set with a one-line
config change. It is not the Phase 1 default: it loads a torch runtime onto an
edge box and costs 60-200 ms per crop, against 1-3 ms for the ONNX heads.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from ..types import PlateRead
from .base import PlateRecognizer

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class EasyOcrRecognizer(PlateRecognizer):
    name = "easyocr_legacy"
    charset = ALLOWLIST

    def __init__(self, languages: tuple[str, ...] = ("en",), min_char_confidence: float = 0.30, **_ignored):
        self.languages = list(languages)
        self.min_char_confidence = float(min_char_confidence)
        self._reader = None
        self._lock = threading.Lock()

    @property
    def expects_grayscale(self) -> bool:
        return False

    def _get_reader(self):
        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    import easyocr

                    self._reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
        return self._reader

    def recognize(self, plate_image: np.ndarray) -> Optional[PlateRead]:
        if plate_image is None or plate_image.size == 0:
            return None
        results = self._get_reader().readtext(
            plate_image, allowlist=ALLOWLIST, detail=1, paragraph=False
        )
        if not results:
            return None
        # A plate crop can still yield two boxes on a stacked plate; join them
        # top-to-bottom, which is the reading order of an Indian two-line plate.
        results.sort(key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))
        text = "".join(str(r[1]) for r in results)
        confidences = [float(r[2]) for r in results if r[2] is not None]
        confidence = float(np.mean(confidences)) if confidences else 0.0
        # EasyOCR reports per-box, not per-character; spread the box score
        # across its characters so the validator still has something to weight.
        per_char: list[float] = []
        for r in results:
            per_char.extend([float(r[2])] * len(str(r[1])))
        return self.finalize(text, confidence, per_char)

    def warmup(self, n: int = 1) -> float:
        self._get_reader()
        return 0.0
