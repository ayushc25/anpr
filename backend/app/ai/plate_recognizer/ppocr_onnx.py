"""PP-OCR mobile recognition head on ONNX — the Phase 1 default.

Chosen as the immediate replacement for EasyOCR because it is ~10 MB, runs in
1-3 ms per crop on CPU, and needs no fine-tuning to beat a generic OCR on
plate-shaped text. It differs from LPRNet only in preprocessing: PP-OCR keeps
a fixed height and a variable width (padded to a maximum), which preserves the
aspect ratio of a plate instead of squashing it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..inference.backend import build_backend
from ..inference.threading import ThreadBudget
from ..types import PlateRead
from .base import PlateRecognizer
from .ctc import DEFAULT_TOP_K, ctc_decode_with_alternatives, load_charset, softmax, summarize

#: PP-OCR dictionaries put the blank at index 0 and the characters after it.
DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class PpOcrOnnxRecognizer(PlateRecognizer):
    name = "ppocr_onnx"

    def __init__(
        self,
        artifact: str | Path,
        charset: str | Path | None = None,
        height: int = 48,
        max_width: int = 320,
        min_char_confidence: float = 0.40,
        decode_top_k: int = DEFAULT_TOP_K,
        backend: str = "auto",
        threads: ThreadBudget | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.charset = load_charset(str(charset) if charset else None, DEFAULT_CHARSET)
        self.height = int(height)
        self.max_width = int(max_width)
        self.min_char_confidence = float(min_char_confidence)
        #: Runner-up characters kept per position, for grammar-aware decoding.
        #: 0 disables it and restores the plain greedy behaviour.
        self.decode_top_k = int(decode_top_k)
        self.backend = build_backend(artifact, backend, threads, cache_dir)

    @property
    def expects_grayscale(self) -> bool:
        return False

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        target_w = min(self.max_width, max(16, int(round(w * self.height / max(h, 1)))))
        resized = cv2.resize(image, (target_w, self.height), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((self.height, self.max_width, 3), dtype=np.uint8)
        canvas[:, :target_w] = resized

        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        tensor = (tensor / 255.0 - 0.5) / 0.5
        return np.ascontiguousarray(tensor)

    def recognize(self, plate_image: np.ndarray) -> Optional[PlateRead]:
        if plate_image is None or plate_image.size == 0:
            return None
        outputs = self.backend.run_single(self._preprocess(plate_image))
        raw = outputs[0].astype(np.float32)
        # PP-OCR exports usually already carry a softmax; applying a second one
        # only flattens the distribution, so detect it rather than assume.
        probs = raw if _looks_normalized(raw) else softmax(raw)
        # The alternatives-aware decoder, so the runner-up characters survive
        # the call. Same collapsing rules as ctc_greedy_decode, so the string
        # is identical — only the discarded evidence differs.
        posteriors = ctc_decode_with_alternatives(
            probs, self.charset, blank_index=0, blank_first=True, top_k=self.decode_top_k
        )
        text, confidence, per_char = summarize(posteriors)
        if not text:
            return None
        if per_char and min(per_char) < self.min_char_confidence * 0.5:
            return None
        return self.finalize(
            text, confidence, per_char, [p.alternatives for p in posteriors]
        )

    def warmup(self, n: int = 2) -> float:
        return self.backend.warmup(n)

    def close(self) -> None:
        self.backend.close()


def _looks_normalized(array: np.ndarray) -> bool:
    sample = array[0] if array.ndim == 3 else array
    if sample.size == 0:
        return False
    return bool(sample.min() >= 0.0 and abs(float(sample[0].sum()) - 1.0) < 0.05)
