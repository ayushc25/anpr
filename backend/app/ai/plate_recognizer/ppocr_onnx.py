"""PP-OCR mobile recognition head on ONNX — the Phase 1 default.

Chosen as the immediate replacement for EasyOCR because it is ~10 MB, runs in
1-3 ms per crop on CPU, and needs no fine-tuning to beat a generic OCR on
plate-shaped text. It differs from LPRNet only in preprocessing: PP-OCR keeps
a fixed height and a variable width (padded to a maximum), which preserves the
aspect ratio of a plate instead of squashing it.

THE DICTIONARY IS THE MODEL'S, NOT THE PLATE'S
----------------------------------------------
The published English head is trained on the full printable-ASCII dictionary
(``en_dict.txt``, 95 characters) plus a space: 97 output classes with the
blank. ``charset`` must be that dictionary, in that order — class ``k`` means
``dictionary[k - 1]`` and nothing else. Handing it the 36 plate characters
instead decodes the digits correctly by coincidence (they come first in both)
and every letter as the wrong one, silently. ``_check_classes`` now refuses to
run on a mismatch.

The plate alphabet is then applied by PROJECTION, not by filtering the decoded
string: before decoding, each output column is folded into 1 + 36 classes —
lowercase into its uppercase letter, every punctuation mark and the space into
the blank. Rows still sum to one, so the per-character probabilities and the
runner-up alternatives the grammar decoder depends on stay real probabilities
over the plate alphabet, rather than a distribution with holes in it.
"""
from __future__ import annotations

import time
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
        #: The model's own dictionary. Decoding always happens over the plate
        #: alphabet; this is only what the output columns MEAN.
        self.dictionary = load_charset(str(charset) if charset else None, DEFAULT_CHARSET)
        self.charset = DEFAULT_CHARSET
        self._projection: np.ndarray | None = None
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

        # As PaddleOCR's own resize_norm_img: BGR in (the model was trained on
        # cv2-loaded images, no channel swap), normalized first, THEN padded
        # with zeros — which is mid-grey after normalization, not black. Black
        # padding is a hard edge the model can read as a stroke.
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        canvas = np.zeros((self.height, self.max_width, 3), dtype=np.float32)
        canvas[:, :target_w] = normalized
        return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])

    def _check_classes(self, num_classes: int) -> np.ndarray:
        """Build (once) the matrix folding the model's classes onto blank + the
        plate alphabet, after checking the dictionary actually describes them.

        PP-OCR exports add a trailing space class when trained with
        use_space_char, so both ``len + 1`` and ``len + 2`` are legitimate.
        Anything else means the dictionary file is not this model's, and every
        read would be garbage with plausible-looking confidences.
        """
        if self._projection is not None and self._projection.shape[0] == num_classes:
            return self._projection
        symbols = list(self.dictionary)
        if num_classes == len(symbols) + 2:
            symbols.append(" ")
        elif num_classes != len(symbols) + 1:
            raise ValueError(
                f"ppocr_onnx: model emits {num_classes} classes but the dictionary has "
                f"{len(symbols)} characters (expected {num_classes - 1} or {num_classes - 2}). "
                "plate_recognizer.charset must be the model's own dictionary file "
                "(en_dict.txt for the English head); see scripts/fetch_recognizer.py."
            )
        projection = np.zeros((num_classes, len(self.charset) + 1), dtype=np.float32)
        projection[0, 0] = 1.0
        for k, symbol in enumerate(symbols, start=1):
            target = self.charset.find(symbol.upper())
            # Not a plate character -> "nothing emitted here", i.e. blank.
            projection[k, target + 1 if target >= 0 else 0] = 1.0
        self._projection = projection
        return projection

    def recognize(self, plate_image: np.ndarray) -> Optional[PlateRead]:
        if plate_image is None or plate_image.size == 0:
            return None
        outputs = self.backend.run_single(self._preprocess(plate_image))
        raw = outputs[0].astype(np.float32)
        # PP-OCR exports usually already carry a softmax; applying a second one
        # only flattens the distribution, so detect it rather than assume.
        probs = raw if _looks_normalized(raw) else softmax(raw)
        if probs.ndim == 3:
            probs = probs[0]
        probs = probs @ self._check_classes(probs.shape[-1])
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
        """A real forward pass at the real input shape. The generic backend
        warm-up substitutes 1 for dynamic dimensions — a 1-pixel-wide image —
        and it also would not reach ``_check_classes``, which is the point: a
        wrong dictionary must stop the worker at startup, not at the first car."""
        dummy = np.full((self.height, self.max_width, 3), 255, dtype=np.uint8)
        elapsed = 0.0
        for _ in range(max(1, n)):
            start = time.perf_counter()
            self.recognize(dummy)
            elapsed = (time.perf_counter() - start) * 1000.0
        return elapsed

    def close(self) -> None:
        self.backend.close()


def _looks_normalized(array: np.ndarray) -> bool:
    sample = array[0] if array.ndim == 3 else array
    if sample.size == 0:
        return False
    return bool(sample.min() >= 0.0 and abs(float(sample[0].sum()) - 1.0) < 0.05)
