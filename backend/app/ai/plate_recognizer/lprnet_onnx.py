"""LPRNet-style CTC recognizer on ONNX.

Input is a fixed-size grayscale (or BGR, per config) plate crop; output is a
(T, C) logit matrix decoded greedily. This is the target recognizer once the
model is fine-tuned on harvested Indian plates — see scripts/harvest_dataset.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from ..inference.backend import build_backend
from ..inference.threading import ThreadBudget
from ..types import PlateRead
from .base import PlateRecognizer
from .ctc import ctc_greedy_decode, load_charset, softmax

DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class LprnetOnnxRecognizer(PlateRecognizer):
    name = "lprnet_onnx"

    def __init__(
        self,
        artifact: str | Path,
        charset: str | Path | None = None,
        input_size: tuple[int, int] = (96, 48),
        grayscale: bool = True,
        blank_first: bool = True,
        apply_softmax: bool = True,
        min_char_confidence: float = 0.40,
        backend: str = "auto",
        threads: ThreadBudget | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.charset = load_charset(str(charset) if charset else None, DEFAULT_CHARSET)
        self._input_size = (int(input_size[0]), int(input_size[1]))
        self._grayscale = bool(grayscale)
        self.blank_first = bool(blank_first)
        self.apply_softmax = bool(apply_softmax)
        self.min_char_confidence = float(min_char_confidence)
        self.backend = build_backend(artifact, backend, threads, cache_dir)

    @property
    def expects_grayscale(self) -> bool:
        return self._grayscale

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        width, height = self._input_size
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        if self._grayscale:
            if resized.ndim == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            tensor = resized[None, None, :, :]
        else:
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            tensor = resized[:, :, ::-1].transpose(2, 0, 1)[None]
        return np.ascontiguousarray((tensor.astype(np.float32) / 127.5) - 1.0)

    def _decode(self, raw: np.ndarray) -> Optional[PlateRead]:
        probs = softmax(raw.astype(np.float32)) if self.apply_softmax else raw.astype(np.float32)
        text, confidence, per_char = ctc_greedy_decode(
            probs, self.charset, blank_index=0 if self.blank_first else len(self.charset), blank_first=self.blank_first
        )
        if not text:
            return None
        # Drop reads where a single character is essentially a coin flip; the
        # validator can outvote a bad character, but only if other frames
        # actually contribute, and a read this weak mostly adds noise.
        if per_char and min(per_char) < self.min_char_confidence * 0.5:
            return None
        return self.finalize(text, confidence, per_char)

    def recognize(self, plate_image: np.ndarray) -> Optional[PlateRead]:
        if plate_image is None or plate_image.size == 0:
            return None
        outputs = self.backend.run_single(self._preprocess(plate_image))
        return self._decode(outputs[0])

    def recognize_batch(self, plate_images: Sequence[np.ndarray]) -> list[Optional[PlateRead]]:
        usable = [(i, im) for i, im in enumerate(plate_images) if im is not None and im.size]
        if not usable:
            return [None] * len(plate_images)
        batch = np.concatenate([self._preprocess(im) for _, im in usable], axis=0)
        try:
            outputs = self.backend.run_single(batch)[0]
        except Exception:
            # A model exported with a fixed batch dimension cannot batch; fall
            # back rather than losing the frame.
            return [self.recognize(im) for im in plate_images]
        results: list[Optional[PlateRead]] = [None] * len(plate_images)
        for slot, (index, _) in enumerate(usable):
            results[index] = self._decode(outputs[slot])
        return results

    def warmup(self, n: int = 2) -> float:
        return self.backend.warmup(n)

    def close(self) -> None:
        self.backend.close()
