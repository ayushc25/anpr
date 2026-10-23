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

#: Two boxes share a row when their vertical extents overlap by at least this
#: fraction of the shorter box's height. Relative, not a pixel gap, because
#: the crop arrives at whatever scale the plate was seen at.
ROW_OVERLAP = 0.5


def reading_order(results: list) -> list:
    """Sort EasyOCR boxes into plate reading order: rows top-to-bottom, boxes
    left-to-right within a row.

    Sorting by (top, left) alone breaks a single-line plate that EasyOCR split
    into two boxes: if the right-hand box's top edge sits a pixel higher, it
    sorts first and "DL5CU" + "1624" comes out as "1624DL5CU". Grouping into
    rows first keeps the stacked two-line plate working (its rows do not
    overlap) and fixes the split single line.
    """
    boxes = []
    for r in results:
        ys = [p[1] for p in r[0]]
        xs = [p[0] for p in r[0]]
        boxes.append((min(ys), max(ys), min(xs), r))
    boxes.sort(key=lambda b: (b[0] + b[1]) / 2.0)

    rows: list[list[tuple]] = []
    for box in boxes:
        top, bottom = box[0], box[1]
        for row in rows:
            row_top = min(b[0] for b in row)
            row_bottom = max(b[1] for b in row)
            overlap = min(bottom, row_bottom) - max(top, row_top)
            shorter = min(bottom - top, row_bottom - row_top)
            if shorter > 0 and overlap >= ROW_OVERLAP * shorter:
                row.append(box)
                break
        else:
            rows.append([box])

    rows.sort(key=lambda row: min(b[0] for b in row))
    return [b[3] for row in rows for b in sorted(row, key=lambda b: b[2])]


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
        results = reading_order(results)
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
