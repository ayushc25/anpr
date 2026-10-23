"""Recognizer output -> plate text, for the two ways it went wrong.

PP-OCR: the published English head has 97 output classes (blank + a 95-char
printable-ASCII dictionary + space). Decoding it against the 36 plate
characters mapped digits correctly by coincidence and every letter to the
wrong character, with no error anywhere.

EasyOCR: boxes were joined in (top, left) order, so a single-line plate split
into two boxes came out back to front when the right box sat a pixel higher.
"""
from __future__ import annotations

import string

import numpy as np
import pytest

from backend.app.ai.plate_recognizer import ppocr_onnx
from backend.app.ai.plate_recognizer.easyocr_legacy import reading_order
from backend.app.ai.plate_recognizer.ppocr_onnx import PpOcrOnnxRecognizer

#: PaddleOCR's en_dict.txt, in file order — its last line is a space. The
#: model adds one more space class (use_space_char), hence 97 classes.
EN_DICT = string.digits + ":;<=>?@" + string.ascii_uppercase + "[\\]^_`" + string.ascii_lowercase + "{|}~!\"#$%&'()*+,-./ "


class _FakeBackend:
    """Returns a fixed (1, T, C) probability tensor, whatever the input."""

    def __init__(self, probs: np.ndarray):
        self.probs = probs

    def run_single(self, _tensor):
        return [self.probs[None]]

    def close(self):
        pass


def _recognizer(monkeypatch, probs: np.ndarray, dictionary: str, tmp_path) -> PpOcrOnnxRecognizer:
    path = tmp_path / "dict.txt"
    path.write_text("\n".join(dictionary), encoding="utf-8")
    monkeypatch.setattr(ppocr_onnx, "build_backend", lambda *a, **k: _FakeBackend(probs))
    return PpOcrOnnxRecognizer(artifact="unused.onnx", charset=path)


def _emit(dictionary: str, text: str, n_classes: int, extra: dict | None = None) -> np.ndarray:
    """A clean CTC path: each character followed by a blank."""
    rows = []
    for char in text:
        row = np.full(n_classes, 1e-4, dtype=np.float32)
        row[1 + dictionary.index(char)] = 0.95
        rows.append(row)
        blank = np.full(n_classes, 1e-4, dtype=np.float32)
        blank[0] = 0.95
        rows.append(blank)
    probs = np.stack(rows)
    for (t, cls), p in (extra or {}).items():
        probs[t, cls] = p
    return probs / probs.sum(axis=1, keepdims=True)


class TestPpOcrDictionary:
    def test_letters_decode_through_the_models_own_dictionary(self, monkeypatch, tmp_path):
        probs = _emit(EN_DICT, "DL7CP8161", len(EN_DICT) + 2)
        read = _recognizer(monkeypatch, probs, EN_DICT, tmp_path).recognize(np.zeros((40, 180, 3), np.uint8))
        assert read is not None
        assert read.text == "DL7CP8161"

    def test_lowercase_folds_into_its_letter(self, monkeypatch, tmp_path):
        probs = _emit(EN_DICT, "DL7cP8161", len(EN_DICT) + 2)
        read = _recognizer(monkeypatch, probs, EN_DICT, tmp_path).recognize(np.zeros((40, 180, 3), np.uint8))
        assert read.text == "DL7CP8161"

    def test_punctuation_is_not_a_character(self, monkeypatch, tmp_path):
        """A hyphen between groups becomes 'nothing here', not a stray glyph."""
        probs = _emit(EN_DICT, "DL-7CP8161", len(EN_DICT) + 2)
        read = _recognizer(monkeypatch, probs, EN_DICT, tmp_path).recognize(np.zeros((40, 180, 3), np.uint8))
        assert read.text == "DL7CP8161"

    def test_runner_up_survives_as_a_plate_character(self, monkeypatch, tmp_path):
        """grammar_decode depends on the alternatives; a lowercase runner-up
        must arrive as its uppercase letter with its probability intact."""
        n = len(EN_DICT) + 2
        # Position 2 (row 4): '7' wins, 'z' is the runner-up.
        probs = _emit(EN_DICT, "DL7CP8161", n, extra={(4, 1 + EN_DICT.index("z")): 0.30})
        read = _recognizer(monkeypatch, probs, EN_DICT, tmp_path).recognize(np.zeros((40, 180, 3), np.uint8))
        assert read.per_char_alternatives[2][0][0] == "Z"

    def test_a_plate_alphabet_file_is_refused(self, monkeypatch, tmp_path):
        """The original failure: 36 chars against a 97-class model."""
        probs = _emit(EN_DICT, "DL7CP8161", len(EN_DICT) + 2)
        rec = _recognizer(monkeypatch, probs, string.digits + string.ascii_uppercase, tmp_path)
        with pytest.raises(ValueError, match="97 classes"):
            rec.recognize(np.zeros((40, 180, 3), np.uint8))

    def test_warmup_surfaces_the_mismatch_at_startup(self, monkeypatch, tmp_path):
        probs = _emit(EN_DICT, "A", len(EN_DICT) + 2)
        rec = _recognizer(monkeypatch, probs, string.digits + string.ascii_uppercase, tmp_path)
        with pytest.raises(ValueError):
            rec.warmup(1)

    def test_a_plate_alphabet_model_still_works(self, monkeypatch, tmp_path):
        """A head trained on exactly 0-9A-Z (37 classes) is untouched."""
        alphabet = string.digits + string.ascii_uppercase
        probs = _emit(alphabet, "HR29BG7381", len(alphabet) + 1)
        read = _recognizer(monkeypatch, probs, alphabet, tmp_path).recognize(np.zeros((40, 180, 3), np.uint8))
        assert read.text == "HR29BG7381"

    def test_padding_is_mid_grey_after_normalization(self, monkeypatch, tmp_path):
        rec = _recognizer(monkeypatch, _emit(EN_DICT, "A", 97), EN_DICT, tmp_path)
        tensor = rec._preprocess(np.full((48, 96, 3), 255, np.uint8))
        assert tensor.shape == (1, 3, 48, 320)
        assert np.allclose(tensor[..., :96], 1.0)
        assert np.allclose(tensor[..., 96:], 0.0)


def _box(text: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9):
    return ([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text, conf)


class TestEasyOcrReadingOrder:
    def test_split_single_line_with_higher_right_box(self):
        results = [_box("DL5CU", 10, 12, 120, 52), _box("1624", 130, 10, 220, 50)]
        assert "".join(r[1] for r in reading_order(results)) == "DL5CU1624"

    def test_stacked_two_line_plate_reads_top_row_first(self):
        results = [_box("5343", 20, 60, 120, 100), _box("DL3CC", 10, 10, 130, 50)]
        assert "".join(r[1] for r in reading_order(results)) == "DL3CC5343"

    def test_two_rows_each_left_to_right(self):
        results = [
            _box("5343", 70, 62, 130, 100), _box("CC", 80, 8, 130, 48),
            _box("DL3", 10, 10, 70, 50), _box("S", 20, 60, 60, 98),
        ]
        assert "".join(r[1] for r in reading_order(results)) == "DL3CCS5343"
