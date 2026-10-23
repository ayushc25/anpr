"""How the Qwen-VL recognizer turns generated tokens into a PlateRead.

The model is not loaded: ``_generate`` is replaced with scripted tokens, so
what is under test is the part that is OURS — what we report as confidence and
what we refuse to invent.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.app.ai.plate_recognizer.qwen_vl_onnx import (
    DEFAULT_PROMPT,
    DEFAULT_VERIFY_PROMPT,
    QwenVlOnnxRecognizer,
    _ascii_piece,
    _Generation,
    _Token,
)


def recognizer(tokens):
    """A recognizer whose model is a script."""
    rec = object.__new__(QwenVlOnnxRecognizer)
    rec.prompt = DEFAULT_PROMPT
    rec.verify_prompt = DEFAULT_VERIFY_PROMPT
    rec.max_new_tokens = 24
    rec._generate = lambda image, prompt, max_tokens, want_alternatives: _Generation(tokens=list(tokens))
    return rec


CROP = np.zeros((40, 120, 3), dtype=np.uint8)


def digits(text, prob=0.99):
    return [_Token(c, prob) for c in text]


def test_digits_and_single_letters_map_one_to_one():
    read = recognizer(digits("UP32AB1234")).recognize(CROP)
    assert read.text == "UP32AB1234"
    assert len(read.per_char_confidence) == 10
    assert read.confidence == pytest.approx(0.99)


def test_merged_tokens_share_one_probability_and_offer_no_alternatives():
    """'HR' and 'BG' are single tokens in Qwen's vocabulary. The model scored
    them as units, so there is no per-letter evidence to report, and none is
    invented."""
    tokens = [
        _Token("HR", 0.80, alternatives=()),
        _Token("2", 0.99, alternatives=(("Z", 0.004),)),
        _Token("9", 0.99),
        _Token("BG", 0.60),
        *digits("7381"),
    ]
    read = recognizer(tokens).recognize(CROP)
    assert read.text == "HR29BG7381"
    assert read.per_char_confidence[:2] == [0.80, 0.80]       # H, R: one token, one number
    assert read.per_char_confidence[4:6] == [0.60, 0.60]      # B, G
    assert read.per_char_alternatives[0] == () and read.per_char_alternatives[1] == ()
    assert read.per_char_alternatives[2] == (("Z", 0.004),)   # a real single-character runner-up
    assert len(read.per_char_alternatives) == len(read.text)


def test_confidence_is_the_geometric_mean_of_token_probabilities():
    tokens = [_Token(c, p) for c, p in zip("UP14FS3664", [1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1])]
    read = recognizer(tokens).recognize(CROP)
    assert read.confidence == pytest.approx(math.exp(math.log(0.5) / 10))
    # One doubtful token drags the whole plate down; it is not averaged away.
    assert read.confidence < 0.99


def test_unreadable_is_none_not_a_guess():
    assert recognizer([_Token("UNREADABLE", 0.97)]).recognize(CROP) is None
    assert recognizer([_Token("unreadable", 0.9)]).recognize(CROP) is None


def test_empty_generation_is_none():
    assert recognizer([]).recognize(CROP) is None
    assert recognizer([_Token(" ", 0.9)]).recognize(CROP) is None


def test_too_short_to_be_a_plate_is_none():
    assert recognizer(digits("UP1")).recognize(CROP) is None


def test_separators_are_dropped_and_arrays_stay_aligned():
    tokens = [*digits("UP32"), _Token("-", 0.5), *digits("AB1234")]
    read = recognizer(tokens).recognize(CROP)
    assert read.text == "UP32AB1234"
    assert len(read.per_char_confidence) == len(read.text) == len(read.per_char_alternatives)
    assert 0.5 not in read.per_char_confidence                # the dash's number left with the dash


def test_fullwidth_letters_fold_to_ascii():
    """Seen on a real crop: U+FF23 for 'C'. isalnum() accepts it, so without
    folding it would reach the plate string and never match a registry row."""
    assert _ascii_piece("Ｃ") == "C"
    assert _ascii_piece("１２") == "12"

    # ...and it is applied where tokens are actually decoded.
    class FakeTokenizer:
        def decode(self, ids):
            return "Ｃ"

    rec = object.__new__(QwenVlOnnxRecognizer)
    rec._tokenizer, rec._piece_cache = FakeTokenizer(), {}
    assert rec._piece(42) == "C"
    assert rec._piece_cache[42] == "C"


def test_non_ascii_noise_becomes_a_dropped_separator():
    assert _ascii_piece("�") == "-"
    assert _ascii_piece("१") == "-"                       # a Devanagari digit is not a plate character
    assert len(_ascii_piece("AB�")) == 3                 # one out per in: alignment survives


def test_verification_is_capability_gated_on_the_base_class():
    from backend.app.ai.plate_recognizer.base import PlateRecognizer

    class Ctc(PlateRecognizer):
        def recognize(self, plate_image):
            return None

    assert Ctc().supports_verification is False
    assert Ctc().verify(CROP, "UP32AB1234") is None           # "no answer", never 0.5


def test_prompts_are_the_agreed_ones():
    assert "Return ONLY the registration number" in DEFAULT_PROMPT
    assert "UNREADABLE" in DEFAULT_PROMPT
    assert '"{candidate}"' in DEFAULT_VERIFY_PROMPT
    assert "Answer only Yes or No" in DEFAULT_VERIFY_PROMPT
