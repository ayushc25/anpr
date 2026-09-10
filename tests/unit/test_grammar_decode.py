"""Grammar-constrained decoding from the recognizer's own alternatives.

Built around the real misread that motivated it:

    visible plate   DL7CP8161
    OCR output      CLZCP0161

The grammar knows position 2 must be a digit and ``Z`` is not one. The
recognizer knows the answer is ``7``. Neither alone is enough.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.app.ai.plate_recognizer import postprocess
from backend.app.ai.plate_recognizer.ctc import (
    CharPosterior,
    ctc_decode_with_alternatives,
    ctc_greedy_decode,
    summarize,
)
from backend.app.ai.plate_recognizer.grammar_decode import GrammarDecodeConfig, decode

CFG = GrammarDecodeConfig()


def alts(*pairs):
    """One position's alternative list."""
    return tuple(pairs)


class TestCtcKeepsTheDistribution:
    """WP1's premise: argmax used to discard everything but the winner."""

    def _matrix(self, rows, n_classes=37):
        """rows: list of {class_index: prob}. Class 0 is the CTC blank."""
        probs = np.full((len(rows), n_classes), 1e-6, dtype=np.float32)
        for t, row in enumerate(rows):
            for index, prob in row.items():
                probs[t, index] = prob
        return probs

    def test_the_same_string_comes_out_as_greedy_decode(self):
        """The two decoders must never disagree about what was read — only
        about how much evidence survives the call."""
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        rng = np.random.default_rng(3)
        probs = rng.random((24, len(charset) + 1)).astype(np.float32)
        probs /= probs.sum(axis=1, keepdims=True)

        greedy_text, _, greedy_conf = ctc_greedy_decode(probs, charset)
        rich_text, _, rich_conf = summarize(ctc_decode_with_alternatives(probs, charset))
        assert greedy_text == rich_text
        assert greedy_conf == pytest.approx(rich_conf)

    def test_runner_up_characters_survive(self):
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # index 1..10 are digits 0-9; 11.. are letters A..Z
        z_index = 1 + charset.index("Z")
        seven_index = 1 + charset.index("7")
        probs = self._matrix([{z_index: 0.55, seven_index: 0.30}])
        posteriors = ctc_decode_with_alternatives(probs, charset, top_k=4)
        assert posteriors[0].char == "Z"
        assert "7" in [c for c, _ in posteriors[0].alternatives]

    def test_the_blank_is_never_offered_as_an_alternative(self):
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        probs = self._matrix([{0: 0.40, 1 + charset.index("A"): 0.45}])
        posteriors = ctc_decode_with_alternatives(probs, charset)
        assert posteriors[0].char == "A"
        assert all(char for char, _ in posteriors[0].alternatives)

    def test_top_k_zero_disables_alternatives(self):
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        probs = self._matrix([{1 + charset.index("A"): 0.9, 1 + charset.index("B"): 0.05}])
        assert ctc_decode_with_alternatives(probs, charset, top_k=0)[0].alternatives == ()

    def test_best_of_prefers_the_winner_when_already_allowed(self):
        p = CharPosterior("7", 0.8, 0, (("Z", 0.1),))
        assert p.best_of("0123456789") == ("7", 0.8)

    def test_best_of_falls_through_to_an_alternative(self):
        p = CharPosterior("Z", 0.5, 0, (("7", 0.3), ("2", 0.05)))
        assert p.best_of("0123456789") == ("7", 0.3)

    def test_best_of_returns_none_when_nothing_qualifies(self):
        p = CharPosterior("Z", 0.5, 0, (("X", 0.3),))
        assert p.best_of("0123456789") is None


class TestTheRealMisread:
    """DL7CP8161 -> CLZCP0161, position by position."""

    def test_the_type_error_is_fixed_from_model_evidence(self):
        """Position 2 must be a digit. The model said Z and its runner-up is
        7 — which no confusion map contains (ALPHA_TO_DIGIT['Z'] is ('2',))."""
        text = "DLZCP8161"  # only the type error, prefix already right
        alternatives = [alts()] * 9
        alternatives[2] = alts(("7", 0.31), ("2", 0.04))
        result = decode(text, alternatives, [0.55] * 9, CFG)
        assert result.changed
        assert result.text == "DL7CP8161"
        assert "pos2" in result.corrections[0]

    def test_the_map_based_path_would_have_answered_differently(self):
        """Documents WHY the evidence path runs first: repair reaches 2, the
        model reaches 7, and only one of them is the plate."""
        assert postprocess.ALPHA_TO_DIGIT["Z"] == ("2",)
        repaired = postprocess.repair("DLZCP8161")
        assert repaired.repaired
        assert repaired.text == "DL2CP8161"  # confidently wrong

    def test_the_state_code_error_is_fixed_from_model_evidence(self):
        """CL is not a state code; both characters are letters so the mask
        pass cannot see the error at all."""
        text = "CL7CP8161"
        alternatives = [alts()] * 9
        alternatives[0] = alts(("D", 0.28), ("O", 0.03))
        result = decode(text, alternatives, [0.60] * 9, CFG)
        assert result.changed
        assert result.text == "DL7CP8161"
        assert "state code" in result.corrections[0]

    def test_both_errors_together(self):
        text = "CLZCP8161"
        alternatives = [alts()] * 9
        alternatives[0] = alts(("D", 0.28))
        alternatives[2] = alts(("7", 0.31))
        result = decode(text, alternatives, [0.55] * 9, CFG)
        assert result.changed
        assert result.text == "DL7CP8161"

    def test_the_same_type_digit_error_is_not_recoverable_here(self):
        """Position 5, 8 -> 0: both digits, mask satisfied, prefix fine.
        Grammar has nothing to say and must not pretend otherwise. Only
        cross-frame agreement fixes this one."""
        text = "DL7CP0161"
        assert postprocess.is_valid(text)
        alternatives = [alts()] * 9
        alternatives[5] = alts(("8", 0.44))
        result = decode(text, alternatives, [0.55] * 9, CFG)
        assert not result.changed, "a valid plate is never rewritten"


class TestItNeverOverwritesConfidentLegalOcr:
    def test_a_valid_plate_is_left_alone(self):
        alternatives = [alts(("X", 0.9))] * 10
        result = decode("UP32AB1234", alternatives, [0.9] * 10, CFG)
        assert not result.changed
        assert result.text == "UP32AB1234"

    def test_legal_positions_are_untouched_while_fixing_an_illegal_one(self):
        text = "DLZCP8161"
        alternatives = [alts(("Q", 0.40))] * 9  # every position offers a letter
        alternatives[2] = alts(("7", 0.31))
        result = decode(text, alternatives, [0.55] * 9, CFG)
        assert result.text == "DL7CP8161", "only the illegal position moved"

    def test_no_alternatives_means_no_change(self):
        """The EasyOCR case: aggregate confidence only, nothing to decode
        from. Must degrade to exactly the old behaviour."""
        result = decode("CLZCP0161", [], [0.9] * 9, CFG)
        assert not result.changed

    def test_disabled_config_is_a_no_op(self):
        alternatives = [alts()] * 9
        alternatives[2] = alts(("7", 0.9))
        result = decode("DLZCP8161", alternatives, [0.1] * 9, GrammarDecodeConfig(enabled=False))
        assert not result.changed


class TestEvidenceGuards:
    def test_a_tail_probability_cannot_rewrite_a_plate(self):
        alternatives = [alts()] * 9
        alternatives[2] = alts(("7", 0.02))  # below min_alternative_prob
        result = decode("DLZCP8161", alternatives, [0.55] * 9, CFG)
        assert not result.changed
        assert "rejected" in result.declined

    def test_the_ratio_guard_blocks_a_weak_alternative(self):
        alternatives = [alts()] * 9
        alternatives[2] = alts(("7", 0.12))
        # Displaced character was very confident, so 0.12 is far below the
        # 0.25 ratio floor.
        result = decode("DLZCP8161", alternatives, [0.99] * 9, CFG)
        assert not result.changed

    def test_a_decline_is_recorded_for_the_audit_trail(self):
        alternatives = [alts()] * 9
        alternatives[2] = alts(("7", 0.02))
        assert decode("DLZCP8161", alternatives, [0.55] * 9, CFG).declined

    def test_more_than_max_substitutions_is_refused(self):
        """A read needing four evidence-backed type fixes is not a misread of
        a plate, it is a bad crop, and it must stay unresolved.

        Verified needing >=4 under EVERY 9-character mask, so no cheaper
        format can rescue it — the first version of this test used a string
        that had a legitimate 2-fix solution under a different mask, which the
        decoder correctly found.
        """
        text = "DLZCPZZZZ"
        alternatives = [alts(("7", 0.40))] * 9
        result = decode(text, alternatives, [0.55] * 9, GrammarDecodeConfig(max_substitutions=2))
        assert not result.changed

    def test_a_cheaper_mask_may_legitimately_rescue_a_read(self):
        """The flip side: two fixes under mask AANAAANNN is a real solution
        even though AANAANNNN would have needed three. Formats are tried on
        their merits, not in a fixed order."""
        alternatives = [alts()] * 9
        for i in (2, 6):
            alternatives[i] = alts(("7", 0.40))
        result = decode("DLZCPZZ61", alternatives, [0.55] * 9, GrammarDecodeConfig())
        assert result.changed
        assert result.text == "DL7CPZ761"
        assert postprocess.is_valid(result.text)

    def test_an_unfixable_read_stays_unfixed(self):
        """No digit anywhere in the alternatives at the illegal position."""
        alternatives = [alts(("Q", 0.4), ("O", 0.2))] * 9
        result = decode("DLZCP8161", alternatives, [0.55] * 9, CFG)
        assert not result.changed

    def test_an_ambiguous_state_code_is_refused(self):
        """Two plausible real prefixes means guessing which state, which is
        guessing the whole plate."""
        text = "CL32AB1234"
        alternatives = [alts()] * 10
        # D -> DL, and also H -> HL? not a state. Use M (ML) vs D (DL).
        alternatives[0] = alts(("D", 0.30), ("M", 0.29))
        result = decode(text, alternatives, [0.60] * 10, CFG)
        assert not result.changed
        assert "ambiguous" in result.declined

    def test_state_code_constraint_can_be_disabled(self):
        alternatives = [alts()] * 9
        alternatives[0] = alts(("D", 0.28))
        cfg = GrammarDecodeConfig(state_code_constraint=False)
        assert not decode("CL7CP8161", alternatives, [0.60] * 9, cfg).changed


class TestGrammarRepairIsUnchanged:
    """Point 8: the existing map-based path must behave exactly as before."""

    def test_repair_still_only_fixes_mask_mandated_positions(self):
        assert postprocess.repair("UP32A81234").text == "UP32AB1234"

    def test_repair_still_refuses_a_valid_plate(self):
        assert not postprocess.repair("UP32AB1234").repaired

    def test_confusion_maps_are_untouched(self):
        """Point 3 of the brief: no new character pairs on a single sample."""
        assert frozenset(("F", "G")) not in postprocess.SAME_TYPE_CONFUSIONS
        assert "Z" not in postprocess.DIGIT_TO_ALPHA.get("7", ())
        assert postprocess.ALPHA_TO_DIGIT["Z"] == ("2",)
