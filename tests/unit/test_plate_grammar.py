import pytest

from backend.app.ai.plate_recognizer import postprocess as pp


class TestNormalize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("up32ab1234", "UP32AB1234"),
            ("UP 32 AB 1234", "UP32AB1234"),
            ("UP-32-AB-1234", "UP32AB1234"),
            ("INDUP32AB1234", "UP32AB1234"),   # hologram strip picked up by OCR
            ("  DL8CAF5010 ", "DL8CAF5010"),
            ("", ""),
        ],
    )
    def test_normalize(self, raw, expected):
        assert pp.normalize(raw) == expected

    def test_normalize_does_not_substitute_characters(self):
        # Substitution is repair's job, and only under grammar constraint.
        assert pp.normalize("UP32A81234") == "UP32A81234"


class TestValidity:
    @pytest.mark.parametrize(
        "plate",
        ["UP32AB1234", "DL8CAF5010", "MH12DE1433", "KA05MG2234", "21BH2345AA"],
    )
    def test_valid_plates(self, plate):
        assert pp.is_valid(plate), plate

    @pytest.mark.parametrize(
        "plate",
        [
            "UP32A81234",   # digit where the format demands alpha
            "XX32AB1234",   # not a real state code
            "UP32AB",       # too short
            "1234567890",
            "",
        ],
    )
    def test_invalid_plates(self, plate):
        assert not pp.is_valid(plate), plate

    def test_state_code_is_checked(self):
        assert pp.match_format("VP32AB1234") is not None   # right shape
        assert not pp.is_valid("VP32AB1234")               # wrong state


class TestGrammarFactor:
    def test_valid_beats_invalid(self):
        assert pp.grammar_factor("UP32AB1234") == 1.0
        assert pp.grammar_factor("UP32A81234") < 1.0

    def test_unknown_state_is_penalised_but_not_rejected(self):
        factor = pp.grammar_factor("VP32AB1234")
        assert 0.5 < factor < 1.0

    def test_garbage_scores_low(self):
        assert pp.grammar_factor("XZ9") <= 0.35


class TestRepair:
    def test_repairs_digit_for_alpha(self):
        result = pp.repair("UP32A81234")
        assert result.repaired
        assert result.text == "UP32AB1234"
        assert result.corrections == ["pos5: 8->B"]

    def test_repairs_alpha_for_digit(self):
        result = pp.repair("KAO5MG2234")   # O read where a digit belongs
        assert result.repaired
        assert result.text == "KA05MG2234"

    def test_leaves_alone_a_string_that_parses_as_another_real_format(self):
        # UP3ZAB1234 is not a corrupted UP32AB1234: it parses cleanly as
        # UP-3-ZAB-1234, the same single-digit-district shape as DL8CAF5010.
        # Rewriting it would invent a plate that was never on the road.
        assert pp.is_valid("UP3ZAB1234")
        assert not pp.repair("UP3ZAB1234").repaired

    def test_leaves_valid_plates_alone(self):
        result = pp.repair("UP32AB1234")
        assert not result.repaired
        assert result.text == "UP32AB1234"
        assert result.corrections == []

    def test_gives_up_on_unrepairable(self):
        result = pp.repair("QQQQQQQQQQ")
        assert not result.repaired

    def test_respects_max_fixes(self):
        # KAOSMG2Z34 needs at least two substitutions under every format, so a
        # cap of one must decline rather than half-repair it.
        assert not pp.repair("KAOSMG2Z34", max_fixes=1).repaired
        assert pp.repair("KAOSMG2Z34", max_fixes=2).repaired


class TestFragment:
    @pytest.mark.parametrize(
        "raw",
        [
            "DA7486",     # half a plate, the other half behind a parked van
            "R00467",
            "GR3970",
            "1234",
            "",
        ],
    )
    def test_partial_reads_are_fragments(self, raw):
        assert pp.is_fragment(raw), raw

    @pytest.mark.parametrize(
        "raw",
        ["UP32AB1234", "DL8CAF5010", "UP1CY3590", "21BH2345AA", "TP2561992"],
    )
    def test_whole_plates_are_not_fragments(self, raw):
        assert not pp.is_fragment(raw), raw

    def test_a_short_but_valid_plate_survives(self):
        # The length rule must not veto a plate that parses outright, state
        # code and all — the older short series are real.
        assert pp.is_valid("DL81234")
        assert not pp.is_fragment("DL81234")

    def test_fragment_votes_below_everything_else(self):
        # A fragment is read cleanly and repeatedly, so it must not be able to
        # out-vote a genuine plate on agreement alone.
        assert pp.grammar_factor("DA7486") < pp.grammar_factor("XX32AB1234")
        assert pp.grammar_factor("DA7486") < pp.grammar_factor("UP32AB1234")


class TestTrimNoise:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("TUP1GEJ0364", "UP1GEJ0364"),   # glyph off the hologram strip
            ("1UP32AB1234", "UP32AB1234"),   # the reported "extra leading digit"
            ("7DL8CAF5010", "DL8CAF5010"),
            ("UP32AB1234X", "UP32AB1234"),   # bolt head on the right
        ],
    )
    def test_drops_affix_noise(self, raw, expected):
        result = pp.trim_noise(raw)
        assert result.repaired
        assert result.text == expected
        assert result.corrections

    def test_trims_then_repairs(self):
        # Leading noise AND a confusion inside: 'T' is not part of the plate,
        # and the 0 sits where the series demands a letter.
        result = pp.trim_noise("TUP140X0554")
        assert result.text == "UP14OX0554"

    def test_never_trims_a_valid_plate(self):
        assert not pp.trim_noise("UP32AB1234").repaired
        # ...even though UP32AB1234 minus its leading U is not valid anyway,
        # the guard is what stops a real plate being shortened into another.
        assert not pp.trim_noise("MH12DE1433").repaired

    def test_declines_when_the_remainder_is_not_a_plate(self):
        # 160406937 is a misread, not a plate with a prefix; no trim of it is
        # valid, so it must survive unchanged rather than be guessed at.
        assert not pp.trim_noise("160406937").repaired
        assert not pp.trim_noise("TP144945").repaired

    def test_only_ever_returns_a_fully_valid_plate(self):
        # The guarantee the caller relies on: a trim either lands on a real
        # plate — shape and state code — or does not happen at all. Anything
        # weaker and this becomes a machine for inventing registrations.
        noise = ["1UP32AB1234", "TUP1GEJ0364", "160406937", "QQQQQQQQQQ",
                 "TP144945", "XYZ", "UP32AB1234"]
        for raw in noise:
            result = pp.trim_noise(raw)
            if result.repaired:
                assert pp.is_valid(result.text), raw
            else:
                assert result.text == raw

    def test_respects_the_trim_budget(self):
        assert not pp.trim_noise("IND1UP32AB1234").repaired


class TestResolve:
    def test_prefers_substitution_over_deletion(self):
        # UP32A81234 is repairable in place; resolve must not reach for a trim.
        result = pp.resolve("UP32A81234")
        assert result.text == "UP32AB1234"
        assert result.corrections == ["pos5: 8->B"]

    def test_falls_through_to_trimming(self):
        result = pp.resolve("TUP1GEJ0364")
        assert result.text == "UP1GEJ0364"

    def test_valid_plate_is_untouched(self):
        result = pp.resolve("MH12DE1433")
        assert result.text == "MH12DE1433"
        assert not result.repaired

    def test_hopeless_read_is_returned_verbatim(self):
        assert pp.resolve("19450").text == "19450"


class TestConfusable:
    def test_single_confusable_difference(self):
        assert pp.confusable("UP32AB1234", "UP32A81234")
        assert pp.confusable("UP32AB1234", "UP32AB1Z34"[:10])

    def test_rejects_non_confusion_difference(self):
        # X/B is not an OCR confusion; snapping on it would invent a plate.
        assert not pp.confusable("UP32AB1234", "UP32AX1234")

    def test_rejects_multiple_differences(self):
        assert not pp.confusable("UP32AB1234", "UP32A81235")

    def test_rejects_different_lengths(self):
        assert not pp.confusable("UP32AB1234", "UP32AB123")


def test_format_display():
    assert pp.format_display("UP32AB1234") == "UP 32 AB 1234"
    assert pp.format_display("21BH2345AA") == "21 BH 2345 AA"
    assert pp.format_display("NOTAPLATE!") == "NOTAPLATE!"
