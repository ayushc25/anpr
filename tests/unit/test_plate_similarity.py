"""Same-vehicle matching.

This is what stops one car producing a row per read. It has to be loose enough
to join the noisy spellings a weak recognizer produces for one vehicle, and
tight enough never to merge two different vehicles.
"""
import pytest

from backend.app.ai.plate_recognizer.postprocess import (
    best_registry_match, edit_distance, registry_cost, same_vehicle, similarity,
)

#: A small society registry, of the size and shape a real one has.
REGISTRY = [
    "UP32AB1234",
    "UP14OX0554",
    "UP1GEJ0364",
    "MH12DE1433",
    "DL8CAF5010",
    "KA05MG2234",
]


class TestEditDistance:
    @pytest.mark.parametrize("a,b,expected", [
        ("UP32AB1234", "UP32AB1234", 0),
        ("UP32AB1234", "UP32A81234", 1),
        ("ABC", "", 3),
        ("", "", 0),
        ("UP32AB1234", "UP32AB123", 1),      # truncation
        ("P32AB1234", "UP32AB1234", 1),      # clipped leading char
    ])
    def test_distance(self, a, b, expected):
        assert edit_distance(a, b) == expected

    def test_symmetric(self):
        assert edit_distance("TUP14F49450", "TP1449450") == edit_distance("TP1449450", "TUP14F49450")


class TestSimilarity:
    def test_identical_is_one(self):
        assert similarity("UP32AB1234", "UP32AB1234") == 1.0

    def test_unrelated_is_low(self):
        assert similarity("UP32AB1234", "DL8CAF5010") < 0.25

    def test_bounds(self):
        assert 0.0 <= similarity("ABC", "XYZ") <= 1.0
        assert similarity("", "") == 1.0


class TestSameVehicle:
    @pytest.mark.parametrize("a,b", [
        # Real read pairs observed from one car in the test footage.
        ("TUP14F49450", "TP1449450"),
        ("TUP14HF6592", "UP14HF6592"),
        ("LZCP01607", "DLZCP0161"),
        ("UP23AT0097", "UP234700971"),
        ("UP32AB1234", "UP32A81234"),
    ])
    def test_noisy_reads_of_one_vehicle_match(self, a, b):
        assert same_vehicle(a, b), f"{a} / {b} should be treated as one vehicle"

    @pytest.mark.parametrize("a,b", [
        ("UP32AB1234", "DL8CAF5010"),
        ("MH12DE1433", "KA05MG2234"),
        ("UP32AB1234", "UP99ZZ9999"),
    ])
    def test_different_vehicles_do_not_match(self, a, b):
        assert not same_vehicle(a, b)

    def test_symmetric(self):
        assert same_vehicle("TUP14F49450", "TP1449450") == same_vehicle("TP1449450", "TUP14F49450")

    def test_empty_never_matches(self):
        assert not same_vehicle("", "UP32AB1234")
        assert not same_vehicle("UP32AB1234", "")

    def test_shared_prefix_is_enough(self):
        """Misreads corrupt the tail far more often than the state+district,
        so a solid five-character prefix is strong evidence on its own."""
        assert same_vehicle("UP32AB1234", "UP32AB9999")

    def test_threshold_is_adjustable(self):
        assert not same_vehicle("UP32AB1234", "UP99XY8888", threshold=0.95)


class TestRegistryCost:
    def test_a_confusion_costs_less_than_an_unrelated_substitution(self):
        # Plain edit distance calls both of these 1. The whole point of the
        # weighting is that one is a misread and the other is another car.
        assert edit_distance("UP32A81234", "UP32AB1234") == edit_distance("UP32AX1234", "UP32AB1234")
        assert registry_cost("UP32A81234", "UP32AB1234") < registry_cost("UP32AX1234", "UP32AB1234")

    def test_identical_costs_nothing(self):
        assert registry_cost("UP32AB1234", "UP32AB1234") == 0.0

    def test_a_dropped_character_is_costed(self):
        assert registry_cost("UP32AB123", "UP32AB1234") > 0


class TestBestRegistryMatch:
    @pytest.mark.parametrize("read,expected", [
        ("UP32AB1234", "UP32AB1234"),   # exact
        ("UP32A81234", "UP32AB1234"),   # one confusion
        ("UP140X0554", "UP14OX0554"),   # 0 read where the series has an O
        ("TUP1GEJ0364", "UP1GEJ0364"),  # a glyph off the hologram strip
        ("MH12DE1A33", "MH12DE1433"),
    ])
    def test_recovers_a_noisy_read(self, read, expected):
        match = best_registry_match(read, REGISTRY)
        assert match is not None, read
        assert match.plate == expected

    def test_declines_a_plate_that_is_not_registered(self):
        # A visitor. The registry is a closed set, and a read that is simply
        # not in it must come back unknown rather than snapping to whichever
        # resident happens to be nearest.
        assert best_registry_match("WB06XY9999", REGISTRY) is None

    def test_declines_when_two_registered_plates_fit_equally(self):
        # The society has two plates differing in one character. A read that
        # sits between them identifies neither, and naming one would put the
        # wrong resident on the gate log.
        registry = ["UP32AB1234", "UP32AB1284"]
        assert best_registry_match("UP32AB12B4", registry) is None

    def test_an_exact_read_still_matches_among_near_neighbours(self):
        # The ambiguity guard must not veto a read that is exactly right.
        registry = ["UP32AB1234", "UP32AB1284"]
        match = best_registry_match("UP32AB1284", registry)
        assert match is not None and match.plate == "UP32AB1284"

    def test_declines_a_fragment(self):
        # Half a plate is close to everything and identifies nothing.
        assert best_registry_match("DA7486", REGISTRY) is None

    def test_declines_against_an_empty_registry(self):
        assert best_registry_match("UP32AB1234", []) is None
