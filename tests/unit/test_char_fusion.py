"""Per-character temporal fusion.

The tests that matter here are about what the OLD positional vote got wrong:
it normalized by the mass it had assigned, so unanimous guessing scored the
same as unanimous certainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.events import char_fusion
from backend.app.events.char_fusion import fuse


@dataclass
class Read:
    """Minimal stand-in for a WeightedRead — fusion takes a Protocol."""

    text: str
    weight: float = 1.0
    per_char: list[float] = field(default_factory=list)

    def char_confidence(self, i: int) -> float:
        if i < len(self.per_char):
            return self.per_char[i]
        return 1.0


def reads(text: str, p: float, n: int = 3, weight: float = 1.0) -> list[Read]:
    return [Read(text, weight, [p] * len(text)) for _ in range(n)]


class TestWithheldProbabilityMass:
    """The bug this module exists to fix."""

    def test_unanimous_but_unconfident_does_not_score_as_certain(self):
        """Three reads agreeing on every character at 10% confidence used to
        report support 1.0, because the only character with any mass was the
        only character considered."""
        result = fuse(reads("UP32AB1234", 0.10))
        assert result is not None
        for position in result.positions:
            assert position.posterior < 0.15, position.describe()
        assert result.char_support() < 0.15

    def test_unanimous_and_confident_still_scores_high(self):
        result = fuse(reads("UP32AB1234", 0.97))
        assert result is not None
        assert result.char_support() > 0.9

    def test_posterior_tracks_the_recognizer_confidence(self):
        """Monotonic in per-character confidence — the property the old
        normalization destroyed entirely."""
        scores = [fuse(reads("UP32AB1234", p)).char_support() for p in (0.2, 0.5, 0.8, 0.95)]
        assert scores == sorted(scores)

    def test_more_agreeing_reads_do_not_manufacture_certainty(self):
        """Ten coin flips that agree are still coin flips. Weight of evidence
        raises agreement, not per-character confidence."""
        few = fuse(reads("UP32AB1234", 0.30, n=2)).char_support()
        many = fuse(reads("UP32AB1234", 0.30, n=10)).char_support()
        assert abs(few - many) < 0.05


class TestCompetingCharacters:
    def test_a_contested_position_is_found(self):
        pool = [
            Read("UP32AB1234", 1.0, [0.9] * 10),
            Read("UP32AB1234", 1.0, [0.9] * 10),
            Read("UP32AX1234", 1.0, [0.9] * 10),
        ]
        result = fuse(pool)
        assert result.text == "UP32AB1234"
        contested = result.position(5)
        assert contested.char == "B"
        assert contested.runner_up == "X"
        assert contested.variants == 2

    def test_the_majority_character_wins_each_position(self):
        pool = [
            Read("UP32AB1234", 1.0, [0.8] * 10),
            Read("MP32AB1234", 1.0, [0.8] * 10),
            Read("UP32AB1234", 1.0, [0.8] * 10),
        ]
        assert fuse(pool).text == "UP32AB1234"

    def test_a_split_with_no_clear_winner_is_unstable(self):
        """0.45 vs 0.44 passes no honest test, which is why margin exists
        alongside posterior."""
        pool = [
            Read("UP32AB1234", 1.0, [0.95] * 10),
            Read("UP32AX1234", 1.0, [0.95] * 10),
        ]
        result = fuse(pool)
        assert 5 in [p.index for p in result.unstable()]


class TestOffLengthReads:
    def test_only_modal_length_reads_participate(self):
        pool = reads("UP32AB1234", 0.9, n=3) + reads("UP32AB123", 0.9, n=1)
        result = fuse(pool)
        assert result.modal_length == 10
        assert result.pool_size == 3

    def test_off_length_weight_is_reported_not_discarded(self):
        """Disagreeing about the plate's LENGTH is a different and worse
        problem than disagreeing about a character, so a caller gets told."""
        pool = reads("UP32AB1234", 0.9, n=3) + reads("UP32AB123", 0.9, n=1)
        assert 0.0 < fuse(pool).off_length_weight < 0.5

    def test_a_single_read_is_not_fusable(self):
        """One read's own confidence must not be relabelled as agreement."""
        assert fuse(reads("UP32AB1234", 0.9, n=1)) is None

    def test_no_reads_at_all(self):
        assert fuse([]) is None


class TestGrammarCorroboration:
    def test_the_format_settles_a_position_it_excludes(self):
        """Position 5 must be a letter. The model says B, its runner-up is 8.
        The format has already eliminated the runner-up."""
        pool = [
            Read("UP32AB1234", 1.0, [0.45] * 10),
            Read("UP32A81234", 1.0, [0.45] * 10),
            Read("UP32AB1234", 1.0, [0.45] * 10),
        ]
        result = fuse(pool)
        position = result.position(5)
        assert position.char == "B"
        assert position.runner_up == "8"
        assert position.grammar_locked
        assert position.adjusted_posterior >= char_fusion.DEFAULT_GRAMMAR_FLOOR

    def test_no_credit_when_the_runner_up_also_fits_the_format(self):
        """B vs X are both letters, so the grammar has nothing to say and must
        not pretend otherwise."""
        pool = [
            Read("UP32AB1234", 1.0, [0.45] * 10),
            Read("UP32AX1234", 1.0, [0.45] * 10),
            Read("UP32AB1234", 1.0, [0.45] * 10),
        ]
        assert not fuse(pool).position(5).grammar_locked

    def test_no_credit_without_a_competing_character(self):
        """A position where every read agrees but at low confidence gets
        nothing: "must be a digit" narrows 36 options to 10, which is real
        information but not enough to call a coin flip settled."""
        result = fuse(reads("UP32AB1234", 0.30))
        assert not any(p.grammar_locked for p in result.positions)

    def test_no_credit_when_the_string_matches_no_format(self):
        pool = [Read("22473248", 1.0, [0.45] * 8), Read("22473248", 1.0, [0.45] * 8)]
        assert not any(p.grammar_locked for p in fuse(pool).positions)


class TestCharSupportAggregate:
    def _mixed(self):
        """Nine strong positions and one weak one, with no grammar rescue."""
        pool = [
            Read("UP32AB1234", 1.0, [0.99] * 3 + [0.50] + [0.99] * 6),
            Read("UP32AB1234", 1.0, [0.99] * 3 + [0.50] + [0.99] * 6),
        ]
        return fuse(pool)

    def test_the_weakest_position_dominates_but_does_not_dictate(self):
        result = self._mixed()
        weakest = result.weakest_posterior
        support = result.char_support(0.70)
        assert weakest < support < result.char_support(0.0)

    def test_min_weight_one_restores_the_old_pure_minimum(self):
        result = self._mixed()
        assert result.char_support(1.0) == result.weakest_posterior

    def test_ten_weak_positions_still_fail(self):
        """Softening the aggregate must not rescue a plate that is weak
        everywhere — that is not a reading at all."""
        result = fuse(reads("UP32AB1234", 0.45))
        assert result.char_support(0.70) < 0.55

    def test_one_weak_position_is_not_treated_like_ten(self):
        one_weak = self._mixed().char_support(0.70)
        all_weak = fuse(reads("UP32AB1234", 0.50)).char_support(0.70)
        assert one_weak > all_weak


class TestAuditTrail:
    def test_a_position_describes_itself_for_the_event_record(self):
        pool = [
            Read("UP32AB1234", 1.0, [0.9] * 10),
            Read("UP32AX1234", 1.0, [0.9] * 10),
        ]
        described = fuse(pool).position(5).describe()
        assert described.startswith("pos5=")
        assert " vs " in described
