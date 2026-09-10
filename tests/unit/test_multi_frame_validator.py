"""The validator is the accuracy story, so it gets the most thorough tests.

Reads are constructed directly rather than run through a model: what is under
test is the voting arithmetic, not the recognizer.
"""
import pytest

from backend.app.ai.plate_recognizer import postprocess
from backend.app.ai.quality.plate_quality import QualityBreakdown
from backend.app.ai.types import PlateRead
from backend.app.events.multi_frame_validator import (
    MultiFrameValidator,
    RecognitionState,
    SnapshotRegistry,
    ValidationConfig,
)
from backend.app.events.track_state import TrackState


def quality(score: float) -> QualityBreakdown:
    return QualityBreakdown(
        score=score, width_px=120, sharpness=200.0,
        resolution_score=score, sharpness_score=score,
        aspect_score=1.0, exposure_score=score,
    )


def make_state(reads, camera_id: int = 1, track_id: int = 1) -> TrackState:
    """reads: iterable of (text, rec_conf, det_conf, quality)."""
    state = TrackState(track_id=track_id, camera_id=camera_id)
    for index, (text, rec, det, qual) in enumerate(reads):
        state.add_read(
            PlateRead(text=text, confidence=rec, per_char_confidence=[rec] * len(text), raw_text=text),
            plate_det_confidence=det,
            quality=quality(qual),
            frame_ts=float(index),
        )
    return state


class TestWorkedExample:
    """The case from the architecture doc, frame by frame."""

    READS = [
        ("UP32AB1234", 0.91, 0.88, 0.72),
        ("UP32AB1234", 0.88, 0.90, 0.80),
        ("UP32A81234", 0.62, 0.71, 0.41),
        ("UP32AB1234", 0.93, 0.92, 0.85),
    ]

    def test_picks_the_majority_plate(self):
        result = MultiFrameValidator().validate(make_state(self.READS))
        assert result is not None
        assert result.text == "UP32AB1234"
        assert result.read_count == 4
        # UP32A81234 is repaired to UP32AB1234 at ingest, so by the time the
        # ballot is counted there is only one candidate string.
        assert result.distinct_variants == 1
        assert result.grammar_valid

    def test_support_and_confidence_are_high(self):
        result = MultiFrameValidator().validate(make_state(self.READS))
        assert result.support > 0.85
        assert result.confidence > 0.80

    def test_grammar_demotes_the_odd_read_even_when_it_is_most_confident(self):
        # Give the structurally impossible read the best scores of the four.
        reads = [
            ("UP32AB1234", 0.70, 0.70, 0.60),
            ("UP32AB1234", 0.70, 0.70, 0.60),
            ("UP32A81234", 0.99, 0.99, 0.99),
        ]
        result = MultiFrameValidator(ValidationConfig(min_reads=3)).validate(make_state(reads))
        assert result.text == "UP32AB1234"


class TestQuorum:
    def test_rejects_too_few_reads(self):
        state = make_state([("UP32AB1234", 0.95, 0.95, 0.9)] * 2)
        assert MultiFrameValidator().validate(state) is None

    def test_accepts_at_the_minimum(self):
        state = make_state([("UP32AB1234", 0.95, 0.95, 0.9)] * 3)
        assert MultiFrameValidator().validate(state) is not None

    def test_ignores_reads_below_the_weight_floor(self):
        reads = [("UP32AB1234", 0.95, 0.95, 0.9)] * 3 + [("QQQQQQQQQQ", 0.05, 0.05, 0.02)]
        result = MultiFrameValidator().validate(make_state(reads))
        assert result.text == "UP32AB1234"
        assert result.read_count == 3


class TestPositionalVote:
    def test_resolves_a_split_with_no_string_majority(self):
        # Every read differs, but each POSITION has a clear winner.
        reads = [
            ("UP32AB1234", 0.80, 0.85, 0.70),
            ("UP32AB1284", 0.78, 0.85, 0.70),
            ("UP32AB1734", 0.76, 0.85, 0.70),
            ("UP32AB1234", 0.82, 0.85, 0.70),
        ]
        result = MultiFrameValidator().validate(make_state(reads))
        assert result is not None
        assert result.text == "UP32AB1234"

    def test_differing_lengths_do_not_shift_positions(self):
        reads = [
            ("UP32AB1234", 0.85, 0.85, 0.75),
            ("UP32AB123", 0.60, 0.60, 0.40),    # truncated read
            ("UP32AB1234", 0.85, 0.85, 0.75),
            ("P32AB1234", 0.55, 0.60, 0.40),    # clipped first character
        ]
        result = MultiFrameValidator().validate(make_state(reads))
        assert result.text == "UP32AB1234"

    def test_weakest_position_caps_support(self):
        # Position 5 is a genuine 50/50; support must reflect that.
        reads = [
            ("UP32AB1234", 0.80, 0.80, 0.60),
            ("UP32AC1234", 0.80, 0.80, 0.60),
            ("UP32AB1234", 0.80, 0.80, 0.60),
            ("UP32AC1234", 0.80, 0.80, 0.60),
        ]
        validator = MultiFrameValidator(ValidationConfig(drop_low_confidence=False))
        result = validator.validate(make_state(reads))
        assert result is None or result.support <= 0.75


class TestGrammarRepair:
    def test_repairs_a_unanimous_but_invalid_read(self):
        reads = [("UP32A81234", 0.85, 0.88, 0.75)] * 4
        result = MultiFrameValidator().validate(make_state(reads))
        assert result is not None
        assert result.text == "UP32AB1234"
        assert any("8->B" in c for c in result.corrections)
        assert result.grammar_valid


class TestRegistrySnapping:
    def test_snaps_to_a_registered_plate_one_confusion_away(self):
        registry = SnapshotRegistry({"UP32AB1234": "registered"})
        # Deliberately split so support stays under snap_below.
        reads = [
            ("UP32AD1234", 0.70, 0.70, 0.55),
            ("UP32AD1234", 0.68, 0.70, 0.55),
            ("UP32AB1234", 0.66, 0.70, 0.55),
        ]
        validator = MultiFrameValidator(
            ValidationConfig(strong_support=0.99, min_final_confidence=0.0), registry
        )
        result = validator.validate(make_state(reads))
        assert result is not None
        assert result.text == "UP32AB1234"

    def test_never_snaps_onto_a_blacklisted_plate(self):
        registry = SnapshotRegistry({"UP32AB1234": "blacklist"})
        reads = [("UP32AD1234", 0.70, 0.70, 0.55)] * 3
        validator = MultiFrameValidator(
            ValidationConfig(strong_support=0.99, min_final_confidence=0.0), registry
        )
        result = validator.validate(make_state(reads))
        assert result.text != "UP32AB1234"

    def test_does_not_snap_on_an_arbitrary_neighbour(self):
        # X is not a confusion for B, so this must not snap.
        registry = SnapshotRegistry({"UP32AB1234": "registered"})
        reads = [("UP32AX1234", 0.70, 0.70, 0.55)] * 3
        validator = MultiFrameValidator(
            ValidationConfig(strong_support=0.99, min_final_confidence=0.0), registry
        )
        assert validator.validate(make_state(reads)).text == "UP32AX1234"

    def test_ambiguous_neighbours_are_left_alone(self):
        # B is confusable with 8 (cross-type) and with D (same-type), so this
        # read is one confusion from two registered plates. Guessing between
        # them would be worse than reporting what was actually read.
        registry = SnapshotRegistry({"UP32AB8234": "registered", "UP32ABD234": "registered"})
        assert registry.unique_confusable_neighbour("UP32ABB234") is None


class TestRejection:
    def test_low_confidence_is_dropped_and_flagged(self):
        # Reads strong enough to clear the weight floor, but split three ways
        # and individually weak — the honest "I am not sure" case.
        reads = [
            ("UP32AB1234", 0.35, 0.35, 0.42),
            ("UP32AD1234", 0.35, 0.35, 0.42),
            ("UP32AB1834", 0.35, 0.35, 0.42),
        ]
        state = make_state(reads)
        assert MultiFrameValidator().validate(state) is None
        assert state.disputed

    def test_rejects_a_half_visible_plate(self):
        # An occluded plate reads cleanly and unanimously — every frame sees
        # the same half. High confidence, high support, still not a plate.
        reads = [("DA7486", 0.92, 0.90, 0.85)] * 4
        state = make_state(reads)
        assert MultiFrameValidator().validate(state) is None
        assert state.disputed

    def test_a_kept_fragment_is_recorded_but_never_believed(self):
        """A fragment must never be presented as a plate. It may, when the
        site has asked for a complete log, be RECORDED as a thing we saw and
        did not resolve.

        This test previously asserted the fragment was dropped even with
        drop_low_confidence=False, on the reasoning that the knob governs
        uncertain reads and a fragment is incomplete rather than uncertain.
        That reasoning was sound when a result was a bare confidence number:
        emitting the row at all was indistinguishable from believing it. With
        explicit recognition states it no longer is, and the two cases an
        operator most needs to tell apart — "no vehicle" and "a vehicle whose
        plate we could only half see" — are otherwise both a gap in the log.

        So the contract is now the stronger one: dropped by default, and when
        kept, UNRESOLVED with the ANPR confidence capped well below anything
        that reads as trustworthy, however sure the recognizer was.
        """
        reads = [("DA7486", 0.92, 0.90, 0.85)] * 4
        cfg = ValidationConfig(drop_low_confidence=False)

        assert MultiFrameValidator().validate(make_state(reads)) is None, "dropped by default"

        state = make_state(reads)
        result = MultiFrameValidator(cfg).validate(state)
        assert result is not None
        assert result.state is RecognitionState.UNRESOLVED
        assert result.cap_reason == "incomplete"
        assert result.confidence <= cfg.cap_incomplete
        assert state.disputed
        # The recognizer's own confidence is reported unchanged: it is true,
        # and it is what separates "could not read it" from "read a fragment
        # of it perfectly well".
        assert result.ocr_confidence > 0.9
        assert result.confidence < result.ocr_confidence

    def test_the_whole_plate_wins_however_long_the_occlusion_lasted(self):
        # The vehicle clears the obstruction late in the track: six frames of
        # confident fragment, three of the real plate seen less well. On
        # accumulated weight the fragments win — which is exactly why they are
        # struck from the ballot rather than merely weighted down.
        reads = [("DA7486", 0.95, 0.92, 0.90)] * 6 + [("UP32DA7486", 0.70, 0.80, 0.60)] * 3
        result = MultiFrameValidator().validate(make_state(reads))
        assert result is not None
        assert result.text == "UP32DA7486"

    def test_a_track_that_only_ever_saw_half_is_flagged_not_emitted(self):
        # Nothing usable is left after the fragments are struck, which is the
        # same situation as a vehicle the system could not read at all: the
        # operator should see it, the events log should not.
        state = make_state([("DA7486", 0.95, 0.92, 0.90)] * 6)
        assert MultiFrameValidator().validate(state) is None
        assert state.disputed

    def test_can_be_configured_to_keep_low_confidence_results(self):
        # Reads strong enough to clear the weight floor, but split three ways
        # and individually weak — the honest "I am not sure" case.
        reads = [
            ("UP32AB1234", 0.35, 0.35, 0.42),
            ("UP32AD1234", 0.35, 0.35, 0.42),
            ("UP32AB1834", 0.35, 0.35, 0.42),
        ]
        state = make_state(reads)
        validator = MultiFrameValidator(ValidationConfig(drop_low_confidence=False))
        assert validator.validate(state) is not None
        assert state.disputed


class TestTrackLocking:
    def test_locks_after_consistent_reads(self):
        state = make_state([("UP32AB1234", 0.92, 0.90, 0.85)] * 4)
        state.consider_lock()
        assert state.locked
        assert not state.should_attempt_plate(min_interval_frames=0)

    def test_does_not_lock_on_a_disputed_plate(self):
        state = make_state([
            ("UP32AB1234", 0.80, 0.80, 0.70),
            ("UP32AD1234", 0.80, 0.80, 0.70),
            ("UP32AB1834", 0.80, 0.80, 0.70),
            ("UP32AB1234", 0.80, 0.80, 0.70),
        ])
        state.consider_lock()
        assert not state.locked

    def test_does_not_lock_on_an_invalid_plate(self):
        state = make_state([("XX99ZZ9999", 0.95, 0.95, 0.9)] * 5)
        state.consider_lock()
        assert not state.locked


class TestReadWeight:
    def test_weight_rewards_every_component(self):
        strong = make_state([("UP32AB1234", 0.95, 0.95, 0.95)]).reads[0]
        weak = make_state([("UP32AB1234", 0.50, 0.50, 0.30)]).reads[0]
        assert strong.weight > weak.weight * 3

    def test_a_repairable_read_is_promoted_to_full_grammar_weight(self):
        valid = make_state([("UP32AB1234", 0.90, 0.90, 0.80)]).reads[0]
        repaired = make_state([("UP32A81234", 0.90, 0.90, 0.80)]).reads[0]
        assert repaired.text == "UP32AB1234"
        assert repaired.corrections == ["pos5: 8->B"]
        assert repaired.weight == pytest.approx(valid.weight)

    def test_unrepairable_read_keeps_a_reduced_weight(self):
        valid = make_state([("UP32AB1234", 0.90, 0.90, 0.80)]).reads[0]
        junk = make_state([("QQQQQQQQQQ", 0.90, 0.90, 0.80)]).reads[0]
        assert junk.weight < valid.weight
        assert junk.grammar == pytest.approx(postprocess.grammar_factor("QQQQQQQQQQ"))
