"""OCR confidence vs ANPR confidence, and the three recognition states.

The governing rule, and the reason this file exists:

    A recognizer that is 94% sure of six glyphs has said NOTHING about
    whether those six glyphs are a vehicle's registration.

Every test below is a variation on keeping those two claims apart.
"""
from __future__ import annotations

import pytest

from backend.app.ai.plate_recognizer import postprocess
from backend.app.events.multi_frame_validator import (
    MultiFrameValidator,
    RecognitionState,
    SnapshotRegistry,
    ValidationConfig,
)
from backend.app.events.track_state import TrackState, WeightedRead

#: Keep results instead of dropping them, so a test can inspect the verdict
#: rather than only observing that something vanished.
KEEP = ValidationConfig(drop_low_confidence=False)


def state_with(texts, ocr=0.94, quality=0.90, det=0.92, per_char=None):
    state = TrackState(track_id=1, camera_id=1)
    for text in texts:
        state.reads.append(
            WeightedRead(
                text=text,
                raw_text=text,
                rec_confidence=ocr,
                per_char_confidence=list(per_char) if per_char else [ocr] * len(text),
                plate_det_confidence=det,
                quality=quality,
                grammar=postprocess.grammar_factor(text),
                frame_ts=0.0,
            )
        )
    return state


def verdict(texts, cfg=KEEP, **kwargs):
    return MultiFrameValidator(cfg).validate(state_with(texts, **kwargs))


class TestTheHeadlineRequirement:
    """OCR 94% on an incomplete read must not become ANPR 94%."""

    def test_a_confident_fragment_is_not_a_confident_plate(self):
        result = verdict(["UP1606"] * 5, ocr=0.94)
        assert result is not None
        assert result.ocr_confidence == pytest.approx(0.94, abs=0.01)
        assert result.confidence <= KEEP.cap_incomplete
        assert result.state is RecognitionState.UNRESOLVED
        assert result.cap_reason == "incomplete"

    def test_the_recognizer_confidence_is_reported_unchanged(self):
        """Not suppressed — it is true, and it is what distinguishes 'could
        not read it' from 'read a fragment of it perfectly well'."""
        unreadable = verdict(["UP32AB1234"] * 5, ocr=0.20)
        fragment = verdict(["UP1606"] * 5, ocr=0.94)
        assert unreadable.ocr_confidence < 0.3
        assert fragment.ocr_confidence > 0.9
        # Both are unresolved, for completely different reasons: one because
        # the model could not make out the characters, the other because the
        # characters it made out perfectly well are not a whole plate.
        assert unreadable.state is RecognitionState.UNRESOLVED
        assert unreadable.cap_reason == ""
        assert fragment.state is RecognitionState.UNRESOLVED
        assert fragment.cap_reason == "incomplete"

    def test_unanimous_guessing_is_not_evidence(self):
        """Five reads agreeing on characters each 20% sure produce a string
        support of 1.0, because agreement is a ratio and therefore scale-free.
        Agreement plus a good crop must not carry a plate over the PROBABLE
        floor when not one character is settled."""
        result = verdict(["UP32AB1234"] * 5, ocr=0.20, quality=0.90)
        assert result.support == pytest.approx(1.0), "the frames did all agree"
        assert len(result.unstable_positions) == 10, "and not one character is settled"
        assert result.state is RecognitionState.UNRESOLVED

    def test_by_default_a_fragment_produces_no_event_at_all(self):
        assert verdict(["UP1606"] * 5, cfg=ValidationConfig()) is None

    @pytest.mark.parametrize(
        "text,cap",
        [
            ("UP1606", "incomplete"),          # too short to be a registration
            ("22473248", "no_known_format"),   # all digits, no plate shape
            ("7WP25614992", "no_known_format"),  # 11 chars, hologram glyph glued on
        ],
    )
    def test_the_reported_malformed_outputs_are_never_confident(self, text, cap):
        result = verdict([text] * 5, ocr=0.94)
        assert result.cap_reason == cap
        assert result.state is RecognitionState.UNRESOLVED
        assert result.confidence < result.ocr_confidence


class TestGrammarHole:
    """The root cause of UP1606 scoring 0.95, fixed in postprocess."""

    def test_a_six_character_string_is_no_longer_a_valid_plate(self):
        assert not postprocess.is_valid("UP1606")
        assert postprocess.is_fragment("UP1606")

    def test_the_seven_character_no_series_form_still_is(self):
        """DL81234 is a genuine old Delhi registration. Rejecting it would
        trade one wrong answer for another."""
        assert postprocess.is_valid("DL81234")
        assert not postprocess.is_fragment("DL81234")

    def test_the_modern_formats_are_untouched(self):
        for plate in ("UP32AB1234", "DL8CAF5010", "21BH2345AA", "MH121234"):
            assert postprocess.is_valid(plate), plate


class TestRecognitionStates:
    def test_a_clean_plate_is_confirmed(self):
        result = verdict(["UP32AB1234"] * 5, ocr=0.94)
        assert result.state is RecognitionState.CONFIRMED
        assert result.is_confirmed
        assert not result.review_required

    def test_a_moderately_read_plate_is_probable(self):
        """Every character settled, but on middling crops and middling
        recognizer confidence. A good reading, not an established one."""
        result = verdict(["UP32AB1234"] * 5, ocr=0.65, quality=0.40)
        assert not result.unstable_positions
        assert result.state is RecognitionState.PROBABLE
        assert result.review_required

    def test_an_unknown_state_code_can_never_be_confirmed(self):
        """Right shape, prefix not in the RTO list. It may be a valid new
        series or a misread prefix, and the system cannot tell which."""
        result = verdict(["ZZ32AB1234"] * 5, ocr=0.99, quality=0.99)
        assert result.cap_reason == "unknown_state_code"
        assert result.confidence <= KEEP.cap_unknown_state
        assert result.state is not RecognitionState.CONFIRMED

    def test_one_unstable_character_blocks_confirmation(self):
        """A plate can score well on agreement and quality while one
        character stays a coin flip. That is a good guess about a specific
        vehicle — which is what PROBABLE is for."""
        per_char = [0.99] * 5 + [0.50] + [0.99] * 4
        state = TrackState(track_id=1, camera_id=1)
        for text in ("UP32AB1234", "UP32AX1234", "UP32AB1234", "UP32AB1234"):
            state.reads.append(
                WeightedRead(
                    text=text, raw_text=text, rec_confidence=0.95,
                    per_char_confidence=list(per_char), plate_det_confidence=0.95,
                    quality=0.95, grammar=1.0, frame_ts=0.0,
                )
            )
        result = MultiFrameValidator(KEEP).validate(state)
        assert result.unstable_positions, "position 5 is contested B vs X"
        assert result.state is RecognitionState.PROBABLE

    def test_every_state_is_reachable(self):
        states = {
            verdict(["UP32AB1234"] * 5, ocr=0.94).state,
            verdict(["UP32AB1234"] * 5, ocr=0.65, quality=0.40).state,
            verdict(["UP1606"] * 5, ocr=0.94).state,
        }
        assert states == {
            RecognitionState.CONFIRMED,
            RecognitionState.PROBABLE,
            RecognitionState.UNRESOLVED,
        }


class TestConfidenceComposition:
    def test_ocr_confidence_cannot_carry_a_plate_on_its_own(self):
        """Weighted at 0.15. A perfect recognizer score with no cross-frame
        agreement and poor crops must not reach the CONFIRMED floor."""
        state = TrackState(track_id=1, camera_id=1)
        for text in ("UP32AB1234", "MH12CD5678", "KA05EF9012"):
            state.reads.append(
                WeightedRead(
                    text=text, raw_text=text, rec_confidence=1.0,
                    per_char_confidence=[1.0] * len(text), plate_det_confidence=0.5,
                    quality=0.36, grammar=1.0, frame_ts=0.0,
                )
            )
        result = MultiFrameValidator(KEEP).validate(state)
        assert result.confidence < KEEP.confirm_confidence

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ValidationConfig(weight_ocr=0.5)

    def test_confirm_floor_cannot_sit_below_the_probable_floor(self):
        with pytest.raises(ValueError, match="confirm_confidence"):
            ValidationConfig(min_final_confidence=0.9, confirm_confidence=0.5)

    def test_char_support_is_reported_separately(self):
        result = verdict(["UP32AB1234"] * 5, ocr=0.94)
        assert result.char_support > 0.0
        assert result.weakest_char_posterior > 0.0

    def test_unanimous_low_per_char_confidence_is_not_confirmed(self):
        """The fusion fix, seen through the validator: reads that agree on
        every character at 30% used to report support 1.0."""
        result = verdict(["UP32AB1234"] * 5, ocr=0.30, per_char=[0.30] * 10)
        assert result.char_support < 0.5
        assert result.state is not RecognitionState.CONFIRMED


class TestRegistrySnappingStaysConservative:
    def test_a_settled_position_is_not_revised_by_the_registry(self):
        """Snapping resolves characters we were UNSURE about. A position the
        frames pinned is not up for revision by a list that happens to hold a
        near-neighbour — that is how a visitor becomes a resident."""
        registry = SnapshotRegistry({"UP32AB1234": "resident"})
        # Every read says 8 at position 5, confidently. The registry offers B.
        state = state_with(["UP32A81234"] * 4, ocr=0.97)
        cfg = ValidationConfig(drop_low_confidence=False, registry_snap=True, snap_below=1.0)
        result = MultiFrameValidator(cfg, registry).validate(state)
        assert result is not None
        assert not any("registry-snap" in c for c in result.corrections)

    def test_an_unsettled_position_may_still_be_snapped(self):
        registry = SnapshotRegistry({"UP32AB1234": "resident"})
        state = TrackState(track_id=1, camera_id=1)
        # Position 5 is genuinely uncertain: reads disagree AND are unsure.
        for text in ("UP32A81234", "UP32AB1234", "UP32A81234", "UP32AQ1234"):
            state.reads.append(
                WeightedRead(
                    text=text, raw_text=text, rec_confidence=0.45,
                    per_char_confidence=[0.45] * 10, plate_det_confidence=0.6,
                    quality=0.5, grammar=postprocess.grammar_factor(text), frame_ts=0.0,
                )
            )
        cfg = ValidationConfig(
            drop_low_confidence=False, registry_snap=True,
            snap_below=1.0, snap_protect_posterior=0.80,
        )
        result = MultiFrameValidator(cfg, registry).validate(state)
        assert result is not None
        assert result.text == "UP32AB1234"

    def test_snapping_never_manufactures_a_blacklist_hit(self):
        """The target is reachable only by a SAME-TYPE confusion (B/D), which
        grammar repair never applies — both characters satisfy the format — so
        the registry is the only path to it, and it must refuse.
        """
        registry = SnapshotRegistry({"UP32DB1234": "blacklist"})
        state = TrackState(track_id=1, camera_id=1)
        # Reads disagree slightly, so support drops below snap_below and
        # snapping is actually attempted.
        for text in ("UP32BB1234", "UP32BB1234", "UP32BB1234", "UP32BB1284"):
            state.reads.append(
                WeightedRead(
                    text=text, raw_text=text, rec_confidence=0.45,
                    per_char_confidence=[0.45] * 10, plate_det_confidence=0.6,
                    quality=0.5, grammar=1.0, frame_ts=0.0,
                )
            )
        cfg = ValidationConfig(drop_low_confidence=False, registry_snap=True)
        result = MultiFrameValidator(cfg, registry).validate(state)
        assert result is not None
        assert result.text == "UP32BB1234"
        assert not any("registry-snap" in c for c in result.corrections)

    def test_ambiguous_registry_neighbours_produce_no_snap(self):
        """Two registered plates equally close means the system does not know
        which resident arrived. Guessing puts the wrong name on the gate log.

        ``0`` is confusable with both ``O`` and ``D``, so both registered
        plates fit and neither may be chosen.
        """
        registry = SnapshotRegistry({"UP32AO1234": "resident", "UP32AD1234": "resident"})
        assert registry.unique_confusable_neighbour("UP32A01234") is None


class TestAuditTrail:
    def test_the_read_evidence_survives(self):
        """Requirement: preserve the existing audit trail. The per-read trail
        is what answers 'why this plate?'."""
        from backend.app.events.event_builder import EventBuilder

        state = state_with(["UP32AB1234"] * 5, ocr=0.94)
        final = MultiFrameValidator(KEEP).validate(state)
        draft = EventBuilder(camera_id=1).build(state, final)
        assert len(draft.reads) == 5
        assert draft.reads[0].raw_text == "UP32AB1234"
        assert draft.plate_raw

    def test_the_new_decision_data_reaches_the_event(self):
        from backend.app.events.event_builder import EventBuilder

        state = state_with(["UP32AB1234"] * 5, ocr=0.94)
        final = MultiFrameValidator(KEEP).validate(state)
        draft = EventBuilder(camera_id=1).build(state, final)
        assert draft.recognition_state == "confirmed"
        assert draft.ocr_confidence == pytest.approx(0.94, abs=0.01)
        assert draft.char_support > 0
        # Round-trips through the disk spool.
        from backend.app.events.event_builder import EventDraft

        assert EventDraft.from_dict(draft.to_dict()).recognition_state == "confirmed"

    def test_a_capped_event_records_why(self):
        from backend.app.events.event_builder import EventBuilder

        state = state_with(["22473248"] * 5, ocr=0.94)
        final = MultiFrameValidator(KEEP).validate(state)
        draft = EventBuilder(camera_id=1).build(state, final)
        assert draft.confidence_cap_reason == "no_known_format"
        assert draft.recognition_state == "unresolved"
        assert draft.disputed


class TestTargetedRetry:
    def test_a_nearly_settled_plate_asks_for_more_reads(self):
        per_char = [0.99] * 5 + [0.50] + [0.99] * 4
        state = TrackState(track_id=1, camera_id=1)
        for text in ("UP32AB1234", "UP32AX1234", "UP32AB1234", "UP32AB1234"):
            state.reads.append(
                WeightedRead(
                    text=text, raw_text=text, rec_confidence=0.95,
                    per_char_confidence=list(per_char), plate_det_confidence=0.95,
                    quality=0.95, grammar=1.0, frame_ts=0.0,
                )
            )
        assert state.nearly_settled(max_unstable=2) == [5]

    def test_a_settled_plate_asks_for_nothing(self):
        state = state_with(["UP32AB1234"] * 4, ocr=0.97)
        assert state.nearly_settled(max_unstable=2) is None

    def test_a_hopeless_plate_asks_for_nothing(self):
        """Not nearly-settled but unread. One more look will not fix it, and
        spending on it is what the budget exists to prevent."""
        state = state_with(["UP32AB1234"] * 4, ocr=0.25, per_char=[0.25] * 10)
        assert state.nearly_settled(max_unstable=2) is None

    def test_a_retry_extends_the_budget_exactly_once(self):
        state = state_with(["UP32AB1234"] * 4, ocr=0.9)
        state.ocr_calls = 12
        assert state.ocr_remaining(12) == 0
        state.grant_retry(4, [5])
        assert state.ocr_remaining(12) == 4
        state.grant_retry(4, [5])  # a retry cannot grant itself another
        assert state.ocr_remaining(12) == 4
        assert state.retry_positions == [5]
