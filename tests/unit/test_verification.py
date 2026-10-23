"""Conditional VLM verification: ambiguous results only, downgrade only.

The property that matters is structural — nothing in this module can raise a
result's state — so most of these tests try to make it do exactly that.
"""
from __future__ import annotations

import pytest

from backend.app.events.multi_frame_validator import FinalPlate, RecognitionState
from backend.app.events.result_class import (
    REASON_VERIFICATION_REJECTED,
    REASON_VERIFICATION_UNAVAILABLE,
    ResultClass,
    classify,
)
from backend.app.events.verification import (
    VerificationConfig,
    VerificationOutcome,
    apply,
    should_verify,
    triggers,
)


def plate(state=RecognitionState.CONFIRMED, **overrides) -> FinalPlate:
    base = dict(
        text="UP32AB1234", confidence=0.93, support=1.0, read_count=6, distinct_variants=1,
        grammar_valid=True, best_quality=0.8, state=state,
    )
    base.update(overrides)
    return FinalPlate(**base)


CFG = VerificationConfig()


# ---- triggers ---------------------------------------------------------------
def test_clean_confirmed_result_is_not_verified():
    assert should_verify(plate(), CFG) == []


@pytest.mark.parametrize("overrides,expected", [
    ({"distinct_variants": 3}, "multiple_variants"),
    ({"corrections": ["UP32A81234->UP32AB1234"]}, "grammar_correction"),
    ({"unstable_positions": ["pos4=B@0.52 vs 8@0.44"]}, "unstable_positions"),
    ({"best_quality": 0.4}, "low_quality"),
])
def test_each_ambiguity_triggers_verification(overrides, expected):
    assert expected in should_verify(plate(**overrides), CFG)


def test_declined_grammar_decode_is_not_a_correction():
    """'declined' is the audit note for a fix considered and refused; the read
    reached the validator exactly as the model produced it."""
    final = plate(corrections=["grammar-decode declined: pos2 no digit among alternatives"])
    assert "grammar_correction" not in triggers(final, CFG)


def test_probable_is_verified_only_when_configured():
    probable = plate(RecognitionState.PROBABLE, confidence=0.7)
    assert "probable" in should_verify(probable, CFG)
    assert "probable" not in should_verify(probable, VerificationConfig(verify_probable=False))


def test_unresolved_is_never_verified():
    """Already class B and not a string anyone stakes anything on."""
    assert should_verify(plate(RecognitionState.UNRESOLVED, distinct_variants=4, best_quality=0.1), CFG) == []


def test_disabled_never_triggers():
    assert should_verify(plate(distinct_variants=4), VerificationConfig(enabled=False)) == []


# ---- downgrade only ---------------------------------------------------------
def run(final, p_yes, why=("multiple_variants",), cfg=CFG, ran=True):
    outcome = VerificationOutcome(ran=ran, p_yes=p_yes, answers=[p_yes] if p_yes is not None else [])
    return apply(final, outcome, cfg, list(why), confirm_confidence=0.80)


def test_a_no_downgrades_confirmed_to_probable_and_class_b():
    final = run(plate(), p_yes=0.05)
    assert final.state is RecognitionState.PROBABLE
    assert classify(final).result_class is ResultClass.HUMAN_REVIEW
    assert classify(final).reason == REASON_VERIFICATION_REJECTED
    assert final.verification["downgraded"] is True
    # The number must stop reading as settled too.
    assert final.confidence < 0.80
    assert "verification_rejected" in final.cap_reason


def test_a_yes_never_upgrades_probable():
    probable = plate(RecognitionState.PROBABLE, confidence=0.7)
    run(probable, p_yes=0.999, why=("probable",))
    assert probable.state is RecognitionState.PROBABLE
    assert classify(probable).result_class is ResultClass.HUMAN_REVIEW
    assert probable.confidence == 0.7
    # ...but the agreement is kept for the reviewer.
    assert probable.verification["p_yes"] == 0.999
    assert probable.verification["downgraded"] is False


def test_a_yes_never_upgrades_unresolved():
    unresolved = plate(RecognitionState.UNRESOLVED, confidence=0.3)
    run(unresolved, p_yes=1.0, why=("incomplete",))
    assert unresolved.state is RecognitionState.UNRESOLVED
    assert classify(unresolved).result_class is ResultClass.HUMAN_REVIEW


def test_a_yes_leaves_confirmed_confirmed():
    final = run(plate(), p_yes=0.97)
    assert final.state is RecognitionState.CONFIRMED
    assert classify(final).result_class is ResultClass.AUTO_ACCEPTED
    assert final.verification["downgraded"] is False


def test_no_state_transition_can_raise_a_rank():
    """Exhaustive: every starting state x every answer never ends higher."""
    rank = {RecognitionState.UNRESOLVED: 0, RecognitionState.PROBABLE: 1, RecognitionState.CONFIRMED: 2}
    for start in RecognitionState:
        for p_yes in (None, 0.0, 0.3, 0.5, 0.9, 1.0):
            for policy in ("downgrade", "keep"):
                final = run(plate(start), p_yes, cfg=VerificationConfig(on_unavailable=policy))
                assert rank[final.state] <= rank[start], (start, p_yes, policy)


def test_unavailable_verification_downgrades_by_default():
    """'Could not check' is not 'checked and fine' for a result that was
    triggered because it was ambiguous."""
    final = run(plate(), p_yes=None, ran=False)
    assert final.state is RecognitionState.PROBABLE
    assert classify(final).reason == REASON_VERIFICATION_UNAVAILABLE
    assert final.verification["unavailable"]


def test_unavailable_verification_can_be_configured_to_keep():
    final = run(plate(), p_yes=None, ran=False, cfg=VerificationConfig(on_unavailable="keep"))
    assert final.state is RecognitionState.CONFIRMED


def test_reject_threshold_is_respected():
    assert run(plate(), p_yes=0.49).state is RecognitionState.PROBABLE
    assert run(plate(), p_yes=0.50).state is RecognitionState.CONFIRMED


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        VerificationConfig(on_unavailable="promote")
    with pytest.raises(ValueError):
        VerificationConfig(crops=0)
