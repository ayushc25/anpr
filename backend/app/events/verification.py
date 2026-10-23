"""Conditional VLM verification: a second opinion that can only say "no".

After the validator settles on a candidate, ambiguous results get one more
question put to the VLM on the best crop: "does this read exactly <candidate>?"

DOWNGRADE-ONLY, BY CONSTRUCTION
-------------------------------
``apply`` is the only function that changes a plate's state, and it only ever
moves it DOWN (CONFIRMED -> PROBABLE). There is no code path that raises a
state, so a "Yes" can never promote a result and a bad plate can never become
class A on the strength of a model agreeing with itself. A yes is recorded — a
reviewer looking at a class B result is helped by knowing the VLM agreed — but
it changes nothing.

This is not conservatism for its own sake. The VLM is the same model that
produced the reads being checked. Asked to confirm a candidate it just helped
choose, it says yes far more readily than an independent check would; its
yes is weak evidence and its no is strong evidence. The asymmetry in the
decision rule matches the asymmetry in the evidence.

A verification that cannot run (queue full, service down, timeout) is not a
"no" — but for a result that was triggered because it WAS ambiguous, "could not
check" is not "checked and fine" either. ``on_unavailable`` picks which way to
fail; the default is the safe one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .multi_frame_validator import FinalPlate, RecognitionState
from .result_class import REASON_VERIFICATION_REJECTED, REASON_VERIFICATION_UNAVAILABLE


@dataclass(frozen=True)
class VerificationConfig:
    enabled: bool = True
    #: Crops put to the VLM. 1 is the cost-conscious default; each is a
    #: separate ~2 s call on CPU.
    crops: int = 1
    #: Mean P(yes) below this is a rejection.
    reject_below: float = 0.5
    #: Best plate-crop quality below this triggers verification.
    low_quality_below: float = 0.50
    #: Also verify PROBABLE results. They are already class B, so this is
    #: advisory: recorded for the reviewer, never able to change the class.
    verify_probable: bool = True
    #: "downgrade": a triggered CONFIRMED result that could not be verified
    #: becomes PROBABLE. "keep": it stays CONFIRMED.
    on_unavailable: str = "downgrade"

    def __post_init__(self) -> None:
        if self.on_unavailable not in ("downgrade", "keep"):
            raise ValueError("on_unavailable must be 'downgrade' or 'keep'")
        if self.crops < 1:
            raise ValueError("verification crops must be at least 1")


@dataclass
class VerificationOutcome:
    ran: bool = False
    #: Mean P(yes) over the crops that answered; None when none did.
    p_yes: Optional[float] = None
    answers: list = field(default_factory=list)
    unavailable_reason: str = ""


def triggers(final: FinalPlate, cfg: VerificationConfig) -> list[str]:
    """Why this result is worth a second look. Empty means: do not spend the call."""
    if not cfg.enabled:
        return []
    found: list[str] = []
    if final.distinct_variants > 1:
        found.append("multiple_variants")
    if _real_corrections(final):
        found.append("grammar_correction")
    if final.unstable_positions:
        found.append("unstable_positions")
    if final.cap_reason == "incomplete":
        found.append("incomplete")
    if final.best_quality < cfg.low_quality_below:
        found.append("low_quality")
    if final.state is RecognitionState.PROBABLE and cfg.verify_probable:
        found.append("probable")
    return found


def should_verify(final: FinalPlate, cfg: VerificationConfig) -> list[str]:
    """Triggers, but only for results where a verdict can mean something.

    UNRESOLVED is already class B and is not a candidate string anyone would
    stake anything on; asking the VLM whether it reads "UP1606" is a wasted
    call that cannot change the outcome.
    """
    if final.state is RecognitionState.UNRESOLVED:
        return []
    return triggers(final, cfg)


def apply(
    final: FinalPlate,
    outcome: VerificationOutcome,
    cfg: VerificationConfig,
    why: list[str],
    confirm_confidence: float,
) -> FinalPlate:
    """Record the outcome on ``final`` and DOWNGRADE it if warranted. Never upgrades."""
    record = {
        "ran": outcome.ran,
        "p_yes": None if outcome.p_yes is None else round(outcome.p_yes, 4),
        "answers": [None if a is None else round(a, 4) for a in outcome.answers],
        "triggers": list(why),
        "downgraded": False,
    }
    final.verification = record

    reject = outcome.p_yes is not None and outcome.p_yes < cfg.reject_below
    unavailable = outcome.p_yes is None
    if unavailable:
        record["unavailable"] = outcome.unavailable_reason or "no_answer"

    if final.state is not RecognitionState.CONFIRMED:
        # Nothing to downgrade. The record stays as advice to the reviewer.
        return final

    if reject:
        _downgrade(final, record, REASON_VERIFICATION_REJECTED, confirm_confidence)
    elif unavailable and cfg.on_unavailable == "downgrade":
        _downgrade(final, record, REASON_VERIFICATION_UNAVAILABLE, confirm_confidence)
    return final


def _downgrade(final: FinalPlate, record: dict, reason: str, confirm_confidence: float) -> None:
    final.state = RecognitionState.PROBABLE
    # Cap, as the validator's own caps do: a plate that is not settled must not
    # carry a confidence that reads as settled.
    final.confidence = round(min(final.confidence, max(0.0, confirm_confidence - 0.01)), 4)
    final.cap_reason = f"{final.cap_reason}+{reason}" if final.cap_reason else reason
    record["downgraded"] = True
    record["reason"] = reason


def _real_corrections(final: FinalPlate) -> list[str]:
    """Corrections that CHANGED the string. "grammar-decode declined" is the
    audit note for a correction that was considered and refused, which left the
    read exactly as the model produced it."""
    return [c for c in final.corrections if not c.startswith("grammar-decode declined")]
