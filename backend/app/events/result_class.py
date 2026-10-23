"""The three outcomes a vehicle's pass through the gate can have.

    A  auto-accepted     the plate is settled; safe to act on unattended
    B  human review      a vehicle was there and the system is not willing to
                         claim the plate — ambiguous, conflicting, unreadable,
                         or not read at all because the system was overloaded
    C  no plate visible  the vehicle crossed the read zone and not one usable
                         plate observation came out of it

``RecognitionState`` (CONFIRMED / PROBABLE / UNRESOLVED) is how sure the
validator is about a STRING. A/B/C is what the system does about the VEHICLE,
and it needs two cases the string-level state cannot express: a vehicle with no
string at all. Hence a separate, deliberately small vocabulary, derived from
the state by one function so there is exactly one place that decides it.

C is not "nothing was detected". A vehicle that never reached the read zone
produces no event at all; C means the system looked, in the place where a plate
should be legible, and found nothing usable. That distinction is the point of
storing it: it is the difference between an empty gate and a vehicle whose
plate was missing, covered, mud-caked or outside the camera's view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .multi_frame_validator import FinalPlate, RecognitionState


class ResultClass(str, Enum):
    AUTO_ACCEPTED = "A"
    HUMAN_REVIEW = "B"
    NO_PLATE = "C"


#: Machine-readable reasons. Stored on the event so a reviewer can see WHY a
#: vehicle landed in B without reading the log.
REASON_CONFIRMED = "confirmed"
REASON_AMBIGUOUS = "ambiguous"
REASON_UNRESOLVED = "unresolved"
REASON_VERIFICATION_REJECTED = "verification_rejected"
REASON_VERIFICATION_UNAVAILABLE = "verification_unavailable"
REASON_QUEUE_OVERFLOW = "vlm_queue_overflow"
REASON_VLM_UNAVAILABLE = "vlm_unavailable"
REASON_VLM_TIMEOUT = "vlm_timeout"
REASON_VLM_UNREADABLE = "vlm_unreadable"
REASON_NO_CONSENSUS = "no_consensus"
REASON_NO_USABLE_OBSERVATION = "no_usable_plate_observation"


@dataclass
class ResultInfo:
    result_class: ResultClass
    reason: str
    #: Free-form audit detail (verification verdict, job latency, ...).
    detail: dict = field(default_factory=dict)

    @property
    def value(self) -> str:
        return self.result_class.value


def class_for_state(state: RecognitionState) -> ResultClass:
    """CONFIRMED -> A. PROBABLE and UNRESOLVED -> B. There is no state that
    maps to C: C is the absence of a string, which a state cannot describe."""
    if state is RecognitionState.CONFIRMED:
        return ResultClass.AUTO_ACCEPTED
    return ResultClass.HUMAN_REVIEW


def classify(final: Optional[FinalPlate]) -> ResultInfo:
    """Result for a vehicle the validator produced a plate for."""
    if final is None:
        return ResultInfo(ResultClass.HUMAN_REVIEW, REASON_NO_CONSENSUS)
    result_class = class_for_state(final.state)
    if result_class is ResultClass.AUTO_ACCEPTED:
        reason = REASON_CONFIRMED
    elif final.state is RecognitionState.UNRESOLVED:
        reason = REASON_UNRESOLVED
    else:
        reason = REASON_AMBIGUOUS
    # A downgrade by verification leaves the state at PROBABLE; say why.
    verdict = final.verification or {}
    if verdict.get("downgraded"):
        reason = verdict.get("reason") or REASON_VERIFICATION_REJECTED
    return ResultInfo(result_class, reason, detail={"verification": dict(verdict)} if verdict else {})


def review(reason: str, **detail) -> ResultInfo:
    """A B result with no plate string: the vehicle needs a human, and the
    reason says which of the ways the system could not read it applied."""
    return ResultInfo(ResultClass.HUMAN_REVIEW, reason, detail)


def no_plate(**detail) -> ResultInfo:
    return ResultInfo(ResultClass.NO_PLATE, REASON_NO_USABLE_OBSERVATION, detail)
