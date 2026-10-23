"""TrackState -> EventDraft.

The draft is a plain, JSON-serializable record plus two encoded JPEGs. It is
deliberately not an ORM object: the worker process that builds it must not
hold a database session, and the same draft has to survive a round trip
through the disk spool when Postgres is unreachable.
"""
from __future__ import annotations

import base64
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from ..video.line_crossing import CrossDirection
from .multi_frame_validator import FinalPlate
from .result_class import ResultClass, ResultInfo, classify
from .track_state import TrackState


@dataclass
class ReadEvidence:
    """One row of the per-frame trail that answers "why this plate?"."""

    raw_text: str
    normalized_text: str
    rec_confidence: float
    plate_det_confidence: float
    quality_score: float
    weight: float
    frame_ts: float


@dataclass
class EventDraft:
    event_uid: str
    camera_id: int
    #: None for a vehicle the system saw but could not name (class B with no
    #: usable string, or class C). Never an empty string and never a placeholder
    #: like "UNKNOWN": a fake plate is a row that matches the wrong thing in a
    #: search, a registry lookup or an alert rule.
    plate_number: Optional[str]
    plate_raw: str
    direction: str
    vehicle_type: str
    detected_at: str                     # ISO-8601 UTC
    #: ANPR confidence: how sure the SYSTEM is of the registration. Capped by
    #: completeness and format — see multi_frame_validator. The number a UI
    #: should show.
    plate_confidence: float
    detect_confidence: float
    validation_support: float
    read_count: int
    distinct_variants: int
    grammar_valid: bool
    #: CONFIRMED / PROBABLE / UNRESOLVED. Carried explicitly so a consumer
    #: never has to re-derive it from a threshold.
    recognition_state: str = "probable"
    #: The RECOGNIZER's own confidence in the glyphs. Diagnostic only: this is
    #: what the model claimed, not what the system concluded, and the two
    #: differ precisely when it matters most.
    ocr_confidence: float = 0.0
    char_support: float = 0.0
    weakest_char_posterior: float = 0.0
    unstable_positions: list[str] = field(default_factory=list)
    #: Why the ANPR confidence was capped below the raw evidence score.
    confidence_cap_reason: str = ""
    #: Character positions an extra-budget retry was bought to resolve.
    retry_positions: list[int] = field(default_factory=list)
    vehicle_color: Optional[str] = None
    plate_color: Optional[str] = None
    corrections: list[str] = field(default_factory=list)
    track_id: int = 0
    finalize_reason: str = ""
    disputed: bool = False
    duration_seconds: float = 0.0
    reads: list[ReadEvidence] = field(default_factory=list)

    # -- A / B / C outcome -------------------------------------------------
    #: "A" auto-accepted, "B" human review, "C" no plate visible. Empty on a
    #: draft spooled before this field existed; ingest derives it from
    #: ``recognition_state`` in that case.
    result_class: str = ""
    #: Why — a short machine-readable reason ("vlm_queue_overflow",
    #: "verification_rejected", "no_usable_plate_observation", ...).
    result_reason: str = ""
    #: The verification pass's verdict, when one ran.
    verification: dict = field(default_factory=dict)
    #: Extra base64 JPEGs of the best plate crops, saved so a reviewer can see
    #: what the system had to work with. Populated for B and C results.
    review_crops_b64: list[str] = field(default_factory=list)

    #: base64 JPEGs. Carried inline so a spooled draft is self-contained;
    #: the API decodes them to the media store on ingest.
    vehicle_image_b64: Optional[str] = None
    plate_image_b64: Optional[str] = None

    # -- media provenance --------------------------------------------------
    #
    # Where the two images came from, so the pairing between a plate and its
    # photograph is inspectable rather than assumed. Without this an operator
    # looking at a mismatched row has no way to tell whether the plate or the
    # picture is the wrong one.
    #: "hypothesis" — the images come from frames that produced the winning
    #: plate string. "derived" — from frames producing a string the winner was
    #: derived from (repair, registry snap, positional fusion).
    #: "fallback_global" — NO read's imagery supports the winning plate, so
    #: the track's best crops were used and the pairing is NOT guaranteed.
    media_source: str = ""
    #: Processed-frame index and timestamp each image was captured on.
    vehicle_image_frame: int = 0
    vehicle_image_ts: float = 0.0
    plate_image_frame: int = 0
    plate_image_ts: float = 0.0
    #: Reads behind the winning plate, and the window they span.
    winning_read_count: int = 0
    winning_window: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EventDraft":
        reads = [ReadEvidence(**r) for r in data.get("reads", [])]
        return cls(**{**data, "reads": reads})


def _encode(image: Optional[np.ndarray], quality: int = 85) -> Optional[str]:
    if image is None or image.size == 0:
        return None
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _encode_many(images: Optional[list[np.ndarray]]) -> list[str]:
    out = []
    for image in images or []:
        encoded = _encode(image, quality=92)
        if encoded:
            out.append(encoded)
    return out


def decode_image(encoded: Optional[str]) -> Optional[bytes]:
    return base64.b64decode(encoded) if encoded else None


class EventBuilder:
    def __init__(self, camera_id: int, keep_read_evidence: bool = True, max_evidence_rows: int = 12):
        self.camera_id = camera_id
        self.keep_read_evidence = keep_read_evidence
        self.max_evidence_rows = max_evidence_rows

    def build(
        self,
        state: TrackState,
        final: Optional[FinalPlate],
        direction: CrossDirection | None = None,
        detected_at: datetime | None = None,
        result: Optional[ResultInfo] = None,
        review_crops: Optional[list[np.ndarray]] = None,
    ) -> EventDraft:
        """``final`` is None for a vehicle with no plate string (class B
        overload / unreadable, or class C); ``result`` then says which."""
        resolved_direction = direction or state.direction
        when = detected_at or datetime.now(timezone.utc)
        if result is None:
            result = classify(final)

        evidence: list[ReadEvidence] = []
        if self.keep_read_evidence:
            # Keep the heaviest reads: those are the ones that decided the
            # vote, and therefore the ones worth showing an operator.
            for read in sorted(state.reads, key=lambda r: r.weight, reverse=True)[: self.max_evidence_rows]:
                evidence.append(
                    ReadEvidence(
                        raw_text=read.raw_text,
                        normalized_text=read.text,
                        rec_confidence=round(read.rec_confidence, 4),
                        plate_det_confidence=round(read.plate_det_confidence, 4),
                        quality_score=round(read.quality, 4),
                        weight=round(read.weight, 5),
                        frame_ts=read.frame_ts,
                    )
                )
            evidence.sort(key=lambda r: r.frame_ts)

        if final is None:
            return self._build_without_plate(
                state, resolved_direction, when, evidence, result, review_crops
            )

        return EventDraft(
            event_uid=str(uuid.uuid4()),
            camera_id=self.camera_id,
            plate_number=final.text,
            plate_raw=max(state.reads, key=lambda r: r.weight).raw_text if state.reads else "",
            direction=_direction_value(resolved_direction),
            vehicle_type=state.vehicle_type,
            vehicle_color=state.vehicle_color,
            plate_color=state.plate_color,
            detected_at=when.isoformat(),
            plate_confidence=final.confidence,
            detect_confidence=round(state.detect_confidence, 4),
            validation_support=final.support,
            read_count=final.read_count,
            distinct_variants=final.distinct_variants,
            grammar_valid=final.grammar_valid,
            recognition_state=final.state.value,
            ocr_confidence=final.ocr_confidence,
            char_support=final.char_support,
            weakest_char_posterior=final.weakest_char_posterior,
            unstable_positions=list(final.unstable_positions),
            confidence_cap_reason=final.cap_reason,
            retry_positions=list(state.retry_positions),
            corrections=list(final.corrections),
            track_id=state.track_id,
            finalize_reason=state.finalize_reason,
            # Anything the validator did not settle wants a human eye. The
            # validator already marks the state disputed for those, so this
            # stays the single source of truth rather than a second rule.
            disputed=state.disputed or result.result_class is not ResultClass.AUTO_ACCEPTED,
            duration_seconds=round(state.duration, 2),
            reads=evidence,
            result_class=result.value,
            result_reason=result.reason,
            verification=dict(final.verification or {}),
            review_crops_b64=_encode_many(review_crops) if result.result_class is not ResultClass.AUTO_ACCEPTED else [],
            **_media_fields(state, final.text),
        )

    def _build_without_plate(
        self,
        state: TrackState,
        direction,
        when: datetime,
        evidence: list[ReadEvidence],
        result: ResultInfo,
        review_crops: Optional[list[np.ndarray]],
    ) -> EventDraft:
        """A vehicle was here and there is no plate string to report.

        Everything numeric about the plate is zero because there is no plate;
        ``recognition_state`` is "unresolved", the honest statement of "the
        system did not identify it". The images are the track's best crops,
        and ``media_source`` says they are a best-effort pairing, since there
        is no winning string for them to be paired with.
        """
        return EventDraft(
            event_uid=str(uuid.uuid4()),
            camera_id=self.camera_id,
            plate_number=None,
            plate_raw="",
            direction=_direction_value(direction),
            vehicle_type=state.vehicle_type,
            vehicle_color=state.vehicle_color,
            plate_color=state.plate_color,
            detected_at=when.isoformat(),
            plate_confidence=0.0,
            detect_confidence=round(state.detect_confidence, 4),
            validation_support=0.0,
            read_count=len(state.reads),
            distinct_variants=state.distinct_texts,
            grammar_valid=False,
            recognition_state="unresolved",
            track_id=state.track_id,
            finalize_reason=state.finalize_reason,
            disputed=True,
            duration_seconds=round(state.duration, 2),
            reads=evidence,
            result_class=result.value,
            result_reason=result.reason,
            verification=dict(result.detail.get("verification", {})) if result.detail else {},
            review_crops_b64=_encode_many(review_crops),
            vehicle_image_b64=_encode(state.best_vehicle_crop),
            plate_image_b64=_encode(state.best_plate_crop, quality=92),
            media_source="best_observation",
        )


def _media_fields(state: TrackState, plate: str) -> dict:
    """Pick the event's two images from the WINNING plate's own evidence.

    The bug this exists to prevent: images and plate were chosen by
    independent criteria over the same track. The plate came from a weighted
    vote across all reads; the images came from the track's single
    highest-quality frame. When a track's identity drifted across vehicles —
    which a 40-second track in dense traffic does — the two landed on
    different ones, producing an event that stated plate ``HR29BG7381`` while
    storing a photograph of the car carrying ``UP14FU2031``.

    Now the plate hypothesis carries its own crops, so whichever string wins
    validation, the images that come with it were captured on frames that
    produced THAT string. The pairing is structural rather than coincidental.

    A quality regression is possible and is the correct trade: a sharper crop
    of a different vehicle is not a better image of this one.

    ``fallback_global`` is retained for the case where no read's imagery
    supports the winning plate at all, and is reported rather than hidden —
    for such an event the pairing is genuinely not guaranteed.
    """
    found = state.evidence_for(plate)
    if found is None or (found.plate_crop is None and found.vehicle_crop is None):
        return {
            "vehicle_image_b64": _encode(state.best_vehicle_crop),
            "plate_image_b64": _encode(state.best_plate_crop, quality=92),
            "media_source": "fallback_global",
        }

    # A hypothesis may hold only one of the two crops — a retry read is
    # recognized from a banked plate crop and has no vehicle crop. Fill only
    # the missing one from the global best, and say so.
    vehicle_crop = found.vehicle_crop
    source = "hypothesis" if found.text == plate else "derived"
    if vehicle_crop is None:
        vehicle_crop = state.best_vehicle_crop
        source += "+global_vehicle"

    return {
        "vehicle_image_b64": _encode(vehicle_crop),
        "plate_image_b64": _encode(found.plate_crop, quality=92),
        "media_source": source,
        "vehicle_image_frame": found.vehicle_frame_idx,
        "vehicle_image_ts": found.vehicle_ts,
        "plate_image_frame": found.plate_frame_idx,
        "plate_image_ts": found.plate_ts,
        "winning_read_count": found.read_count,
        "winning_window": (round(found.first_ts, 3), round(found.last_ts, 3)),
    }


def _direction_value(direction: CrossDirection | str | None) -> str:
    """Events must record a real in/out.

    The prototype copied ``camera.direction`` verbatim, including the value
    ``both``, which makes entry/exit reporting meaningless. Here an
    unclassified crossing stays ``unknown`` and is visible as such, rather than
    being silently counted as an entry.
    """
    if direction is None:
        return "unknown"
    value = direction.value if isinstance(direction, CrossDirection) else str(direction)
    return value if value in ("in", "out") else "unknown"
