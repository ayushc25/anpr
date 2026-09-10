"""Ingest: EventDraft -> Event, in one transaction.

Order matters here and is deliberate:

  1. Persist media first, so the row always points at a file that exists.
  2. Resolve the registry match and the effective status.
  3. Insert the event and its evidence rows.
  4. Evaluate rules against the persisted event and insert alerts.
  5. Enqueue notifications — ENQUEUE only. Nothing here talks to an SMTP
     server, because a slow mail host must never hold a database transaction
     open while vehicles keep arriving.

The caller commits. That keeps the whole thing atomic and lets the API and the
direct-mode worker share exactly the same code path.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..db.base import AlertStatus, EventDirection, NotificationStatus, VehicleStatus
from ..events.event_builder import EventDraft, decode_image
from ..events.rules_engine import RulesEngine, RuleContext
from ..models.alert import Alert, AlertRule, Notification
from ..models.camera import Camera
from ..models.event import Event
from ..repositories.event_repo import EventRepository
from ..repositories.vehicle_repo import VehicleRepository
from ..storage.media_store import MediaStore

logger = logging.getLogger("anpr.services.event")


class DuplicateEvent(Exception):
    """The same event_uid has already been ingested."""

    def __init__(self, event: Event):
        super().__init__(f"event {event.event_uid} already exists")
        self.event = event


class EventProcessor:
    def __init__(self, session: Session, media_store: MediaStore | None = None):
        self.session = session
        self.events = EventRepository(session)
        self.vehicles = VehicleRepository(session)
        if media_store is None:
            from ..core.config import get_settings

            media_store = MediaStore(get_settings().media_root)
        self.media = media_store

    def ingest(self, draft: EventDraft) -> Event:
        existing = self.events.by_uid(draft.event_uid)
        if existing is not None:
            # The worker retried after a timeout that actually succeeded.
            raise DuplicateEvent(existing)

        detected_at = _parse_ts(draft.detected_at)
        camera = self.session.get(Camera, draft.camera_id) if draft.camera_id else None
        camera_code = (camera.code or f"cam{camera.id}") if camera else "cam"

        stored = self.media.save(
            event_uid=draft.event_uid,
            camera_code=camera_code,
            plate=draft.plate_number,
            when=detected_at,
            vehicle_jpeg=decode_image(draft.vehicle_image_b64),
            plate_jpeg=decode_image(draft.plate_image_b64),
        )

        vehicle, match_method = self.vehicles.match(draft.plate_number, detected_at)
        status = self.vehicles.resolve_status(vehicle, detected_at)

        corrections = list(draft.corrections)
        if match_method in ("alias", "confusable", "nearest") and vehicle is not None:
            corrections.append(f"registry-{match_method}: {draft.plate_number}->{vehicle.plate_number}")

        event = Event(
            event_uid=draft.event_uid,
            camera_id=draft.camera_id,
            # Store the registry's spelling once matched, so history for a
            # vehicle is not split across two spellings of its plate.
            plate_number=vehicle.plate_number if vehicle else draft.plate_number,
            plate_raw=draft.plate_raw or draft.plate_number,
            vehicle_id=vehicle.id if vehicle else None,
            resident_id=vehicle.resident_id if vehicle else None,
            direction=_direction(draft.direction),
            status=status,
            vehicle_type=draft.vehicle_type,
            vehicle_color=draft.vehicle_color,
            plate_color=draft.plate_color,
            plate_confidence=draft.plate_confidence,
            detect_confidence=draft.detect_confidence,
            validation_support=draft.validation_support,
            read_count=draft.read_count,
            distinct_variants=draft.distinct_variants,
            grammar_valid=draft.grammar_valid,
            corrections=corrections,
            vehicle_image_path=stored.vehicle_path,
            plate_image_path=stored.plate_path,
            track_id=draft.track_id,
            finalize_reason=draft.finalize_reason,
            is_disputed=draft.disputed,
            detected_at=detected_at,
        )
        self.events.add(event)

        if draft.reads:
            self.events.add_reads(
                event.id,
                [
                    {
                        "frame_ts": r.frame_ts,
                        "raw_text": r.raw_text[:48],
                        "normalized_text": r.normalized_text[:32],
                        "rec_confidence": r.rec_confidence,
                        "plate_det_confidence": r.plate_det_confidence,
                        "quality_score": r.quality_score,
                        "weight": r.weight,
                    }
                    for r in draft.reads
                ],
            )

        self._raise_alerts(event, camera, vehicle is not None and status != VehicleStatus.unknown)

        logger.info(
            "event %s: %s %s status=%s conf=%.2f match=%s",
            event.event_uid, event.plate_number, event.direction.value,
            event.status.value, event.plate_confidence, match_method,
        )
        return event

    # -- alerts ------------------------------------------------------------
    def _raise_alerts(self, event: Event, camera: Optional[Camera], registered: bool) -> None:
        rules = self.session.query(AlertRule).filter(AlertRule.is_enabled.is_(True)).all()
        if not rules:
            return

        ctx = RuleContext(
            now=datetime.utcnow(),
            recent_event_count=self.events.recent_for_vehicle,
            camera_name=camera.name if camera else "",
            registered=registered,
        )
        engine = RulesEngine(rules, cooldown_check=self._in_cooldown)

        for rule_row, draft in engine.evaluate(event, ctx):
            alert = Alert(
                rule_id=rule_row.id,
                event_id=event.id,
                code=draft.code,
                severity=draft.severity,
                title=draft.title,
                message=draft.message,
                context=draft.context,
                status=AlertStatus.new,
            )
            self.session.add(alert)
            self.session.flush()
            for channel in rule_row.channels or []:
                # Persisted before any delivery attempt: the outbox is what
                # makes a mail outage survivable.
                self.session.add(
                    Notification(
                        alert_id=alert.id,
                        channel=channel,
                        status=NotificationStatus.pending,
                        payload={"title": draft.title, "message": draft.message},
                    )
                )
            logger.info("alert %s raised for %s", draft.code, event.plate_number)

    def _in_cooldown(self, code: str, plate: str, cooldown_seconds: int) -> bool:
        if cooldown_seconds <= 0:
            return False
        since = datetime.utcnow() - timedelta(seconds=cooldown_seconds)
        return (
            self.session.query(Alert.id)
            .join(Event, Alert.event_id == Event.id)
            .filter(Alert.code == code, Event.plate_number == plate, Alert.created_at >= since)
            .first()
            is not None
        )

    # -- operator correction ----------------------------------------------
    def correct(self, event_id: int, plate: str, user_id: int, learn_alias: bool = True) -> Optional[Event]:
        """Apply an operator's correction and, optionally, learn from it.

        Recording the original read as an alias is the cheap half of the
        feedback loop: the same camera will make the same mistake on the same
        vehicle tomorrow.
        """
        from ..ai.plate_recognizer import postprocess

        event = self.events.get(event_id)
        if event is None:
            return None

        original = event.plate_number
        corrected = postprocess.normalize(plate)
        if not corrected:
            raise ValueError(f"'{plate}' is not a usable plate number")

        vehicle, _ = self.vehicles.match(corrected)
        event.plate_number = corrected
        event.vehicle_id = vehicle.id if vehicle else None
        event.resident_id = vehicle.resident_id if vehicle else None
        event.status = self.vehicles.resolve_status(vehicle, event.detected_at or datetime.utcnow())
        event.is_manual_override = True
        event.is_disputed = False
        event.reviewed_by = user_id
        event.reviewed_at = datetime.utcnow()

        if learn_alias and vehicle is not None and original and original != corrected:
            self.vehicles.add_alias(vehicle.id, original, reason=f"operator correction on event {event_id}")

        logger.info("event %s corrected %s -> %s by user %s", event_id, original, corrected, user_id)
        return event


def _parse_ts(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            parsed = datetime.now(timezone.utc)
    else:
        parsed = datetime.now(timezone.utc)
    # Stored naive-UTC to match the rest of the schema.
    return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed


def _direction(value: str | None) -> EventDirection:
    try:
        return EventDirection(value or "unknown")
    except ValueError:
        return EventDirection.unknown
