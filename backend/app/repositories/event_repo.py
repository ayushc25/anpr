"""Event persistence and the queries the dashboard and reports run."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from ..db.base import EventDirection, VehicleStatus
from ..models.event import Event, EventRead


class EventRepository:
    def __init__(self, session: Session):
        self.session = session

    # -- writes ------------------------------------------------------------
    def by_uid(self, event_uid: str) -> Optional[Event]:
        if not event_uid:
            return None
        return self.session.query(Event).filter(Event.event_uid == event_uid).first()

    def add(self, event: Event) -> Event:
        self.session.add(event)
        self.session.flush()  # assign the id so reads can reference it
        return event

    def add_reads(self, event_id: int, rows: Iterable[dict]) -> int:
        """Bulk insert of the evidence trail.

        ``bulk_insert_mappings`` rather than ORM objects: this runs on every
        event, a dozen rows at a time, and the ORM overhead is pure waste for
        rows nothing will mutate.
        """
        payload = [{**row, "event_id": event_id} for row in rows]
        if not payload:
            return 0
        self.session.bulk_insert_mappings(EventRead, payload)
        return len(payload)

    # -- reads -------------------------------------------------------------
    def get(self, event_id: int) -> Optional[Event]:
        return (
            self.session.query(Event)
            .options(joinedload(Event.camera), joinedload(Event.vehicle), joinedload(Event.resident))
            .filter(Event.id == event_id)
            .first()
        )

    def reads_for(self, event_id: int) -> list[EventRead]:
        return (
            self.session.query(EventRead)
            .filter(EventRead.event_id == event_id)
            .order_by(EventRead.frame_ts)
            .all()
        )

    def search(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        camera_id: int | None = None,
        direction: str | None = None,
        status: str | None = None,
        plate: str | None = None,
        min_confidence: float | None = None,
        disputed_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Event], int]:
        stmt = self.session.query(Event).options(
            joinedload(Event.camera), joinedload(Event.vehicle), joinedload(Event.resident)
        )
        stmt = self._apply_filters(
            stmt, start, end, camera_id, direction, status, plate, min_confidence, disputed_only
        )
        total = stmt.with_entities(func.count(Event.id)).scalar() or 0
        rows = stmt.order_by(Event.detected_at.desc()).limit(limit).offset(offset).all()
        return rows, total

    def iter_for_export(self, **filters) -> Iterable[Event]:
        """Server-side cursor for the Excel writer.

        ``yield_per`` is what keeps a 200k-row year-end export off the heap of
        an 8 GB edge box.
        """
        stmt = self.session.query(Event).options(
            joinedload(Event.camera), joinedload(Event.vehicle), joinedload(Event.resident)
        )
        stmt = self._apply_filters(stmt, **filters)
        return stmt.order_by(Event.detected_at.desc()).yield_per(500)

    def _apply_filters(
        self,
        stmt,
        start=None,
        end=None,
        camera_id=None,
        direction=None,
        status=None,
        plate=None,
        min_confidence=None,
        disputed_only=False,
    ):
        if start:
            stmt = stmt.filter(Event.detected_at >= start)
        if end:
            stmt = stmt.filter(Event.detected_at <= end)
        if camera_id:
            stmt = stmt.filter(Event.camera_id == camera_id)
        if direction:
            stmt = stmt.filter(Event.direction == direction)
        if status:
            stmt = stmt.filter(Event.status == status)
        if plate:
            stmt = stmt.filter(Event.plate_number.like(f"%{plate.upper()}%"))
        if min_confidence is not None:
            stmt = stmt.filter(Event.plate_confidence >= min_confidence)
        if disputed_only:
            stmt = stmt.filter(Event.is_disputed.is_(True))
        return stmt

    def history_for_plate(self, plate: str, limit: int = 200) -> list[Event]:
        return (
            self.session.query(Event)
            .options(joinedload(Event.camera))
            .filter(Event.plate_number == plate.upper())
            .order_by(Event.detected_at.desc())
            .limit(limit)
            .all()
        )

    def recent_for_vehicle(self, plate: str, since: datetime) -> int:
        return (
            self.session.query(func.count(Event.id))
            .filter(Event.plate_number == plate, Event.detected_at >= since)
            .scalar()
            or 0
        )

    # -- dashboard ---------------------------------------------------------
    def day_stats(self, day_start: datetime) -> dict:
        day_end = day_start + timedelta(days=1)
        base = self.session.query(Event).filter(
            Event.detected_at >= day_start, Event.detected_at < day_end
        )

        def count(query) -> int:
            return query.with_entities(func.count(Event.id)).scalar() or 0

        registered = count(base.filter(Event.status.in_([VehicleStatus.registered, VehicleStatus.whitelist])))
        return {
            "total": count(base),
            "entries": count(base.filter(Event.direction == EventDirection.in_)),
            "exits": count(base.filter(Event.direction == EventDirection.out_)),
            "registered": registered,
            "unknown": count(base.filter(Event.status == VehicleStatus.unknown)),
            "blacklist": count(base.filter(Event.status == VehicleStatus.blacklist)),
            "disputed": count(base.filter(Event.is_disputed.is_(True))),
        }

    def trend(self, since: datetime, bucket: str = "hour") -> list[tuple[datetime, int]]:
        truncated = func.date_trunc(bucket, Event.detected_at)
        rows = (
            self.session.query(truncated.label("bucket"), func.count(Event.id))
            .filter(Event.detected_at >= since)
            .group_by("bucket")
            .order_by("bucket")
            .all()
        )
        return [(row[0], row[1]) for row in rows]

    def purge_reads_older_than(self, cutoff: datetime) -> int:
        """Evidence rows age out long before the events they belong to."""
        subquery = self.session.query(Event.id).filter(Event.detected_at < cutoff).subquery()
        deleted = (
            self.session.query(EventRead)
            .filter(EventRead.event_id.in_(subquery))
            .delete(synchronize_session=False)
        )
        return deleted or 0
