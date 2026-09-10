from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..core.config import display_tz
from ..database import get_db
from ..deps import get_current_user
from ..services import camera_manager

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _to_utc_naive(aware: datetime) -> datetime:
    """A timezone-aware instant, as the naive UTC the columns store."""
    return aware.astimezone(timezone.utc).replace(tzinfo=None)


def _today_start() -> datetime:
    """Local midnight, expressed as naive UTC for querying.

    Was ``datetime(now.year, now.month, now.day)`` off ``utcnow()`` — midnight
    UTC, which in IST is 05:30 local. Everything between local midnight and
    05:30 therefore counted as YESTERDAY, so every night-shift vehicle landed
    on the wrong day and "today" silently rolled over mid-morning.

    Local midnight is what an operator means by "today", so that is what this
    returns — converted back to UTC because the column is naive UTC.
    """
    now_local = datetime.now(display_tz())
    midnight_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return _to_utc_naive(midnight_local)


@router.get("/stats", response_model=schemas.DashboardStats)
def stats(db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    start = _today_start()
    today_events = db.query(models.Event).filter(models.Event.detected_at >= start)

    total_today = today_events.count()
    unknown_today = today_events.filter(models.Event.status == models.VehicleStatus.unknown).count()
    entries_today = today_events.filter(models.Event.direction == models.CameraDirection.in_).count()
    exits_today = today_events.filter(models.Event.direction == models.CameraDirection.out_).count()

    blacklisted_count = db.query(models.Vehicle).filter(models.Vehicle.status == models.VehicleStatus.blacklist).count()

    total_cameras = db.query(models.Camera).count()
    active_cameras = sum(1 for c in db.query(models.Camera).all() if camera_manager.is_camera_online(c.id))

    # naive "inside" estimate: entries - exits today, floored at 0
    vehicles_inside = max(entries_today - exits_today, 0)

    return schemas.DashboardStats(
        total_vehicles_today=total_today,
        vehicles_inside=vehicles_inside,
        blacklisted_count=blacklisted_count,
        unknown_today=unknown_today,
        active_cameras=active_cameras,
        total_cameras=total_cameras,
        entries_today=entries_today,
        exits_today=exits_today,
    )


@router.get("/trend", response_model=List[schemas.TrendPoint])
def trend(db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    """Events per hour of the LOCAL day.

    Previously bucketed the raw UTC column and labelled the bars with the UTC
    hour, so vehicles passing at 17:00 local were charted at 11:00. Worse than
    a shift: because IST is a HALF-hour offset, local hour boundaries fall at
    :30 in UTC, so each local hour was split across two adjacent bars —

        local 14:00 -> buckets ['08:00', '09:00']
        local 15:00 -> buckets ['09:00', '10:00']

    — which distorts the shape of the chart, not just its labels. A genuine
    traffic peak was divided between two bars.

    Truncating AFTER converting to the display timezone puts each event in the
    hour it actually happened in, and the label is that same local hour. The
    frontend plots ``hour`` verbatim, so it needs no change.
    """
    tz = display_tz()
    start = _today_start()
    # The column is naive UTC, so tell Postgres that before converting.
    local_ts = func.timezone(str(tz), func.timezone("UTC", models.Event.detected_at))
    rows = (
        db.query(func.date_trunc("hour", local_ts).label("hour"), func.count(models.Event.id))
        .filter(models.Event.detected_at >= start)
        .group_by("hour")
        .order_by("hour")
        .all()
    )
    counts = {r[0].strftime("%H:00"): r[1] for r in rows}
    return [
        schemas.TrendPoint(hour=f"{h:02d}:00", count=counts.get(f"{h:02d}:00", 0))
        for h in range(24)
    ]


@router.get("/latest", response_model=List[schemas.EventOut])
def latest(db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    events = db.query(models.Event).order_by(models.Event.detected_at.desc()).limit(10).all()
    from .events import _serialize
    return [_serialize(e) for e in events]
