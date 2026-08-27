from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..database import get_db
from ..deps import get_current_user
from ..services import camera_manager

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _today_start():
    now = datetime.utcnow()
    return datetime(now.year, now.month, now.day)


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
    start = _today_start()
    rows = (
        db.query(func.date_trunc("hour", models.Event.detected_at).label("hour"), func.count(models.Event.id))
        .filter(models.Event.detected_at >= start)
        .group_by("hour")
        .order_by("hour")
        .all()
    )
    counts = {r[0].strftime("%H:00"): r[1] for r in rows}
    result = []
    for h in range(24):
        label = f"{h:02d}:00"
        result.append(schemas.TrendPoint(hour=label, count=counts.get(label, 0)))
    return result


@router.get("/latest", response_model=List[schemas.EventOut])
def latest(db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    events = db.query(models.Event).order_by(models.Event.detected_at.desc()).limit(10).all()
    from .events import _serialize
    return [_serialize(e) for e in events]
