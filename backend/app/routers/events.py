from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional

from .. import models, schemas
from ..database import get_db
from ..deps import get_current_user
from ..config import STORAGE_DIR

router = APIRouter(prefix="/events", tags=["events"])


def _serialize(e: models.Event) -> schemas.EventOut:
    return schemas.EventOut(
        id=e.id, plate_number=e.plate_number, vehicle_id=e.vehicle_id, camera_id=e.camera_id,
        vehicle_type=e.vehicle_type, vehicle_color=e.vehicle_color, plate_color=e.plate_color,
        direction=e.direction, status=e.status,
        # The columns were renamed in the Phase 1 schema (confidence ->
        # detect_confidence, ocr_confidence -> plate_confidence, image_path ->
        # vehicle_image_path). The RESPONSE field names are kept as-is so the
        # existing frontend is unaffected; only the mapping moved.
        confidence=e.detect_confidence or 0.0,
        ocr_confidence=e.plate_confidence or 0.0,
        image_path=e.vehicle_image_path, detected_at=e.detected_at,
        camera_name=e.camera.name if e.camera else None,
        # display_owner falls back to the linked resident: once residents
        # were split out of the vehicles table, owner_name is empty on any
        # vehicle registered to one, so reading it directly showed a blank
        # owner for every properly-registered vehicle.
        owner_name=e.vehicle.display_owner if e.vehicle else None,
    )


@router.get("", response_model=List[schemas.EventOut])
def list_events(
    db: Session = Depends(get_db),
    _: models.User = Depends(get_current_user),
    status: Optional[str] = None,
    camera_id: Optional[int] = None,
    plate: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
):
    query = db.query(models.Event)
    if status:
        query = query.filter(models.Event.status == status)
    if camera_id:
        query = query.filter(models.Event.camera_id == camera_id)
    if plate:
        query = query.filter(models.Event.plate_number.ilike(f"%{plate.upper()}%"))
    if date_from:
        query = query.filter(models.Event.detected_at >= date_from)
    if date_to:
        query = query.filter(models.Event.detected_at <= date_to)
    events = query.order_by(models.Event.detected_at.desc()).offset(offset).limit(limit).all()
    return [_serialize(e) for e in events]


@router.get("/{event_id}/image")
def get_event_image(event_id: int, db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    e = db.query(models.Event).get(event_id)
    if not e or not e.vehicle_image_path:
        raise HTTPException(status_code=404, detail="Image not found")
    path = STORAGE_DIR / e.vehicle_image_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing")
    return FileResponse(path)
