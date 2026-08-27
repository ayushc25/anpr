from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from .. import models, schemas
from ..database import get_db
from ..deps import require_permission, get_current_user
from ..utils import log_activity
from .events import _serialize as _serialize_event

router = APIRouter(prefix="/vehicles", tags=["vehicles"])

require_lists_permission = require_permission("lists")


@router.get("", response_model=List[schemas.VehicleOut])
def list_vehicles(
    db: Session = Depends(get_db),
    _: models.User = Depends(get_current_user),
    status: Optional[str] = None,
    q: Optional[str] = Query(None, description="search plate or owner"),
):
    query = db.query(models.Vehicle)
    if status:
        query = query.filter(models.Vehicle.status == status)
    if q:
        like = f"%{q.upper()}%"
        query = query.filter(models.Vehicle.plate_number.ilike(like) | models.Vehicle.owner_name.ilike(like))
    return query.order_by(models.Vehicle.id.desc()).all()


@router.get("/{vehicle_id}", response_model=schemas.VehicleOut)
def get_vehicle(vehicle_id: int, db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    v = db.query(models.Vehicle).get(vehicle_id)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return v


@router.get("/{vehicle_id}/history", response_model=List[schemas.EventOut])
def vehicle_history(vehicle_id: int, db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    v = db.query(models.Vehicle).get(vehicle_id)
    if not v:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    events = db.query(models.Event).filter(models.Event.vehicle_id == vehicle_id).order_by(models.Event.detected_at.desc()).all()
    return [_serialize_event(e) for e in events]


@router.post("", response_model=schemas.VehicleOut)
def create_vehicle(payload: schemas.VehicleCreate, db: Session = Depends(get_db), current: models.User = Depends(require_lists_permission)):
    plate = payload.plate_number.upper().replace(" ", "")
    if db.query(models.Vehicle).filter(models.Vehicle.plate_number == plate).first():
        raise HTTPException(status_code=400, detail="Vehicle with this plate already exists")
    data = payload.model_dump()
    data["plate_number"] = plate
    vehicle = models.Vehicle(**data)
    db.add(vehicle)
    db.commit()
    db.refresh(vehicle)
    log_activity(db, current.username, "create_vehicle", f"Added vehicle {vehicle.plate_number} ({vehicle.status.value})", current.id)
    return vehicle


@router.put("/{vehicle_id}", response_model=schemas.VehicleOut)
def update_vehicle(vehicle_id: int, payload: schemas.VehicleUpdate, db: Session = Depends(get_db), current: models.User = Depends(require_lists_permission)):
    vehicle = db.query(models.Vehicle).get(vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(vehicle, k, v)
    db.commit()
    db.refresh(vehicle)
    log_activity(db, current.username, "update_vehicle", f"Updated vehicle {vehicle.plate_number}", current.id)
    return vehicle


@router.delete("/{vehicle_id}")
def delete_vehicle(vehicle_id: int, db: Session = Depends(get_db), current: models.User = Depends(require_lists_permission)):
    vehicle = db.query(models.Vehicle).get(vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    db.delete(vehicle)
    db.commit()
    log_activity(db, current.username, "delete_vehicle", f"Deleted vehicle {vehicle.plate_number}", current.id)
    return {"ok": True}
