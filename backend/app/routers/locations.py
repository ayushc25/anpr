from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..database import get_db
from ..deps import require_permission, get_current_user
from ..utils import log_activity

router = APIRouter(prefix="/locations", tags=["locations"])

require_locations_permission = require_permission("locations")


@router.get("", response_model=List[schemas.LocationOut])
def list_locations(db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    return db.query(models.Location).order_by(models.Location.id).all()


@router.post("", response_model=schemas.LocationOut)
def create_location(payload: schemas.LocationCreate, db: Session = Depends(get_db), current: models.User = Depends(require_locations_permission)):
    loc = models.Location(**payload.model_dump())
    db.add(loc)
    db.commit()
    db.refresh(loc)
    log_activity(db, current.username, "create_location", f"Created location {loc.name}", current.id)
    return loc


@router.delete("/{location_id}")
def delete_location(location_id: int, db: Session = Depends(get_db), current: models.User = Depends(require_locations_permission)):
    loc = db.query(models.Location).get(location_id)
    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")
    db.delete(loc)
    db.commit()
    log_activity(db, current.username, "delete_location", f"Deleted location {loc.name}", current.id)
    return {"ok": True}
