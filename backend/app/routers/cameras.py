from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..database import get_db
from ..deps import require_permission, get_current_user
from ..utils import log_activity
from ..services import camera_manager

router = APIRouter(prefix="/cameras", tags=["cameras"])

require_cameras_permission = require_permission("cameras")


@router.get("", response_model=List[schemas.CameraOut])
def list_cameras(db: Session = Depends(get_db), _: models.User = Depends(get_current_user)):
    cams = db.query(models.Camera).order_by(models.Camera.id).all()
    for c in cams:
        c.is_online = camera_manager.is_camera_online(c.id)
    return cams


@router.post("", response_model=schemas.CameraOut)
def create_camera(payload: schemas.CameraCreate, db: Session = Depends(get_db), current: models.User = Depends(require_cameras_permission)):
    cam = models.Camera(**payload.model_dump())
    db.add(cam)
    db.commit()
    db.refresh(cam)
    camera_manager.start_camera(cam.id, cam.rtsp_url)
    log_activity(db, current.username, "create_camera", f"Added camera {cam.name}", current.id)
    return cam


@router.put("/{camera_id}", response_model=schemas.CameraOut)
def update_camera(camera_id: int, payload: schemas.CameraUpdate, db: Session = Depends(get_db), current: models.User = Depends(require_cameras_permission)):
    cam = db.query(models.Camera).get(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(cam, k, v)
    db.commit()
    db.refresh(cam)
    if "rtsp_url" in data or "is_active" in data:
        camera_manager.stop_camera(cam.id)
        if cam.is_active:
            camera_manager.start_camera(cam.id, cam.rtsp_url)
    log_activity(db, current.username, "update_camera", f"Updated camera {cam.name}", current.id)
    return cam


@router.delete("/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db), current: models.User = Depends(require_cameras_permission)):
    cam = db.query(models.Camera).get(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera_manager.stop_camera(cam.id)
    db.delete(cam)
    db.commit()
    log_activity(db, current.username, "delete_camera", f"Deleted camera {cam.name}", current.id)
    return {"ok": True}


@router.get("/{camera_id}/stream")
def stream_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = db.query(models.Camera).get(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        camera_manager.mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
