from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..database import get_db
from ..deps import require_permission, get_current_user
from ..permissions import PERMISSIONS
from ..security import hash_password
from ..utils import log_activity

router = APIRouter(prefix="/users", tags=["users"])

require_users_permission = require_permission("users")


@router.get("/permissions", response_model=List[schemas.PermissionOut])
def list_permissions(_: models.User = Depends(require_users_permission)):
    return PERMISSIONS


@router.get("", response_model=List[schemas.UserOut])
def list_users(db: Session = Depends(get_db), _: models.User = Depends(require_users_permission)):
    return db.query(models.User).order_by(models.User.id).all()


@router.post("", response_model=schemas.UserOut)
def create_user(payload: schemas.UserCreate, db: Session = Depends(get_db), current: models.User = Depends(require_users_permission)):
    if db.query(models.User).filter(models.User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    user = models.User(
        username=payload.username,
        full_name=payload.full_name,
        role_name=payload.role_name,
        permissions=payload.permissions,
        is_active=payload.is_active,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log_activity(db, current.username, "create_user", f"Created user {user.username} ({user.role_name})", current.id)
    return user


@router.put("/{user_id}", response_model=schemas.UserOut)
def update_user(user_id: int, payload: schemas.UserUpdate, db: Session = Depends(get_db), current: models.User = Depends(require_users_permission)):
    user = db.query(models.User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.role_name is not None:
        user.role_name = payload.role_name
    if payload.permissions is not None:
        user.permissions = payload.permissions
    if payload.is_active is not None:
        user.is_active = payload.is_active
    if payload.password:
        user.password_hash = hash_password(payload.password)
    db.commit()
    db.refresh(user)
    log_activity(db, current.username, "update_user", f"Updated user {user.username}", current.id)
    return user


@router.delete("/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), current: models.User = Depends(require_users_permission)):
    user = db.query(models.User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    db.delete(user)
    db.commit()
    log_activity(db, current.username, "delete_user", f"Deleted user {user.username}", current.id)
    return {"ok": True}
