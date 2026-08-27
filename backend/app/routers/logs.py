from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..database import get_db
from ..deps import require_permission

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("", response_model=List[schemas.ActivityLogOut])
def list_logs(db: Session = Depends(get_db), _: models.User = Depends(require_permission("logs")), limit: int = 200):
    cutoff = datetime.utcnow() - timedelta(days=90)
    logs = (
        db.query(models.ActivityLog)
        .filter(models.ActivityLog.created_at >= cutoff)
        .order_by(models.ActivityLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return logs
