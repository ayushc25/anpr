from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..deps import require_permission

router = APIRouter(prefix="/logs", tags=["logs"])

#: Activity is kept for three months; older rows are not served.
RETENTION_DAYS = 90


@router.get("", response_model=schemas.ActivityLogPage)
def list_logs(
    db: Session = Depends(get_db),
    _: models.User = Depends(require_permission("logs")),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    """One page of activity, newest first.

    Paged rather than capped at a flat limit: the previous 200-row ceiling
    silently hid everything older on a busy site, which is the opposite of
    what an audit log is for. The count is over the whole retention window,
    so the UI can say how much there is rather than only whether more exists.
    """
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    query = db.query(models.ActivityLog).filter(models.ActivityLog.created_at >= cutoff)
    total = query.count()
    items = (
        query.order_by(models.ActivityLog.created_at.desc(), models.ActivityLog.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return schemas.ActivityLogPage(items=items, total=total, page=page, page_size=page_size)
