from sqlalchemy.orm import Session

from . import models


def log_activity(db: Session, username: str, action: str, details: str = "", user_id: int | None = None):
    entry = models.ActivityLog(username=username, action=action, details=details, user_id=user_id)
    db.add(entry)
    db.commit()
