from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from ..db.base import Base


class Role(Base):
    """Permission sets edited once rather than per user.

    The prototype stores a permissions list on each user, which works until an
    admin needs to add one permission to every guard.
    """

    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(String(255), default="")
    permissions = Column(JSON, nullable=False, default=list)
    is_system = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="role")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    full_name = Column(String(128), nullable=False, default="")
    password_hash = Column(String(255), nullable=False)

    role_id = Column(Integer, ForeignKey("roles.id"), nullable=True)
    #: Kept alongside role_id during the migration; a per-user list still wins
    #: over the role's, so an existing user's access never silently changes.
    role_name = Column(String(64), nullable=False, default="")
    permissions = Column(JSON, nullable=False, default=list)

    is_active = Column(Boolean, default=True)
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    role = relationship("Role", back_populates="users")

    @property
    def effective_permissions(self) -> list[str]:
        if self.permissions:
            return list(self.permissions)
        return list(self.role.permissions) if self.role else []


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    username = Column(String(64), default="system")
    action = Column(String(128), nullable=False)
    entity_type = Column(String(64), default="")
    entity_id = Column(String(64), default="")
    details = Column(Text, default="")
    ip_address = Column(String(64), default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Setting(Base):
    """Runtime-tunable knobs an admin can change without a deploy."""

    __tablename__ = "settings"

    key = Column(String(96), primary_key=True)
    value = Column(JSON, nullable=True)
    description = Column(String(255), default="")
    updated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
