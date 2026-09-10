"""ORM models, one module per aggregate.

This package replaces the single ``models.py`` of the prototype and re-exports
every name it exported, so the existing routers (``from .. import models``;
``models.Camera``) keep working unchanged while they are migrated to the
repository layer one at a time.
"""
from ..db.base import (
    AlertSeverity,
    AlertStatus,
    Base,
    CameraDirection,
    EventDirection,
    NotificationStatus,
    VehicleStatus,
    ZoneKind,
)
from .alert import Alert, AlertRule, Notification
from .camera import Camera, CameraZone, Location
from .event import Event, EventRead
from .user import ActivityLog, Role, Setting, User
from .vehicle import PlateAlias, Resident, Vehicle

__all__ = [
    "Base",
    "VehicleStatus",
    "CameraDirection",
    "EventDirection",
    "AlertStatus",
    "AlertSeverity",
    "NotificationStatus",
    "ZoneKind",
    "User",
    "Role",
    "ActivityLog",
    "Setting",
    "Location",
    "Camera",
    "CameraZone",
    "Resident",
    "Vehicle",
    "PlateAlias",
    "Event",
    "EventRead",
    "AlertRule",
    "Alert",
    "Notification",
]
