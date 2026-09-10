"""The declarative base and the enum types shared across models."""
from __future__ import annotations

import enum

from sqlalchemy.orm import declarative_base

Base = declarative_base()


class VehicleStatus(str, enum.Enum):
    registered = "registered"
    whitelist = "whitelist"
    blacklist = "blacklist"
    unknown = "unknown"
    visitor = "visitor"


class CameraDirection(str, enum.Enum):
    in_ = "in"
    out_ = "out"
    both = "both"


class EventDirection(str, enum.Enum):
    """What an event actually recorded.

    Distinct from CameraDirection on purpose: a camera may be configured as
    ``both``, but an individual event is an entry, an exit, or an honest
    ``unknown``. The prototype copied the camera value straight onto the
    event, which made entry/exit totals meaningless for shared gates.
    """

    in_ = "in"
    out_ = "out"
    unknown = "unknown"


class AlertStatus(str, enum.Enum):
    new = "new"
    acknowledged = "acknowledged"
    resolved = "resolved"


class AlertSeverity(str, enum.Enum):
    info = "info"
    warning = "warning"
    critical = "critical"


class NotificationStatus(str, enum.Enum):
    pending = "pending"
    sent = "sent"
    failed = "failed"


class ZoneKind(str, enum.Enum):
    roi = "roi"
    line = "line"
    mask = "mask"
