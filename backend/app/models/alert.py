from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON, BigInteger, Boolean, Column, DateTime, Enum, ForeignKey, Integer, String, Text
)
from sqlalchemy.orm import relationship

from ..db.base import Base, AlertSeverity, AlertStatus, NotificationStatus


class AlertRule(Base):
    """An operator-tunable rule.

    Parameters live in JSON rather than columns so a new rule type does not
    need a migration; ``code`` binds the row to a Rule class in
    events/rules_engine.py.
    """

    __tablename__ = "alert_rules"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(64), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    description = Column(String(255), default="")
    params = Column(JSON, default=dict)
    severity = Column(Enum(AlertSeverity), default=AlertSeverity.warning)
    is_enabled = Column(Boolean, default=True)
    #: Channel names, e.g. ["smtp", "webhook"]. Empty means in-app only.
    channels = Column(JSON, default=list)
    #: Suppress repeats for the same vehicle within this window; without it a
    #: blacklisted car idling in the ROI would page the guard every few seconds.
    cooldown_seconds = Column(Integer, default=300)
    #: Restrict to specific cameras; empty means all.
    camera_ids = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    alerts = relationship("Alert", back_populates="rule")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, index=True)
    rule_id = Column(Integer, ForeignKey("alert_rules.id"), nullable=True, index=True)
    event_id = Column(
        BigInteger().with_variant(Integer, "sqlite"), ForeignKey("events.id"), nullable=True, index=True
    )

    code = Column(String(64), default="")
    severity = Column(Enum(AlertSeverity), default=AlertSeverity.warning, index=True)
    title = Column(String(160), nullable=False)
    message = Column(Text, default="")
    context = Column(JSON, default=dict)

    status = Column(Enum(AlertStatus), default=AlertStatus.new, index=True)
    acknowledged_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    rule = relationship("AlertRule", back_populates="alerts")
    event = relationship("Event", back_populates="alerts")
    notifications = relationship("Notification", back_populates="alert", cascade="all, delete-orphan")


class Notification(Base):
    """The outbox.

    Rows are written inside the ingest transaction and delivered afterwards by
    a background drain. That ordering is the whole point: an SMTP server that
    is down must never delay or roll back an event.
    """

    __tablename__ = "notifications"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    alert_id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("alerts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    channel = Column(String(32), nullable=False)
    target = Column(String(255), default="")
    payload = Column(JSON, default=dict)
    status = Column(Enum(NotificationStatus), default=NotificationStatus.pending, index=True)
    attempts = Column(Integer, default=0)
    last_error = Column(Text, default="")
    sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    alert = relationship("Alert", back_populates="notifications")
