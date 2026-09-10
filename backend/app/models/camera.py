from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, String
)
from sqlalchemy.orm import relationship

from ..db.base import Base, CameraDirection, ZoneKind


class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    description = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    cameras = relationship("Camera", back_populates="location")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(32), unique=True, index=True, nullable=True)
    name = Column(String(128), nullable=False)

    rtsp_url = Column(String(512), nullable=False)
    #: The sub-stream is what the AI actually decodes. Leaving it empty falls
    #: back to the main stream, at roughly 2.5x the decode cost.
    rtsp_url_sub = Column(String(512), nullable=True)
    #: Pull the evidence snapshot from the main stream even though detection
    #: ran on the sub-stream.
    snapshot_from_main = Column(Boolean, default=True)

    location_id = Column(Integer, ForeignKey("locations.id"), nullable=True)
    direction = Column(Enum(CameraDirection), default=CameraDirection.both)

    processing_fps = Column(Float, default=6.0)
    detect_interval = Column(Integer, nullable=True)
    model_profile = Column(String(64), nullable=True)

    is_active = Column(Boolean, default=True)
    is_online = Column(Boolean, default=False)
    last_seen_at = Column(DateTime, nullable=True)
    health = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    location = relationship("Location", back_populates="cameras")
    zones = relationship("CameraZone", back_populates="camera", cascade="all, delete-orphan")


class CameraZone(Base):
    """ROI polygons and virtual lines, in normalized 0..1 coordinates.

    Versioned rather than mutated: an operator who redraws the gate line next
    month should not make last month's events unexplainable. Superseding a
    zone sets ``is_active`` false and inserts a new row.
    """

    __tablename__ = "camera_zones"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    kind = Column(Enum(ZoneKind), nullable=False, default=ZoneKind.roi)
    name = Column(String(64), default="")
    geometry = Column(JSON, nullable=False, default=list)
    #: For a line: which side of A->B counts as an entry.
    direction_hint = Column(String(8), default="in")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    camera = relationship("Camera", back_populates="zones")
