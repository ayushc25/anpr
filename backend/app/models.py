import enum
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text, Enum, JSON
)
from sqlalchemy.orm import relationship

from .database import Base


class VehicleStatus(str, enum.Enum):
    registered = "registered"
    whitelist = "whitelist"
    blacklist = "blacklist"
    unknown = "unknown"


class CameraDirection(str, enum.Enum):
    in_ = "in"
    out_ = "out"
    both = "both"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    full_name = Column(String(128), nullable=False, default="")
    password_hash = Column(String(255), nullable=False)
    role_name = Column(String(64), nullable=False, default="")
    permissions = Column(JSON, nullable=False, default=list)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


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
    name = Column(String(128), nullable=False)
    rtsp_url = Column(String(512), nullable=False)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=True)
    direction = Column(Enum(CameraDirection), default=CameraDirection.both)
    is_active = Column(Boolean, default=True)
    last_seen_at = Column(DateTime, nullable=True)
    is_online = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    location = relationship("Location", back_populates="cameras")


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String(32), unique=True, index=True, nullable=False)
    owner_name = Column(String(128), default="")
    flat_number = Column(String(32), default="")
    vehicle_type = Column(String(32), default="car")
    status = Column(Enum(VehicleStatus), default=VehicleStatus.registered)
    valid_until = Column(DateTime, nullable=True)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String(32), index=True, nullable=False)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    vehicle_type = Column(String(32), default="car")
    vehicle_color = Column(String(32), nullable=True)
    plate_color = Column(String(32), nullable=True)
    direction = Column(Enum(CameraDirection), default=CameraDirection.both)
    status = Column(Enum(VehicleStatus), default=VehicleStatus.unknown)
    confidence = Column(Float, default=0.0)
    ocr_confidence = Column(Float, default=0.0)
    image_path = Column(String(512), nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)

    vehicle = relationship("Vehicle")
    camera = relationship("Camera")


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    username = Column(String(64), default="system")
    action = Column(String(128), nullable=False)
    details = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
