from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, ForeignKey, Integer, String, Text
)
from sqlalchemy.orm import relationship

from ..db.base import Base, VehicleStatus


class Resident(Base):
    """Split out of ``vehicles`` because one flat routinely has two to four
    vehicles, and the client maintains their list per resident, not per plate."""

    __tablename__ = "residents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    flat_number = Column(String(32), default="", index=True)
    block = Column(String(32), default="")
    phone = Column(String(32), default="")
    email = Column(String(128), default="")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    vehicles = relationship("Vehicle", back_populates="resident")


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, index=True)
    #: Normalized: uppercase, no spaces or hyphens. This is the join key.
    plate_number = Column(String(32), unique=True, index=True, nullable=False)
    #: As the client typed it, for display and for import diffing.
    plate_raw = Column(String(48), default="")

    resident_id = Column(Integer, ForeignKey("residents.id"), nullable=True, index=True)
    #: Retained so the prototype's flat vehicle list keeps working during the
    #: migration; residents is the source of truth once populated.
    owner_name = Column(String(128), default="")
    flat_number = Column(String(32), default="")

    vehicle_type = Column(String(32), default="car")
    make_model = Column(String(64), default="")
    color = Column(String(32), default="")

    status = Column(Enum(VehicleStatus), default=VehicleStatus.registered, index=True)
    valid_from = Column(DateTime, nullable=True)
    valid_until = Column(DateTime, nullable=True)
    notes = Column(Text, default="")
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    resident = relationship("Resident", back_populates="vehicles")
    aliases = relationship("PlateAlias", back_populates="vehicle", cascade="all, delete-orphan")

    @property
    def display_owner(self) -> str:
        return self.resident.name if self.resident else self.owner_name

    @property
    def display_flat(self) -> str:
        return self.resident.flat_number if self.resident else self.flat_number

    def is_valid_at(self, when: datetime) -> bool:
        if self.valid_from and when < self.valid_from:
            return False
        if self.valid_until and when > self.valid_until:
            return False
        return bool(self.is_active)


class PlateAlias(Base):
    """A known misread mapped to the vehicle it belongs to.

    Every deployment accumulates these: one camera's angle turns a particular
    plate's 0 into a D, every single time. Mapping the alias is better than
    loosening the recognizer, which would cost accuracy everywhere else.
    """

    __tablename__ = "plate_aliases"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True)
    alias_plate = Column(String(32), unique=True, index=True, nullable=False)
    reason = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    vehicle = relationship("Vehicle", back_populates="aliases")
