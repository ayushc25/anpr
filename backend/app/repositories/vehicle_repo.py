"""Registry lookups. The only module that queries vehicles."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload

from ..ai.plate_recognizer import postprocess
from ..db.base import VehicleStatus
from ..models.vehicle import PlateAlias, Resident, Vehicle

logger = logging.getLogger("anpr.repo.vehicle")


class VehicleRepository:
    def __init__(self, session: Session):
        self.session = session

    # -- matching ----------------------------------------------------------
    def match(self, plate: str, when: datetime | None = None) -> tuple[Optional[Vehicle], str]:
        """Resolve a recognized plate to a registered vehicle.

        Four tiers, most trusted first: an exact plate, a learned alias, a
        single-character confusion, then the nearest registered plate under a
        confusion-weighted distance. Returns the vehicle (or None) and the
        match method, which is recorded on the event so an operator can see
        whether a hit was exact or inferred.
        """
        normalized = postprocess.normalize(plate)
        if not normalized:
            return None, "none"

        vehicle = (
            self.session.query(Vehicle)
            .options(joinedload(Vehicle.resident))
            .filter(Vehicle.plate_number == normalized, Vehicle.is_active.is_(True))
            .first()
        )
        if vehicle:
            return vehicle, "exact"

        alias = (
            self.session.query(PlateAlias)
            .options(joinedload(PlateAlias.vehicle).joinedload(Vehicle.resident))
            .filter(PlateAlias.alias_plate == normalized)
            .first()
        )
        if alias and alias.vehicle and alias.vehicle.is_active:
            return alias.vehicle, "alias"

        confusable = self.find_unique_confusable(normalized)
        if confusable:
            return confusable, "confusable"

        nearest = self.find_nearest(normalized)
        if nearest:
            return nearest, "nearest"

        return None, "none"

    def find_unique_confusable(self, plate: str) -> Optional[Vehicle]:
        """Exactly one registered plate one OCR-confusion away.

        Never returns a blacklisted vehicle: flagging a car as blacklisted on
        an inferred match would be the most damaging possible false positive.
        """
        candidates = (
            self.session.query(Vehicle)
            .options(joinedload(Vehicle.resident))
            .filter(
                Vehicle.is_active.is_(True),
                func.length(Vehicle.plate_number) == len(plate),
                Vehicle.status != VehicleStatus.blacklist,
            )
            .all()
        )
        matches = [v for v in candidates if postprocess.confusable(plate, v.plate_number)]
        if len(matches) != 1:
            return None
        logger.info("registry: %s matched %s by confusion", plate, matches[0].plate_number)
        return matches[0]

    def find_nearest(self, plate: str) -> Optional[Vehicle]:
        """The registered vehicle a noisy read is closest to, when one stands
        clearly apart from the rest.

        This is the tier that makes the registry useful while the recognizer
        is still imperfect. ``find_unique_confusable`` only rescues a read
        that is the right LENGTH and wrong in exactly ONE character, which is
        a small fraction of real misreads — it cannot help with a dropped
        character or two confusions at once.

        Scoring against the registry is a much easier problem than reading the
        plate correctly in the first place, because the answer is known to be
        one of a few hundred strings. The safety does not come from the cost
        ceiling but from the margin: a read that fits two registered plates
        about equally well returns nothing.

        Blacklisted vehicles are excluded for the same reason they are
        excluded from confusable matching — an inferred blacklist hit is the
        most damaging false positive this system can produce, and a real one
        must stand on an exact match.
        """
        candidates = (
            self.session.query(Vehicle)
            .options(joinedload(Vehicle.resident))
            .filter(
                Vehicle.is_active.is_(True),
                Vehicle.status != VehicleStatus.blacklist,
            )
            .all()
        )
        if not candidates:
            return None
        by_plate = {v.plate_number: v for v in candidates}
        match = postprocess.best_registry_match(plate, by_plate.keys())
        if match is None:
            return None
        logger.info(
            "registry: %s matched %s (cost %.2f, margin %.2f)",
            plate, match.plate, match.cost, match.margin,
        )
        return by_plate[match.plate]

    def resolve_status(self, vehicle: Optional[Vehicle], when: datetime) -> VehicleStatus:
        """A registration that has expired is not a resident vehicle any more."""
        if vehicle is None:
            return VehicleStatus.unknown
        if not vehicle.is_valid_at(when):
            return VehicleStatus.unknown
        return vehicle.status or VehicleStatus.registered

    # -- snapshot for the workers -----------------------------------------
    def snapshot(self) -> dict[str, str]:
        rows = (
            self.session.query(Vehicle.plate_number, Vehicle.status)
            .filter(Vehicle.is_active.is_(True))
            .all()
        )
        snapshot = {plate: getattr(status, "value", status) for plate, status in rows}
        aliases = self.session.query(PlateAlias.alias_plate, Vehicle.status).join(
            Vehicle, PlateAlias.vehicle_id == Vehicle.id
        ).all()
        for alias, status in aliases:
            snapshot.setdefault(alias, getattr(status, "value", status))
        return snapshot

    # -- CRUD --------------------------------------------------------------
    def get(self, vehicle_id: int) -> Optional[Vehicle]:
        return self.session.get(Vehicle, vehicle_id)

    def by_plate(self, plate: str) -> Optional[Vehicle]:
        return (
            self.session.query(Vehicle)
            .filter(Vehicle.plate_number == postprocess.normalize(plate))
            .first()
        )

    def search(self, query: str = "", status: str | None = None, limit: int = 50, offset: int = 0):
        stmt = self.session.query(Vehicle).options(joinedload(Vehicle.resident))
        if query:
            like = f"%{query.upper()}%"
            stmt = stmt.outerjoin(Resident).filter(
                or_(
                    Vehicle.plate_number.like(like),
                    func.upper(Vehicle.owner_name).like(like),
                    func.upper(Vehicle.flat_number).like(like),
                    func.upper(Resident.name).like(like),
                    func.upper(Resident.flat_number).like(like),
                )
            )
        if status:
            stmt = stmt.filter(Vehicle.status == status)
        return stmt.order_by(Vehicle.plate_number).limit(limit).offset(offset).all()

    def upsert(self, plate_raw: str, **fields) -> tuple[Vehicle, bool]:
        """Create or update by normalized plate. Returns (vehicle, created).

        Used by the bulk importer, which must be idempotent: the client will
        send the same spreadsheet again with three rows changed.
        """
        normalized = postprocess.normalize(plate_raw)
        if not normalized:
            raise ValueError(f"'{plate_raw}' does not contain a usable plate number")

        vehicle = self.by_plate(normalized)
        created = vehicle is None
        if vehicle is None:
            vehicle = Vehicle(plate_number=normalized, plate_raw=plate_raw)
            self.session.add(vehicle)
        for key, value in fields.items():
            if value is not None and hasattr(vehicle, key):
                setattr(vehicle, key, value)
        return vehicle, created

    def add_alias(self, vehicle_id: int, alias: str, reason: str = "") -> Optional[PlateAlias]:
        normalized = postprocess.normalize(alias)
        if not normalized:
            return None
        existing = self.session.query(PlateAlias).filter(PlateAlias.alias_plate == normalized).first()
        if existing:
            return existing
        row = PlateAlias(vehicle_id=vehicle_id, alias_plate=normalized, reason=reason)
        self.session.add(row)
        return row
