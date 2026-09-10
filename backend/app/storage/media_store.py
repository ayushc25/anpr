"""Where event images live on disk.

Date-sharded so a directory never accumulates a year of JPEGs — the
prototype's flat ``storage/events/`` already holds thousands of files, and a
flat directory of a hundred thousand is slow to list and painful to prune.
Retention is a matter of deleting whole day directories.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("anpr.storage")

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_component(value: str, fallback: str = "unknown") -> str:
    cleaned = _UNSAFE.sub("_", value or "").strip("_")
    return cleaned or fallback


@dataclass(frozen=True)
class StoredMedia:
    vehicle_path: Optional[str]
    plate_path: Optional[str]

    def as_dict(self) -> dict:
        return {"vehicle_image_path": self.vehicle_path, "plate_image_path": self.plate_path}


class MediaStore:
    """Paths are stored relative to the media root so the database stays
    portable when the install directory moves."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _day_dir(self, when: datetime) -> Path:
        path = self.root / when.strftime("%Y") / when.strftime("%m") / when.strftime("%d")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        event_uid: str,
        camera_code: str,
        plate: str,
        when: datetime,
        vehicle_jpeg: Optional[bytes],
        plate_jpeg: Optional[bytes],
    ) -> StoredMedia:
        directory = self._day_dir(when)
        stem = f"{safe_component(camera_code, 'cam')}_{safe_component(plate, 'unknown')}_{event_uid[:8]}"

        vehicle_path = self._write(directory / f"{stem}.jpg", vehicle_jpeg)
        plate_path = self._write(directory / f"{stem}_plate.jpg", plate_jpeg)
        return StoredMedia(vehicle_path=vehicle_path, plate_path=plate_path)

    def _write(self, path: Path, payload: Optional[bytes]) -> Optional[str]:
        if not payload:
            return None
        try:
            path.write_bytes(payload)
        except OSError:
            logger.exception("could not write media to %s", path)
            return None
        return str(path.relative_to(self.root)).replace("\\", "/")

    def absolute(self, relative: Optional[str]) -> Optional[Path]:
        return (self.root / relative) if relative else None

    def url(self, relative: Optional[str], prefix: str = "/media") -> Optional[str]:
        return f"{prefix}/{relative}" if relative else None

    # -- retention ---------------------------------------------------------
    def purge_older_than(self, days: int, now: datetime | None = None) -> int:
        """Delete whole day directories past the retention horizon.

        Returns the number of files removed. Whole-directory deletion is why
        the date sharding exists: pruning by mtime across a flat directory of
        200k files takes minutes on the spinning disk an edge box often has.
        """
        if days <= 0:
            return 0
        now = now or datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=days)).date()
        removed = 0
        for day_dir in sorted(self.root.glob("*/*/*")):
            if not day_dir.is_dir():
                continue
            try:
                parts = day_dir.parts[-3:]
                day = datetime(int(parts[0]), int(parts[1]), int(parts[2])).date()
            except (ValueError, IndexError):
                continue
            if day >= cutoff:
                continue
            for file in day_dir.iterdir():
                try:
                    file.unlink()
                    removed += 1
                except OSError:
                    logger.warning("could not delete %s", file)
            try:
                day_dir.rmdir()
            except OSError:
                pass
        if removed:
            logger.info("media retention: removed %d files older than %d days", removed, days)
        return removed

    def disk_usage_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*.jpg") if f.is_file())
