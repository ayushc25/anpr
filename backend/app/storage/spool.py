"""On-disk queue for events that could not be submitted.

A gate system must not lose an event because Postgres was restarting or the
API was being upgraded. Drafts are written as JSON files and drained in
timestamp order once submission succeeds again.

Deliberately simple: one file per event, atomic rename on write, delete on
success. No database, no broker — this has to work when those are the things
that are down.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

logger = logging.getLogger("anpr.storage.spool")


class EventSpool:
    def __init__(self, directory: Path, max_files: int = 20000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*.json"))

    def add(self, payload: dict) -> Optional[Path]:
        uid = payload.get("event_uid", "unknown")
        path = self.directory / f"{time.time():.6f}_{uid}.json"
        temp = path.with_suffix(".tmp")
        try:
            temp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temp, path)
        except OSError:
            logger.exception("could not spool event %s", uid)
            return None

        self._enforce_cap()
        logger.warning("spooled event %s (%d pending)", uid, len(self))
        return path

    def _enforce_cap(self) -> None:
        """A spool that grows without bound fills the disk and takes the whole
        system down — a worse failure than losing the oldest events."""
        files = sorted(self.directory.glob("*.json"))
        excess = len(files) - self.max_files
        for path in files[:excess]:
            try:
                path.unlink()
                logger.error("spool full: dropped oldest event %s", path.name)
            except OSError:
                pass

    def pending(self) -> Iterator[tuple[Path, dict]]:
        """Oldest first — the filename's timestamp prefix sorts correctly."""
        for path in sorted(self.directory.glob("*.json")):
            try:
                yield path, json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("discarding unreadable spool file %s", path.name)
                path.unlink(missing_ok=True)

    def drain(self, submit: Callable[[dict], bool], limit: int = 25) -> int:
        """Retry spooled events. Stops at the first failure so ordering is
        preserved and a still-down API is not hammered."""
        sent = 0
        for path, payload in self.pending():
            if sent >= limit:
                break
            try:
                ok = submit(payload)
            except Exception:
                logger.debug("spool drain failed for %s", path.name, exc_info=True)
                break
            if not ok:
                break
            path.unlink(missing_ok=True)
            sent += 1
        if sent:
            logger.info("drained %d spooled events (%d remaining)", sent, len(self))
        return sent
