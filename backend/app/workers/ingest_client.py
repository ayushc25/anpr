"""Worker -> API event submission, with a disk spool behind it.

Two modes, chosen by config and hidden behind the same interface:

  ``http``   POST to /api/v1/internal/events. The right choice when workers
             run in separate containers or on separate machines.
  ``direct`` call the event service in-process. Fewer moving parts on a
             single edge box, at the cost of the worker holding a short-lived
             database session per event.

Either way the pipeline loop never blocks: submission is attempted once with a
short timeout, and anything that fails goes to the spool.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from ..events.event_builder import EventDraft
from ..storage.spool import EventSpool

logger = logging.getLogger("anpr.worker.ingest")


class IngestClient:
    """Base: spool handling and the drain schedule."""

    def __init__(self, spool_dir: Path, drain_interval: float = 10.0):
        self.spool = EventSpool(spool_dir)
        self.drain_interval = drain_interval
        self._last_drain = 0.0
        self._lock = threading.Lock()

    @property
    def spool_depth(self) -> int:
        return len(self.spool)

    def submit(self, draft: EventDraft) -> bool:
        payload = draft.to_dict()
        if self._send(payload):
            return True
        self.spool.add(payload)
        return False

    def drain_spool(self, force: bool = False) -> int:
        """Called from the worker loop on frames where there is nothing to
        process — idle time is exactly when retrying is free."""
        now = time.monotonic()
        if not force and (now - self._last_drain) < self.drain_interval:
            return 0
        self._last_drain = now
        if not self.spool_depth:
            return 0
        return self.spool.drain(self._send)

    def _send(self, payload: dict) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        self.drain_spool(force=True)


class HttpIngestClient(IngestClient):
    def __init__(
        self,
        base_url: str,
        token: str,
        spool_dir: Path,
        timeout: float = 4.0,
        drain_interval: float = 10.0,
    ):
        super().__init__(spool_dir, drain_interval)
        self.url = base_url.rstrip("/") + "/api/v1/internal/events"
        self.token = token
        self.timeout = timeout
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"X-Worker-Token": self.token})
        return self._session

    def _send(self, payload: dict) -> bool:
        try:
            response = self._get_session().post(self.url, json=payload, timeout=self.timeout)
        except Exception as exc:
            logger.warning("ingest POST failed: %s", exc)
            return False
        if response.status_code in (200, 201, 409):
            # 409 means the API already has this event_uid. The event is
            # safely stored, so the draft must be dropped rather than retried
            # forever — that is the point of the idempotency key.
            return True
        logger.warning("ingest rejected: %s %s", response.status_code, response.text[:200])
        return False

    def close(self) -> None:
        super().close()
        if self._session is not None:
            self._session.close()


class DirectIngestClient(IngestClient):
    """In-process ingest for single-box deployments."""

    def __init__(self, spool_dir: Path, drain_interval: float = 10.0):
        super().__init__(spool_dir, drain_interval)

    def _send(self, payload: dict) -> bool:
        try:
            from ..db.session import SessionLocal
            from ..services.event_service import EventProcessor

            session = SessionLocal()
            try:
                EventProcessor(session).ingest(EventDraft.from_dict(payload))
                session.commit()
                return True
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        except Exception:
            logger.warning("direct ingest failed", exc_info=True)
            return False


class NullIngestClient(IngestClient):
    """Logs and discards. Used by scripts/eval_pipeline.py and by
    ``anpr worker --dry-run`` when tuning a camera on site."""

    def __init__(self, spool_dir: Optional[Path] = None):
        super().__init__(spool_dir or Path(".cache/spool-dryrun"))
        self.events: list[dict] = []

    def _send(self, payload: dict) -> bool:
        self.events.append(payload)
        logger.info(
            "[dry-run] %s %s conf=%.2f reads=%d",
            payload["plate_number"], payload["direction"],
            payload["plate_confidence"], payload["read_count"],
        )
        return True


def build_ingest_client(mode: str, base_url: str, token: str, spool_dir: Path) -> IngestClient:
    if mode == "dry-run":
        return NullIngestClient(spool_dir)
    if mode == "direct":
        return DirectIngestClient(spool_dir)
    return HttpIngestClient(base_url, token, spool_dir)
