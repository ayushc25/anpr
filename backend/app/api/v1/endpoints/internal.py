"""Worker ingest.

Not part of the public API: authenticated with a shared secret rather than a
user JWT, because the caller is a camera worker process, not a person.

Idempotent on ``event_uid``. A worker that times out waiting for a response
retries the same draft, and a 409 tells it the event is already stored so it
can drop the spooled copy instead of retrying forever.
"""
from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy.orm import Session

from ....core.config import get_settings
from ....db.session import get_db
from ....events.event_builder import EventDraft
from ....services.event_service import DuplicateEvent, EventProcessor

logger = logging.getLogger("anpr.api.internal")

router = APIRouter(prefix="/internal", tags=["internal"])


def verify_worker(x_worker_token: str = Header(default="")) -> None:
    expected = get_settings().worker_token
    if not expected:
        # No token configured: only sensible on a single box where the API
        # is not reachable off-host. Log it loudly rather than failing closed
        # and silently breaking every camera on an upgrade.
        logger.warning("WORKER_TOKEN is not set; /internal is unauthenticated")
        return
    # Constant-time compare: this endpoint accepts writes to the event log.
    if not hmac.compare_digest(x_worker_token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid worker token")


@router.post("/events", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_worker)])
def ingest_event(payload: dict, response: Response, db: Session = Depends(get_db)):
    try:
        draft = EventDraft.from_dict(payload)
    except (TypeError, KeyError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"malformed event draft: {exc}")

    processor = EventProcessor(db)
    try:
        event = processor.ingest(draft)
    except DuplicateEvent as duplicate:
        # Already stored. 409 is success from the worker's point of view: it
        # drops the draft rather than retrying.
        response.status_code = status.HTTP_409_CONFLICT
        return {"status": "duplicate", "event_id": duplicate.event.id}
    except Exception:
        db.rollback()
        logger.exception("ingest failed for %s", payload.get("event_uid"))
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "ingest failed")

    db.commit()
    return {
        "status": "created",
        "event_id": event.id,
        "plate_number": event.plate_number,
        "status_label": event.status.value,
    }


@router.get("/registry-snapshot", dependencies=[Depends(verify_worker)])
def registry_snapshot(db: Session = Depends(get_db)):
    """Plate -> status, for a worker's in-process confusable lookup.

    Workers refresh this periodically instead of querying per event: the
    pipeline loop must never wait on the database.
    """
    from ....repositories.vehicle_repo import VehicleRepository

    return {"plates": VehicleRepository(db).snapshot()}
