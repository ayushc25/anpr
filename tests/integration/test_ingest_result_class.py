"""Ingest of A / B / C events into a real (in-memory SQLite) schema.

The deferred flow's whole job is to hand the database a vehicle with no plate
string, and the migration that allows it is untestable without a database —
this is the cheapest honest approximation. SQLite is not Postgres, but the
models were written to run on both (see ``BigInteger().with_variant``).
"""
from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import backend.app.models  # noqa: F401  (registers every table on Base)
from backend.app.db.base import AlertSeverity, Base
from backend.app.events.event_builder import EventBuilder
from backend.app.events.multi_frame_validator import FinalPlate, RecognitionState
from backend.app.events.result_class import REASON_QUEUE_OVERFLOW, no_plate, review
from backend.app.events.track_state import TrackState
from backend.app.models.alert import Alert, AlertRule
from backend.app.models.event import Event
from backend.app.services.event_service import EventProcessor
from backend.app.storage.media_store import MediaStore


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def media(tmp_path):
    return MediaStore(tmp_path)


def state():
    s = TrackState(track_id=5, camera_id=1)
    s.best_vehicle_crop = np.full((60, 90, 3), 100, dtype=np.uint8)
    s.best_plate_crop = np.full((24, 70, 3), 200, dtype=np.uint8)
    return s


def plate(text="UP32AB1234", st=RecognitionState.CONFIRMED, conf=0.92):
    return FinalPlate(
        text=text, confidence=conf, support=1.0, read_count=5, distinct_variants=1,
        grammar_valid=True, state=st,
    )


def ingest(session, media, draft):
    event = EventProcessor(session, media).ingest(draft)
    session.commit()
    return event


def crops(n):
    return [np.full((24, 70, 3), 60 + i * 30, dtype=np.uint8) for i in range(n)]


def test_class_c_is_stored_with_no_plate(session, media):
    draft = EventBuilder(camera_id=1).build(state(), None, result=no_plate(observations=0))
    event = ingest(session, media, draft)
    assert event.plate_number is None
    assert (event.result_class, event.result_reason) == ("C", "no_usable_plate_observation")
    assert event.is_disputed is True
    assert event.vehicle_id is None and event.status.value == "unknown"


def test_class_b_overflow_stores_the_saved_crops(session, media):
    draft = EventBuilder(camera_id=1).build(
        state(), None, result=review(REASON_QUEUE_OVERFLOW), review_crops=crops(4)
    )
    event = ingest(session, media, draft)
    assert event.plate_number is None
    assert (event.result_class, event.result_reason) == ("B", REASON_QUEUE_OVERFLOW)
    assert len(event.review_crop_paths) == 4
    for relative in event.review_crop_paths:
        assert media.absolute(relative).exists()          # the crops are really on disk


def test_class_a_event_has_a_plate_and_no_review_crops(session, media):
    event = ingest(session, media, EventBuilder(camera_id=1).build(state(), plate(), review_crops=crops(3)))
    assert (event.plate_number, event.result_class) == ("UP32AB1234", "A")
    assert event.review_crop_paths is None
    assert event.is_disputed is False


def test_probable_plate_is_class_b_and_keeps_its_crops(session, media):
    draft = EventBuilder(camera_id=1).build(
        state(), plate(st=RecognitionState.PROBABLE, conf=0.7), review_crops=crops(2)
    )
    event = ingest(session, media, draft)
    assert (event.plate_number, event.result_class) == ("UP32AB1234", "B")
    assert len(event.review_crop_paths) == 2 and event.is_disputed is True


def test_verification_verdict_is_persisted(session, media):
    final = plate()
    final.verification = {"ran": True, "p_yes": 0.04, "downgraded": True, "reason": "verification_rejected"}
    final.state = RecognitionState.PROBABLE
    event = ingest(session, media, EventBuilder(camera_id=1).build(state(), final))
    assert event.result_class == "B"
    assert event.result_reason == "verification_rejected"
    assert event.result_detail["verification"]["p_yes"] == 0.04


def test_a_draft_spooled_before_the_migration_still_ingests(session, media):
    """Old drafts carry no result_class; ingest derives it from recognition_state."""
    legacy = EventBuilder(camera_id=1).build(state(), plate()).to_dict()
    for key in ("result_class", "result_reason", "verification", "review_crops_b64"):
        legacy.pop(key)
    from backend.app.events.event_builder import EventDraft

    assert ingest(session, media, EventDraft.from_dict(legacy)).result_class == "A"
    legacy["event_uid"] = "another-uid"
    legacy["recognition_state"] = "probable"
    assert ingest(session, media, EventDraft.from_dict(legacy)).result_class == "B"


def test_duplicate_ingest_is_still_idempotent(session, media):
    from backend.app.services.event_service import DuplicateEvent

    draft = EventBuilder(camera_id=1).build(state(), None, result=no_plate())
    ingest(session, media, draft)
    with pytest.raises(DuplicateEvent):
        EventProcessor(session, media).ingest(draft)


class TestAlerts:
    @pytest.fixture(autouse=True)
    def _rule(self, session):
        # LowConfidence fires for any disputed event — which every B and C is.
        session.add(AlertRule(
            code="low_confidence", name="low confidence", severity=AlertSeverity.info,
            is_enabled=True, channels=[], cooldown_seconds=0, params={},
        ))
        session.commit()

    def test_a_vehicle_with_no_plate_raises_no_plate_alert(self, session, media):
        """An alert reading 'Low-confidence read: None' is noise on top of the
        review-queue item the event already is."""
        ingest(session, media, EventBuilder(camera_id=1).build(state(), None, result=no_plate()))
        assert session.query(Alert).count() == 0

    def test_a_disputed_plate_still_alerts_as_before(self, session, media):
        draft = EventBuilder(camera_id=1).build(state(), plate(st=RecognitionState.PROBABLE, conf=0.6))
        ingest(session, media, draft)
        assert session.query(Alert).count() == 1


def test_existing_rows_without_a_class_are_valid(session):
    """The migration adds nullable columns; a pre-existing row must still load."""
    session.add(Event(event_uid="old", plate_number="DL8CAF5010"))
    session.commit()
    assert session.query(Event).one().result_class is None
