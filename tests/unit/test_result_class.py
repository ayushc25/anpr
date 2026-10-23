"""A / B / C: what the system does about a vehicle, derived in one place."""
from __future__ import annotations

import base64

import cv2
import numpy as np

from backend.app.events.event_builder import EventBuilder, EventDraft
from backend.app.events.multi_frame_validator import FinalPlate, RecognitionState
from backend.app.events.result_class import (
    REASON_CONFIRMED,
    REASON_NO_USABLE_OBSERVATION,
    REASON_QUEUE_OVERFLOW,
    ResultClass,
    class_for_state,
    classify,
    no_plate,
    review,
)
from backend.app.events.track_state import TrackState


def final(state):
    return FinalPlate(
        text="UP32AB1234", confidence=0.9, support=1.0, read_count=5, distinct_variants=1,
        grammar_valid=True, state=state,
    )


def test_state_to_class_mapping():
    assert class_for_state(RecognitionState.CONFIRMED) is ResultClass.AUTO_ACCEPTED
    assert class_for_state(RecognitionState.PROBABLE) is ResultClass.HUMAN_REVIEW
    assert class_for_state(RecognitionState.UNRESOLVED) is ResultClass.HUMAN_REVIEW


def test_no_state_maps_to_c():
    """C is the absence of a string; a state about a string cannot express it."""
    assert all(class_for_state(s) is not ResultClass.NO_PLATE for s in RecognitionState)


def test_classify_reports_values_and_reasons():
    a = classify(final(RecognitionState.CONFIRMED))
    assert (a.value, a.reason) == ("A", REASON_CONFIRMED)
    assert classify(final(RecognitionState.PROBABLE)).value == "B"
    assert classify(final(RecognitionState.UNRESOLVED)).value == "B"
    assert classify(None).value == "B"


def test_plateless_results():
    assert review(REASON_QUEUE_OVERFLOW, selected=5).value == "B"
    c = no_plate(observations=0)
    assert (c.value, c.reason) == ("C", REASON_NO_USABLE_OBSERVATION)


# ---- the event record -------------------------------------------------------
def _state():
    state = TrackState(track_id=7, camera_id=2)
    state.best_vehicle_crop = np.full((120, 200, 3), 90, dtype=np.uint8)
    state.best_plate_crop = np.full((40, 120, 3), 200, dtype=np.uint8)
    state.finalize_reason = "track_retired"
    return state


def _decode(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), dtype=np.uint8), cv2.IMREAD_COLOR)


def test_class_c_event_has_no_plate_and_says_why():
    draft = EventBuilder(camera_id=2).build(_state(), None, result=no_plate(observations=3))
    assert draft.plate_number is None            # not "", not "UNKNOWN"
    assert draft.plate_confidence == 0.0
    assert draft.result_class == "C"
    assert draft.result_reason == REASON_NO_USABLE_OBSERVATION
    assert draft.disputed is True
    assert draft.vehicle_image_b64 and draft.plate_image_b64   # the best crops are kept
    assert draft.media_source == "best_observation"


def test_class_b_overflow_saves_the_review_crops():
    crops = [np.full((40, 120, 3), v, dtype=np.uint8) for v in (60, 120, 180)]
    draft = EventBuilder(camera_id=2).build(
        _state(), None, result=review(REASON_QUEUE_OVERFLOW), review_crops=crops
    )
    assert draft.result_class == "B"
    assert draft.result_reason == REASON_QUEUE_OVERFLOW
    assert draft.plate_number is None
    assert len(draft.review_crops_b64) == 3
    assert abs(float(_decode(draft.review_crops_b64[2]).mean()) - 180) < 8


def test_class_a_event_carries_no_review_crops():
    crops = [np.zeros((40, 120, 3), dtype=np.uint8)]
    state = _state()
    draft = EventBuilder(camera_id=2).build(state, final(RecognitionState.CONFIRMED), review_crops=crops)
    assert draft.result_class == "A"
    assert draft.review_crops_b64 == []
    assert draft.plate_number == "UP32AB1234"


def test_class_b_with_a_plate_keeps_crops_and_is_disputed():
    crops = [np.zeros((40, 120, 3), dtype=np.uint8)]
    draft = EventBuilder(camera_id=2).build(_state(), final(RecognitionState.PROBABLE), review_crops=crops)
    assert draft.result_class == "B"
    assert draft.disputed is True
    assert len(draft.review_crops_b64) == 1


def test_draft_round_trips_through_json_for_the_spool():
    """A plate-less draft must survive the disk spool, which is JSON."""
    import json

    draft = EventBuilder(camera_id=2).build(_state(), None, result=no_plate())
    clone = EventDraft.from_dict(json.loads(json.dumps(draft.to_dict())))
    assert clone.plate_number is None and clone.result_class == "C"


def test_a_draft_spooled_before_the_field_existed_still_loads():
    legacy = EventBuilder(camera_id=2).build(_state(), final(RecognitionState.CONFIRMED)).to_dict()
    for key in ("result_class", "result_reason", "verification", "review_crops_b64"):
        legacy.pop(key)
    clone = EventDraft.from_dict(legacy)
    assert clone.result_class == ""           # ingest derives it from recognition_state
