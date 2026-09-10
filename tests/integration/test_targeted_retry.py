"""Targeted retry: spend the last calls on BANKED crops, not future frames.

Item 11 of the brief. The distinction that matters is what the extra calls are
spent on. A finalized track has no next frame — the vehicle has crossed the
line or the tracker retired it — so extending a future-frame budget is
worthless exactly when the plate is still in doubt. The retained observation
set, however, holds diverse crops the recognizer has never seen.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.app.ai.quality import crop_hash
from backend.app.ai.types import PlateObservation, PlateRead
from backend.app.ai.vehicle_tracker.bytetrack import ByteTracker
from backend.app.video.frame_processor import FrameProcessor, PipelineConfig
from backend.app.video.ocr_scheduler import OcrPolicy
from backend.app.video.roi import RoiPolygon


def crop(seed: int, width: int = 130, height: int = 34) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(40, 220, (height, width, 3), dtype=np.uint8)


class ScriptedRecognizer:
    """Returns the next scripted text on each call, so a test can make the
    retry produce a different answer from the frames already read."""

    name = "scripted"
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    expects_grayscale = True

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def recognize(self, image):
        text = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if text is None:
            return None
        return PlateRead(
            text=text, confidence=0.85,
            per_char_confidence=[0.85] * len(text), raw_text=text,
        )

    def close(self):
        pass


def processor(script, cfg=None):
    return FrameProcessor(
        detector=None,
        tracker=ByteTracker(),
        plate_detector=None,
        recognizer=ScriptedRecognizer(script),
        roi=RoiPolygon.from_config([]),
        cfg=cfg or PipelineConfig(),
        camera_id=1,
    )


def contested_state(proc, unread_crops=2):
    """A track whose position 5 is genuinely contested, with banked unread
    crops available for a retry."""
    from backend.app.events.track_state import TrackState, WeightedRead

    state = TrackState(track_id=1, camera_id=1)
    per_char = [0.99] * 5 + [0.50] + [0.99] * 4
    for i, text in enumerate(("UP32AB1234", "UP32AX1234", "UP32AB1234", "UP32AB1234")):
        state.reads.append(
            WeightedRead(
                text=text, raw_text=text, rec_confidence=0.95,
                per_char_confidence=list(per_char), plate_det_confidence=0.95,
                quality=0.95, grammar=1.0, frame_ts=1000.0 + i,
                frame_idx=i * 5, appearance=crop_hash.crop_hash(crop(i)),
            )
        )
    for j in range(unread_crops):
        c = crop(500 + j)
        state.note_observation(
            PlateObservation(
                frame_idx=100 + j * 5, ts=1010.0 + j,
                width=c.shape[1], height=c.shape[0],
                quality=0.85, det_confidence=0.9, crop=c,
                appearance=crop_hash.crop_hash(c),
            ),
            keep=6,
        )
    return state


class TestRetryUsesBankedCrops:
    def test_it_spends_calls_on_unread_observations(self):
        proc = processor(["UP32AB1234"])
        state = contested_state(proc, unread_crops=2)
        spent = proc.resolve_pending(state, now=2000.0)
        assert spent == 2
        assert proc.recognizer.calls == 2
        assert all(o.ocr_ran for o in state.plate_observations)

    def test_the_extra_reads_join_the_ballot(self):
        proc = processor(["UP32AB1234"])
        state = contested_state(proc)
        before = len(state.reads)
        proc.resolve_pending(state, now=2000.0)
        assert len(state.reads) == before + 2

    def test_a_retry_can_settle_a_contested_position(self):
        """The point of the whole mechanism: two extra reads off different
        crops break a near-tie at position 5."""
        from backend.app.events.char_fusion import fuse

        proc = processor(["UP32AB1234", "UP32AB1234"])
        state = contested_state(proc)
        contested_before = fuse(state.reads).position(5)
        proc.resolve_pending(state, now=2000.0)
        contested_after = fuse(state.reads).position(5)
        assert contested_after.margin > contested_before.margin

    def test_it_records_which_positions_it_was_bought_for(self):
        proc = processor(["UP32AB1234"])
        state = contested_state(proc)
        proc.resolve_pending(state, now=2000.0)
        assert state.retry_positions == [5]

    def test_a_read_crop_is_never_retried(self):
        """A deterministic model on a crop it has already seen returns the
        same characters, so retrying one is a wasted call."""
        proc = processor(["UP32AB1234"])
        state = contested_state(proc, unread_crops=2)
        for o in state.plate_observations:
            o.ocr_ran = True
        assert proc.resolve_pending(state, now=2000.0) == 0
        assert proc.recognizer.calls == 0


class TestRetryIsBounded:
    def test_it_never_exceeds_the_retry_bonus(self):
        cfg = PipelineConfig(ocr=OcrPolicy(retry_bonus=1))
        proc = processor(["UP32AB1234"], cfg)
        state = contested_state(proc, unread_crops=4)
        assert proc.resolve_pending(state, now=2000.0) == 1

    def test_it_runs_at_most_once_per_track(self):
        proc = processor(["UP32AB1234"])
        state = contested_state(proc)
        first = proc.resolve_pending(state, now=2000.0)
        second = proc.resolve_pending(state, now=2001.0)
        assert first == 2
        assert second == 0, "a retry cannot grant itself another"

    def test_disabled_when_retry_bonus_is_zero(self):
        cfg = PipelineConfig(ocr=OcrPolicy(retry_bonus=0))
        proc = processor(["UP32AB1234"], cfg)
        assert proc.resolve_pending(contested_state(proc), now=2000.0) == 0

    def test_budget_accounting_includes_retry_calls(self):
        proc = processor(["UP32AB1234"])
        state = contested_state(proc)
        before = state.ocr_calls
        proc.resolve_pending(state, now=2000.0)
        assert state.ocr_calls == before + 2


class TestRetryIsSelective:
    def test_a_settled_plate_gets_no_retry(self):
        from backend.app.events.track_state import TrackState, WeightedRead

        proc = processor(["UP32AB1234"])
        state = TrackState(track_id=1, camera_id=1)
        for i in range(4):
            state.reads.append(
                WeightedRead(
                    text="UP32AB1234", raw_text="UP32AB1234", rec_confidence=0.97,
                    per_char_confidence=[0.97] * 10, plate_det_confidence=0.95,
                    quality=0.95, grammar=1.0, frame_ts=1000.0 + i,
                    frame_idx=i * 5, appearance=crop_hash.crop_hash(crop(i)),
                )
            )
        c = crop(900)
        state.note_observation(
            PlateObservation(
                frame_idx=200, ts=1020.0, width=c.shape[1], height=c.shape[0],
                quality=0.9, det_confidence=0.9, crop=c,
                appearance=crop_hash.crop_hash(c),
            ),
            keep=6,
        )
        assert proc.resolve_pending(state, now=2000.0) == 0
        assert proc.recognizer.calls == 0

    def test_a_hopeless_plate_gets_no_retry(self):
        """Not nearly-settled but unread. One more look will not fix it, and
        spending on it is what the budget exists to prevent."""
        from backend.app.events.track_state import TrackState, WeightedRead

        proc = processor(["UP32AB1234"])
        state = TrackState(track_id=1, camera_id=1)
        for i, text in enumerate(("UP32AB1234", "MH12CD5678", "KA05EF9012", "TN09GH3456")):
            state.reads.append(
                WeightedRead(
                    text=text, raw_text=text, rec_confidence=0.30,
                    per_char_confidence=[0.30] * 10, plate_det_confidence=0.5,
                    quality=0.4, grammar=1.0, frame_ts=1000.0 + i,
                    frame_idx=i * 5, appearance=crop_hash.crop_hash(crop(i)),
                )
            )
        assert proc.resolve_pending(state, now=2000.0) == 0

    def test_low_quality_crops_are_not_worth_a_retry(self):
        proc = processor(["UP32AB1234"])
        state = contested_state(proc, unread_crops=0)
        c = crop(700)
        state.note_observation(
            PlateObservation(
                frame_idx=300, ts=1030.0, width=c.shape[1], height=c.shape[0],
                quality=0.10, det_confidence=0.9, crop=c,
                appearance=crop_hash.crop_hash(c),
            ),
            keep=6,
        )
        assert proc.resolve_pending(state, now=2000.0) == 0

    def test_no_banked_crops_means_no_retry(self):
        proc = processor(["UP32AB1234"])
        state = contested_state(proc, unread_crops=0)
        assert proc.resolve_pending(state, now=2000.0) == 0

    def test_a_recognizer_failure_still_consumes_its_call(self):
        """A model that returns nothing on a crop has still been given the
        call, and must not be retried forever on the same evidence."""
        proc = processor([None])
        state = contested_state(proc, unread_crops=2)
        assert proc.resolve_pending(state, now=2000.0) == 2
        assert len(state.reads) == 4, "no new reads, but the calls were spent"
