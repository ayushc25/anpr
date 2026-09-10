"""Appearance hashing, diverse Top-K retention, and correlated-evidence discounting.

Item 10 of the brief, stated as a property: four consecutive nearly identical
frames must not count as four fully independent observations.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.app.ai.quality import crop_hash
from backend.app.ai.types import PlateObservation
from backend.app.events import char_fusion
from backend.app.events.char_fusion import fuse
from backend.app.events.track_state import TrackState


def plate_crop(seed: int, width: int = 130, height: int = 34) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(40, 220, (height, width, 3), dtype=np.uint8)


def jitter(crop: np.ndarray, amount: int = 2, seed: int = 0) -> np.ndarray:
    """The same view again, with sensor noise — what a stopped vehicle looks
    like frame to frame."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-amount, amount + 1, crop.shape, dtype=np.int16)
    return np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def observation(idx: int, quality: float, crop: np.ndarray) -> PlateObservation:
    return PlateObservation(
        frame_idx=idx,
        ts=1000.0 + idx * 0.16,
        width=crop.shape[1],
        height=crop.shape[0],
        quality=quality,
        det_confidence=0.9,
        crop=crop,
        appearance=crop_hash.crop_hash(crop),
    )


class TestCropHash:
    def test_identical_crops_hash_identically(self):
        crop = plate_crop(1)
        assert crop_hash.crop_hash(crop) == crop_hash.crop_hash(crop.copy())

    def test_sensor_noise_does_not_change_the_hash_much(self):
        crop = plate_crop(2)
        a, b = crop_hash.crop_hash(crop), crop_hash.crop_hash(jitter(crop, 2, seed=5))
        assert crop_hash.hamming(a, b) < crop_hash.DEFAULT_MIN_DISTANCE
        assert not crop_hash.is_distinct(a, b)

    def test_a_different_view_is_distinct(self):
        a, b = crop_hash.crop_hash(plate_crop(3)), crop_hash.crop_hash(plate_crop(99))
        assert crop_hash.is_distinct(a, b)

    def test_brightness_shift_alone_does_not_change_the_hash(self):
        """dHash encodes gradient direction, not intensity — so a plate that
        just moved into shade is still the same view."""
        crop = plate_crop(4)
        darker = np.clip(crop.astype(np.int16) - 30, 0, 255).astype(np.uint8)
        assert not crop_hash.is_distinct(crop_hash.crop_hash(crop), crop_hash.crop_hash(darker))

    def test_an_unhashable_crop_counts_as_distinct(self):
        """Refusing to store evidence because a hash could not be computed
        would be the wrong way to be wrong."""
        assert crop_hash.crop_hash(None) == 0
        assert crop_hash.is_distinct(0, 12345)

    def test_the_hash_is_64_bits(self):
        assert 0 <= crop_hash.crop_hash(plate_crop(7)) < 2 ** 64


class TestDiverseRetention:
    def test_near_identical_frames_do_not_fill_the_set(self):
        """A stopped vehicle produces frames differing only by noise. Pure
        top-K stores six copies of one view; the set must hold one."""
        state = TrackState(track_id=1, camera_id=1)
        base = plate_crop(11)
        for i in range(6):
            state.note_observation(observation(i, 0.80, jitter(base, 2, seed=i)), keep=6)
        assert len(state.plate_observations) == 1
        assert state.observation_count == 6, "all six were still recorded as looks"

    def test_the_best_of_a_redundant_group_is_the_one_kept(self):
        state = TrackState(track_id=1, camera_id=1)
        base = plate_crop(12)
        state.note_observation(observation(0, 0.50, jitter(base, 2, seed=1)), keep=6)
        state.note_observation(observation(1, 0.90, jitter(base, 2, seed=2)), keep=6)
        state.note_observation(observation(2, 0.60, jitter(base, 2, seed=3)), keep=6)
        assert len(state.plate_observations) == 1
        assert state.plate_observations[0].quality == pytest.approx(0.90)

    def test_genuinely_different_views_all_survive(self):
        state = TrackState(track_id=1, camera_id=1)
        for i in range(5):
            state.note_observation(observation(i, 0.70, plate_crop(100 + i)), keep=6)
        assert len(state.plate_observations) == 5

    def test_the_set_is_still_bounded(self):
        state = TrackState(track_id=1, camera_id=1)
        for i in range(20):
            state.note_observation(observation(i, 0.50 + i * 0.01, plate_crop(200 + i)), keep=6)
        assert len(state.plate_observations) <= 6

    def test_evicted_observations_release_their_pixels(self):
        state = TrackState(track_id=1, camera_id=1)
        for i in range(12):
            state.note_observation(observation(i, 0.90 - i * 0.05, plate_crop(300 + i)), keep=4)
        assert len(state.plate_observations) == 4
        assert all(o.crop is not None for o in state.plate_observations)

    def test_unread_observations_are_the_retry_pool(self):
        state = TrackState(track_id=1, camera_id=1)
        for i in range(4):
            state.note_observation(observation(i, 0.60 + i * 0.05, plate_crop(400 + i)), keep=6)
        state.plate_observations[0].ocr_ran = True
        pool = state.unread_observations()
        assert len(pool) == 3
        assert all(not o.ocr_ran for o in pool)
        # Best first, so a retry spends its call on the strongest unread crop.
        assert pool == sorted(pool, key=lambda o: o.quality, reverse=True)


class Read:
    """Minimal read for fusion, carrying the independence signals."""

    def __init__(self, text, p=0.9, weight=1.0, frame_idx=0, appearance=0):
        self.text = text
        self.weight = weight
        self._p = p
        self.frame_idx = frame_idx
        self.appearance = appearance

    def char_confidence(self, i):
        return self._p


class TestCorrelatedEvidenceDiscount:
    def test_four_near_identical_reads_are_not_four_observations(self):
        """The headline property of item 10."""
        base = crop_hash.crop_hash(plate_crop(21))
        reads = [Read("UP32AB1234", 0.9, 1.0, frame_idx=i, appearance=base) for i in range(4)]
        result = fuse(reads)
        assert result.pool_size == 4
        assert result.independent_views == pytest.approx(2.0), "sqrt(4)"

    def test_four_distinct_reads_are_four_observations(self):
        reads = [
            Read("UP32AB1234", 0.9, 1.0, frame_idx=i * 5, appearance=crop_hash.crop_hash(plate_crop(30 + i)))
            for i in range(4)
        ]
        assert fuse(reads).independent_views == pytest.approx(4.0)

    def test_a_repeated_misread_cannot_outvote_a_correct_minority(self):
        """The failure this prevents: a stationary vehicle whose one bad crop
        is sampled four times, against two genuinely different good frames.

        UP14GS3664 read four times off one view vs UP14FS3664 twice off two
        different views — the F/G confusion from the brief.
        """
        stuck = crop_hash.crop_hash(plate_crop(41))
        reads = [Read("UP14GS3664", 0.80, 1.0, frame_idx=i, appearance=stuck) for i in range(4)]
        reads += [
            Read("UP14FS3664", 0.80, 1.0, frame_idx=20, appearance=crop_hash.crop_hash(plate_crop(50))),
            Read("UP14FS3664", 0.80, 1.0, frame_idx=30, appearance=crop_hash.crop_hash(plate_crop(51))),
        ]
        result = fuse(reads)
        # 4/sqrt(4) = 2.0 for the stuck view vs 2.0 for the diverse pair, so
        # position 4 is now genuinely contested rather than a 4-2 landslide.
        contested = result.position(4)
        assert contested.runner_up, "the correct character is still in contention"
        assert contested.margin < 0.30, f"should be near-tied, got {contested.margin:.2f}"

    def test_the_discount_does_not_fire_without_appearance_hashes(self):
        """EasyOCR path: no hash, so no clustering, so exactly the old
        behaviour. Being conservative when we cannot tell."""
        reads = [Read("UP32AB1234", 0.9, 1.0, frame_idx=i) for i in range(4)]
        assert fuse(reads).independent_views == pytest.approx(4.0)

    def test_a_wide_frame_gap_defeats_clustering(self):
        """Similar-looking crops far apart in time are two real passes, not
        one view sampled twice."""
        same = crop_hash.crop_hash(plate_crop(61))
        reads = [
            Read("UP32AB1234", 0.9, 1.0, frame_idx=0, appearance=same),
            Read("UP32AB1234", 0.9, 1.0, frame_idx=100, appearance=same),
        ]
        assert fuse(reads).independent_views == pytest.approx(2.0)

    def test_char_support_is_unchanged_by_pure_redundancy(self):
        """Discounting must not make redundant evidence WORSE than a single
        read — it removes phantom confirmation, it does not punish."""
        base = crop_hash.crop_hash(plate_crop(71))
        one = fuse([
            Read("UP32AB1234", 0.60, 1.0, frame_idx=0, appearance=base),
            Read("UP32AB1234", 0.60, 1.0, frame_idx=1, appearance=base),
        ])
        many = fuse([
            Read("UP32AB1234", 0.60, 1.0, frame_idx=i, appearance=base) for i in range(4)
        ])
        assert one.char_support() == pytest.approx(many.char_support(), abs=0.02)
