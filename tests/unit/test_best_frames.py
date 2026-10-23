"""Best-frame selection for the deferred VLM flow.

What is under test is the CHOICE of crops, not recognition: which of a track's
banked observations are worth the VLM's ~2 s each, and which are the same view
paid for twice.
"""
from __future__ import annotations

import numpy as np

from backend.app.ai.types import PlateObservation
from backend.app.video.best_frames import SelectionConfig, is_usable, select_best, usable_observations


def obs(frame_idx, quality, appearance, width=120, crop=True):
    return PlateObservation(
        frame_idx=frame_idx,
        ts=float(frame_idx),
        width=width,
        height=40,
        quality=quality,
        det_confidence=0.9,
        crop=np.zeros((40, width, 3), dtype=np.uint8) if crop else None,
        appearance=appearance,
    )


# Hashes that differ in >= 6 bits from each other (the default distinctness
# threshold) and from zero, which means "unknown".
H1 = 0x0000_0000_0000_00FF
H2 = 0x0000_0000_00FF_0000
H3 = 0x0000_FF00_0000_0000
H4 = 0xFF00_0000_0000_0000
H5 = 0x00FF_0000_0000_0000
H6 = 0x0000_0000_FF00_0000
H7 = 0x0000_0000_0000_FF00
H8 = 0x00_00_00_FF_00_00_00_00


def test_best_quality_first_and_capped_at_k():
    pool = [obs(i * 3, 0.5 + i * 0.05, h) for i, h in enumerate([H1, H2, H3, H4, H5, H6, H7, H8])]
    chosen = select_best(pool, SelectionConfig(best_k=5))
    assert len(chosen) == 5
    qualities = [o.quality for o in chosen]
    assert qualities == sorted(qualities, reverse=True)
    # The five best of eight, not the five newest or the first five.
    assert min(qualities) == sorted((o.quality for o in pool), reverse=True)[4]


def test_near_identical_views_take_one_slot():
    """Seven copies of one stationary view are one observation.

    This is the failure the selector exists to prevent: a fluent VLM reads all
    seven identically and the validator counts that as sevenfold agreement.
    """
    same_view = [obs(i * 3, 0.9 - i * 0.01, H1 ^ (1 << i)) for i in range(7)]  # 1 bit apart
    other = obs(30, 0.6, H4)
    chosen = select_best(same_view + [other], SelectionConfig(best_k=7))
    assert len(chosen) == 2
    assert {id(o) for o in chosen} == {id(same_view[0]), id(other)}


def test_neighbouring_frames_are_not_both_chosen():
    a = obs(10, 0.9, H1)
    b = obs(11, 0.8, H4)          # a different-looking crop, but one frame later
    c = obs(14, 0.7, H3)
    chosen = select_best([a, b, c], SelectionConfig(best_k=7, min_frame_gap=2))
    assert [o.frame_idx for o in chosen] == [10, 14]


def test_not_padded_to_reach_k():
    """A vehicle that sat still produced one view; sending it once is right."""
    chosen = select_best([obs(1, 0.8, H1), obs(5, 0.7, H1 ^ 1)], SelectionConfig(best_k=7))
    assert len(chosen) == 1


def test_unusable_observations_never_chosen():
    cfg = SelectionConfig(best_k=6, min_quality=0.35, min_width=45)
    low_quality = obs(1, 0.2, H1)
    narrow = obs(5, 0.9, H2, width=30)
    released = obs(9, 0.9, H3, crop=False)       # pixels already dropped
    good = obs(13, 0.5, H4)
    assert not is_usable(low_quality, cfg)
    assert not is_usable(narrow, cfg)
    assert not is_usable(released, cfg)
    assert usable_observations([low_quality, narrow, released, good], cfg) == [good]
    assert select_best([low_quality, narrow, released, good], cfg) == [good]


def test_empty_when_nothing_usable():
    assert select_best([], SelectionConfig()) == []
    assert select_best([obs(1, 0.1, H1)], SelectionConfig()) == []


def test_unknown_hash_counts_as_distinct():
    """A zero hash means 'could not compute', and refusing a crop on the
    strength of a hash we do not have would be the wrong way to be wrong."""
    chosen = select_best([obs(1, 0.9, 0), obs(5, 0.8, 0)], SelectionConfig(best_k=3))
    assert len(chosen) == 2


def test_invalid_k_rejected():
    import pytest

    with pytest.raises(ValueError):
        SelectionConfig(best_k=0)


# ---- chains: one physical plate -------------------------------------------
def chained(frame_idx, quality, appearance, group):
    o = obs(frame_idx, quality, appearance)
    o.group = group
    return o


def test_only_one_physical_plate_is_selected():
    """A track in dense traffic can span several cars. The diversity rule would
    pick their plates ON PURPOSE (a different car looks maximally different),
    so selection must first commit to one plate."""
    car_a = [chained(10 + i * 4, 0.8 - i * 0.02, h, group=1) for i, h in enumerate([H1, H2, H3, H4])]
    car_b = [chained(12 + i * 4, 0.85 - i * 0.02, h, group=2) for i, h in enumerate([H5, H6, H7])]
    chosen = select_best(car_a + car_b, SelectionConfig(best_k=6))
    assert {o.group for o in chosen} == {1}                    # the chain with more (and good) evidence
    assert len(chosen) == 4


def test_the_chain_with_the_most_evidence_wins_not_the_single_best_crop():
    strong = [chained(10 + i * 4, 0.7, h, group=1) for i, h in enumerate([H1, H2, H3, H4, H5])]
    lone = [chained(100, 0.95, H6, group=2)]                   # one lovely crop of another car
    chosen = select_best(strong + lone, SelectionConfig(best_k=6))
    assert {o.group for o in chosen} == {1}


def test_pools_without_chain_information_are_untouched():
    pool = [obs(i * 3, 0.9 - i * 0.01, h) for i, h in enumerate([H1, H2, H3])]   # all group 0
    assert len(select_best(pool, SelectionConfig(best_k=6))) == 3
