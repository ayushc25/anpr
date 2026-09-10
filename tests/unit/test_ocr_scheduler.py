"""The OCR scheduler's decision table.

These are the tests that matter for the CPU bill: every RUN here is a
recognizer call, and every WAIT is one we did not pay for. The scheduler is
pure, so each case states a gate situation directly rather than driving a
pipeline to reach it.
"""
from __future__ import annotations

import pytest

from backend.app.video.ocr_scheduler import OcrAction, OcrContext, OcrPolicy, decide

POLICY = OcrPolicy()


def ctx(**overrides) -> OcrContext:
    """A crop that would be read on the normal cadence, unless overridden."""
    base = dict(
        quality=0.60,          # 'good' but not 'excellent'
        width=120,             # comfortably readable
        quality_floor=0.35,    # the existing PipelineConfig floor
        frames_since_ocr=99,   # cadence not a constraint
        ocr_spent=0,
        consecutive_waits=0,
        peak_width=120,
        approach_growth=1.0,   # stationary
        stale=False,
    )
    base.update(overrides)
    return OcrContext(**base)


class TestReadableSizeIsNotDetectableSize:
    def test_a_detectable_plate_below_reading_size_waits(self):
        """The core of the phase: the plate detector found it, and we are
        deliberately not reading it yet."""
        verdict = decide(ctx(width=POLICY.min_plate_width_read - 1), POLICY)
        assert verdict.action is OcrAction.WAIT
        assert verdict.reason == "plate_too_small"

    def test_it_waits_rather_than_skips(self):
        """WAIT, not SKIP: an approaching vehicle WILL become readable, and
        the distinction is what lets the caller keep observing it."""
        assert decide(ctx(width=30), POLICY).action is OcrAction.WAIT

    def test_at_the_readable_floor_it_runs(self):
        assert decide(ctx(width=POLICY.min_plate_width_read), POLICY).run

    def test_size_is_checked_before_quality(self):
        """A tiny crop can still score well on sharpness and exposure. Size
        has to win, or the resolution term alone decides readability."""
        verdict = decide(ctx(width=20, quality=0.95), POLICY)
        assert verdict.reason == "plate_too_small"


class TestQualityTiers:
    def test_excellent_frame_preempts_the_cadence(self):
        verdict = decide(ctx(quality=0.80, frames_since_ocr=1), POLICY)
        assert verdict.run and verdict.reason == "excellent_frame"

    def test_good_frame_waits_for_its_interval(self):
        assert not decide(ctx(quality=0.60, frames_since_ocr=1), POLICY).run
        assert decide(ctx(quality=0.60, frames_since_ocr=2), POLICY).run

    def test_marginal_frame_is_sampled_far_less_often(self):
        marginal = dict(quality=0.40)
        assert not decide(ctx(**marginal, frames_since_ocr=3), POLICY).run
        verdict = decide(ctx(**marginal, frames_since_ocr=5), POLICY)
        assert verdict.run and verdict.reason == "marginal_cadence"

    def test_below_the_existing_quality_floor_is_skipped(self):
        verdict = decide(ctx(quality=0.20), POLICY)
        assert verdict.action is OcrAction.SKIP
        assert verdict.reason == "below_quality_floor"

    def test_priority_is_quality_so_the_best_crop_wins_a_contested_frame(self):
        assert decide(ctx(quality=0.9), POLICY).priority > decide(ctx(quality=0.5), POLICY).priority


class TestApproachAndDeparture:
    def test_a_still_growing_plate_waits_for_a_better_frame(self):
        """The saving. The vehicle is coming to us and about to stop; reading
        it now spends a call on the worst frame we will be offered."""
        verdict = decide(ctx(quality=0.45, approach_growth=1.20), POLICY)
        assert verdict.action is OcrAction.WAIT
        assert verdict.reason == "still_approaching"

    def test_growth_does_not_delay_an_already_good_frame(self):
        """Waiting is only ever worth it when the frame in hand is mediocre."""
        assert decide(ctx(quality=0.60, approach_growth=1.20), POLICY).run

    def test_growth_below_the_ratio_is_treated_as_stationary(self):
        """A stopped vehicle's box jitters by a few percent. That must not
        read as 'still approaching' or it never gets recognized at all."""
        assert decide(ctx(quality=0.45, approach_growth=1.03), POLICY).run

    def test_patience_is_bounded(self):
        """A vehicle creeping in over a long lane must not be waited on
        forever — the safety valve on the rule above."""
        verdict = decide(
            ctx(quality=0.45, approach_growth=1.50,
                consecutive_waits=POLICY.max_consecutive_waits),
            POLICY,
        )
        assert verdict.run and verdict.reason == "patience_exhausted"

    def test_a_departing_vehicle_is_read_with_what_is_left(self):
        verdict = decide(ctx(quality=0.45, width=80, peak_width=140), POLICY)
        assert verdict.run and verdict.reason == "departing"

    def test_departure_beats_growth_when_both_somehow_hold(self):
        """A spurious oversized plate box can set a peak the vehicle never
        really reached, leaving a track both 'growing' and 'below peak'.
        Departure wins, so the tie resolves toward reading rather than
        waiting — the safe direction when the signals disagree."""
        verdict = decide(
            ctx(quality=0.45, width=80, peak_width=200, approach_growth=1.3), POLICY
        )
        assert verdict.run
        assert verdict.reason == "departing"

    def test_departure_needs_real_shrinkage_not_jitter(self):
        # 95% of peak is jitter, not departure.
        verdict = decide(ctx(quality=0.45, width=133, peak_width=140, frames_since_ocr=1), POLICY)
        assert not verdict.run


class TestBudget:
    def test_an_exhausted_budget_stops_everything(self):
        verdict = decide(ctx(quality=0.99, ocr_spent=POLICY.budget_per_track), POLICY)
        assert verdict.action is OcrAction.SKIP
        assert verdict.reason == "budget_exhausted"

    def test_the_reserve_is_withheld_from_ordinary_frames(self):
        spent = POLICY.budget_per_track - POLICY.reserve
        verdict = decide(ctx(quality=0.60, ocr_spent=spent), POLICY)
        assert verdict.action is OcrAction.WAIT
        assert verdict.reason == "holding_reserve"

    def test_the_reserve_is_released_for_an_excellent_frame(self):
        """The reserve exists to protect the good frames, so it must not
        block the best one."""
        spent = POLICY.budget_per_track - POLICY.reserve
        assert decide(ctx(quality=0.90, ocr_spent=spent), POLICY).run

    def test_the_reserve_is_released_on_departure(self):
        spent = POLICY.budget_per_track - POLICY.reserve
        verdict = decide(ctx(quality=0.45, width=80, peak_width=140, ocr_spent=spent), POLICY)
        assert verdict.run and verdict.reason == "departing"


class TestStaleness:
    def test_a_stale_frame_is_never_recognized(self):
        verdict = decide(ctx(quality=0.99, stale=True), POLICY)
        assert verdict.action is OcrAction.SKIP
        assert verdict.reason == "stale_frame"

    def test_staleness_outranks_every_other_rule(self):
        """Including the endgame rules, which otherwise force a read."""
        verdict = decide(
            ctx(quality=0.99, width=80, peak_width=200, stale=True,
                consecutive_waits=99, ocr_spent=0),
            POLICY,
        )
        assert verdict.action is OcrAction.SKIP


class TestPolicyValidation:
    def test_a_reserve_that_swallows_the_budget_is_rejected(self):
        with pytest.raises(ValueError, match="reserve"):
            OcrPolicy(budget_per_track=3, reserve=3)

    def test_inverted_quality_tiers_are_rejected(self):
        with pytest.raises(ValueError, match="good_quality"):
            OcrPolicy(good_quality=0.9, excellent_quality=0.5)

    def test_a_zero_per_frame_cap_is_rejected(self):
        with pytest.raises(ValueError, match="max_per_frame"):
            OcrPolicy(max_per_frame=0)


class TestGateNarrative:
    """The sequence the phase is specified against, in order."""

    def test_too_small_then_readable_then_excellent_then_done(self):
        # Far away: detected, not readable.
        assert decide(ctx(width=40, quality=0.30), POLICY).reason == "plate_too_small"
        # Approaching, readable, still growing: hold out for better. peak
        # tracks width, because a growing plate is setting the peak each frame.
        assert decide(
            ctx(width=80, peak_width=80, quality=0.45, approach_growth=1.3), POLICY
        ).reason == "still_approaching"
        # Stopped at the boom: the frame we waited for.
        assert decide(ctx(width=140, quality=0.85, approach_growth=1.0), POLICY).reason == "excellent_frame"
        # Locked upstream, budget spent: nothing more to pay for.
        assert not decide(ctx(width=140, quality=0.85, ocr_spent=99), POLICY).run
