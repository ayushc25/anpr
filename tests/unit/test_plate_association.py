"""Plate-to-vehicle geometric consistency, and the partially-visible guard.

The rules here encode one mounting: a fixed camera at 3-4 ft watching a
controlled lane, so vehicles are seen front-on or rear-on with the plate low
and roughly centred. Every threshold is wrong for a different mounting, which
is why they are all configurable.
"""
from __future__ import annotations

import pytest

from backend.app.video.plate_association import (
    AssociationConfig,
    plate_within_frame,
    validate,
    vehicle_laterally_visible,
)

FRAME = (720, 1280)  # height, width
CFG = AssociationConfig()

#: A car mid-frame, 300x240, plate 70x18 low and centred.
VEHICLE = (400, 300, 700, 540)
PLATE = (515, 470, 585, 488)


class TestAPlausiblePlate:
    def test_a_normal_plate_on_a_normal_vehicle_is_accepted(self):
        assert validate(PLATE, VEHICLE, FRAME, CFG)

    def test_it_works_without_a_frame_shape(self):
        """frame_shape is optional so the geometry can be reasoned about in
        isolation; the frame-edge checks simply do not run."""
        assert validate(PLATE, VEHICLE, None, CFG)

    def test_disabled_config_accepts_anything(self):
        nonsense = (0, 0, 5, 5)
        assert validate(nonsense, VEHICLE, FRAME, AssociationConfig(enabled=False))


class TestThingsThatAreNotPlates:
    def test_a_windscreen_permit_is_rejected(self):
        """Plate-shaped, inside the vehicle, right size — but in the upper
        band, where no registration plate sits at this mounting."""
        permit = (515, 330, 585, 348)
        result = validate(permit, VEHICLE, FRAME, CFG)
        assert not result
        assert result.reason == "plate_too_high_on_vehicle"

    def test_a_bumper_sticker_is_too_small_relative_to_the_vehicle(self):
        sticker = (515, 470, 533, 480)  # 18 px wide against a 300 px vehicle
        result = validate(sticker, VEHICLE, FRAME, CFG)
        assert not result
        assert result.reason == "plate_too_small_for_vehicle"

    def test_something_spanning_the_vehicle_is_rejected(self):
        """A reflective strip or a painted commercial legend."""
        strip = (410, 470, 690, 500)  # 93% of the vehicle's width
        result = validate(strip, VEHICLE, FRAME, CFG)
        assert not result
        assert result.reason == "plate_too_large_for_vehicle"

    def test_a_plate_outside_its_own_vehicle_is_rejected(self):
        """The signature of the detector finding the plate of the vehicle
        BEHIND this one through a gap — which would attach a real plate to
        the wrong track."""
        elsewhere = (760, 470, 830, 488)
        result = validate(elsewhere, VEHICLE, FRAME, CFG)
        assert not result
        assert result.reason == "plate_outside_vehicle"

    def test_a_degenerate_box_is_rejected(self):
        assert not validate((10, 10, 10, 10), VEHICLE, FRAME, CFG)

    def test_slight_overhang_is_tolerated(self):
        """The plate crop is padded and the vehicle box jitters, so strict
        containment would reject real plates while the tracker is coasting."""
        overhanging = (515, 470, 585, 545)  # a little below the vehicle box
        assert validate(overhanging, VEHICLE, FRAME, AssociationConfig(inside_tolerance=0.30))


class TestPartiallyVisibleVehicle:
    def test_a_vehicle_half_off_the_left_edge_is_rejected(self):
        """The reported issue: a partially visible vehicle still triggering a
        plate read. Its width is a fragment, so every size ratio measured
        against it is meaningless."""
        clipped = (0, 300, 200, 540)
        result = vehicle_laterally_visible(clipped, FRAME, CFG)
        assert not result
        assert result.reason == "vehicle_clipped_laterally"

    def test_a_vehicle_half_off_the_right_edge_is_rejected(self):
        assert not vehicle_laterally_visible((1150, 300, 1280, 540), FRAME, CFG)

    def test_a_fully_visible_vehicle_is_accepted(self):
        assert vehicle_laterally_visible(VEHICLE, FRAME, CFG)

    def test_running_off_the_BOTTOM_is_allowed(self):
        """The one that matters most at this mounting. A vehicle driving
        toward a 3-4 ft camera legitimately runs off the bottom of the frame
        as it arrives, and those closest frames are the best the camera will
        ever get. Rejecting them would discard exactly what the scheduler
        spends its whole budget trying to reach."""
        arriving = (400, 480, 700, 900)  # bottom well past 720
        assert vehicle_laterally_visible(arriving, FRAME, CFG)

    def test_the_vertical_position_test_is_skipped_when_height_is_a_fragment(self):
        """A vertically clipped box has no real height, so plate-position-
        within-vehicle cannot be computed — it must degrade, not reject."""
        arriving = (400, 480, 700, 900)
        # This plate would fail the vertical band if height were trusted.
        high_plate = (515, 500, 585, 518)
        assert validate(high_plate, arriving, FRAME, CFG)

    def test_lateral_check_can_be_disabled(self):
        cfg = AssociationConfig(require_lateral_visibility=False)
        assert cfg.require_lateral_visibility is False


class TestPlateClippedByTheFrame:
    def test_a_plate_touching_the_frame_edge_is_rejected(self):
        """The geometric detector for a truncated read — the UP1606 failure.
        The recognizer reports the characters it can see with full confidence,
        having no way to know the plate continues past the image."""
        vehicle = (0, 300, 300, 540)
        plate = (0, 470, 70, 488)
        result = plate_within_frame(plate, FRAME, CFG)
        assert not result
        assert result.reason == "plate_clipped_at_edge"

    def test_a_plate_at_the_right_edge_is_rejected(self):
        assert not plate_within_frame((1215, 470, 1280, 488), FRAME, CFG)

    def test_a_plate_clear_of_the_edges_is_accepted(self):
        assert plate_within_frame(PLATE, FRAME, CFG)

    def test_the_check_runs_inside_validate(self):
        vehicle = (10, 300, 310, 540)
        plate = (12, 470, 82, 488)  # 12 px from the left edge, margin is 2
        assert validate(plate, vehicle, FRAME, CFG)
        strict = AssociationConfig(plate_edge_margin=20)
        result = validate(plate, vehicle, FRAME, strict)
        assert not result
        assert result.reason == "plate_clipped_at_edge"

    def test_the_check_can_be_disabled(self):
        vehicle = (0, 300, 300, 540)
        plate = (0, 470, 70, 488)
        cfg = AssociationConfig(require_whole_plate=False)
        assert validate(plate, vehicle, FRAME, cfg)


class TestMotorcycles:
    def test_a_narrow_vehicle_with_a_proportionally_wide_plate_passes(self):
        """A motorcycle's plate is a large fraction of its visible width, and
        the upper bound must not exclude it."""
        bike = (600, 400, 700, 560)   # 100 px wide
        plate = (625, 500, 680, 528)  # 55 px = 55% of vehicle width
        assert validate(plate, bike, FRAME, CFG)
