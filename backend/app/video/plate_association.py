"""Is this plate plausibly ON this vehicle?

The plate detector runs on a vehicle crop and returns whatever looks
plate-shaped inside it. Nothing downstream ever asked whether the result is
geometrically consistent with the vehicle it supposedly belongs to, and at a
gate several things are reliably plate-shaped and reliably not plates: the
DVR's burnt-in timestamp bar, a bumper sticker, a windscreen permit, a
reflective strip on the vehicle behind, a phone number painted on a commercial
body. All of them pass the detector's only checks — width >= 24 px and an
aspect ratio between 1.2 and 8.0.

The other half of the problem is the vehicle, not the plate. A vehicle
half-inside the frame can still produce a confident plate read from whatever
part is visible, and that read is attached to a track whose box is wrong,
whose size gates are meaningless and whose plate may belong to the vehicle
behind it. A partially visible vehicle should not trigger a plate read at all.

Both checks are pure arithmetic on boxes already in hand — nanoseconds against
the ~15 ms plate-detector call they can avoid and the far more expensive OCR
call behind it. They run BEFORE quality scoring, so a rejected candidate costs
nothing further.

Everything here is a geometric plausibility test, never an identity test. It
can say "that is not where a plate sits on a car"; it can never say which car.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..ai.types import BBox

# --- defaults ---------------------------------------------------------------
#
# Derived from the mounting this system is specified for: a fixed camera at
# 3-4 ft watching a controlled lane, so a vehicle is seen front-on or rear-on
# and its plate sits low and roughly centred.

#: Fraction of the plate box allowed to fall outside the vehicle box. Small but
#: non-zero: the plate crop is padded by ``plate_crop_pad`` and the vehicle box
#: itself jitters, so demanding strict containment would reject real plates on
#: the frames where the tracker is coasting.
DEFAULT_INSIDE_TOLERANCE = 0.15

#: Plate width as a fraction of vehicle width. A real plate on a car seen
#: front-on is roughly a fifth to a third of the visible width; a motorcycle
#: plate against a narrow vehicle box runs higher. Outside 0.08-0.65 the box is
#: either a sticker-sized fragment or something spanning the whole vehicle.
DEFAULT_MIN_WIDTH_RATIO = 0.08
DEFAULT_MAX_WIDTH_RATIO = 0.65

#: Where the plate's centre may sit vertically within the vehicle box, as a
#: fraction of its height from the top. The lower band, because at a 3-4 ft
#: mount both front and rear plates are below the midline. Rejecting the upper
#: band is what discards a windscreen permit, a roof-line reflection and the
#: DVR timestamp when it happens to overlap the vehicle.
DEFAULT_MIN_VERTICAL = 0.30
DEFAULT_MAX_VERTICAL = 1.02  # slightly over 1.0: the box can clip a low plate

#: How far a vehicle box may extend past the LEFT or RIGHT frame edge, as a
#: fraction of its own width, before it counts as partially visible.
#:
#: Horizontal only, and that asymmetry is the whole point. At a 3-4 ft mount a
#: vehicle driving toward the camera legitimately runs off the BOTTOM of the
#: frame as it arrives — and those closest frames are the best ones the camera
#: will ever get, the largest and most frontal view of the plate. A rule that
#: rejected any box touching any edge would discard precisely the frames this
#: pipeline spends its whole scheduling budget trying to reach.
#:
#: Lateral clipping is different. A vehicle half-in at the side of the frame
#: has a box whose WIDTH is a measurement of a fragment, and width is what the
#: plate-size ratio below is measured against — so that test becomes
#: meaningless, and the plate found inside the fragment may belong to the
#: vehicle beside it. That is the "partially visible vehicle triggers a read"
#: case, and it is horizontal.
DEFAULT_MAX_EDGE_CLIP = 0.02

#: Margin, in pixels, the PLATE box must keep from the frame edge.
#:
#: A plate clipped by the frame boundary is the direct cause of a truncated
#: read — the recognizer reports the characters it can see, with full
#: confidence, having no way to know the plate continues past the edge. This
#: is the geometric detector for exactly the ``UP1606`` failure, and it is
#: cheaper and far more reliable than catching it afterwards by length.
DEFAULT_PLATE_EDGE_MARGIN = 2


@dataclass(frozen=True)
class AssociationConfig:
    enabled: bool = True
    inside_tolerance: float = DEFAULT_INSIDE_TOLERANCE
    min_width_ratio: float = DEFAULT_MIN_WIDTH_RATIO
    max_width_ratio: float = DEFAULT_MAX_WIDTH_RATIO
    min_vertical: float = DEFAULT_MIN_VERTICAL
    max_vertical: float = DEFAULT_MAX_VERTICAL
    #: Reject plate reads from a vehicle clipped by the LEFT or RIGHT frame
    #: edge. Vertical clipping is allowed and expected — see max_edge_clip.
    require_lateral_visibility: bool = True
    max_edge_clip: float = DEFAULT_MAX_EDGE_CLIP
    #: Reject a plate box that touches the frame edge, since the plate itself
    #: is then probably truncated.
    require_whole_plate: bool = True
    plate_edge_margin: int = DEFAULT_PLATE_EDGE_MARGIN


@dataclass(frozen=True)
class AssociationResult:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


OK = AssociationResult(True)


def _area(box: BBox) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def vehicle_laterally_visible(
    vehicle: BBox, frame_shape: tuple[int, int], cfg: AssociationConfig
) -> AssociationResult:
    """Whether the vehicle's full WIDTH is in frame.

    Horizontal only. See ``DEFAULT_MAX_EDGE_CLIP`` for why: vertical clipping
    is normal and desirable at a low gate mount, lateral clipping means the
    box is a fragment and its width — the denominator of the plate-size test —
    is not a real measurement.

    Expressed as a fraction of the box's own width so it behaves the same for
    a motorcycle and a truck.
    """
    width = frame_shape[1]
    x1, _y1, x2, _y2 = vehicle
    slack = max(1.0, cfg.max_edge_clip * max(1, x2 - x1))
    if x1 <= slack - 1 or x2 >= width - slack:
        return AssociationResult(False, "vehicle_clipped_laterally")
    return OK


def _vertically_clipped(vehicle: BBox, frame_shape: tuple[int, int]) -> bool:
    """Whether the vehicle box runs off the top or bottom of the frame.

    Not a rejection — an approaching vehicle does this and those are the best
    frames. It does mean the box HEIGHT is a fragment, so the vertical-position
    test below cannot be trusted and is skipped.
    """
    height = frame_shape[0]
    _x1, y1, _x2, y2 = vehicle
    return y1 <= 0 or y2 >= height - 1


def plate_within_frame(
    plate: BBox, frame_shape: tuple[int, int], cfg: AssociationConfig
) -> AssociationResult:
    """Whether the plate box keeps clear of the frame boundary.

    A plate touching the edge is very likely cut off, and a cut-off plate is
    read as a confident fragment — the recognizer cannot know the characters
    continue past the image. Catching it here, geometrically, is both cheaper
    and more reliable than inferring it later from the string's length.
    """
    height, width = frame_shape[:2]
    margin = max(0, cfg.plate_edge_margin)
    x1, y1, x2, y2 = plate
    if x1 <= margin or y1 <= margin:
        return AssociationResult(False, "plate_clipped_at_edge")
    if x2 >= width - 1 - margin or y2 >= height - 1 - margin:
        return AssociationResult(False, "plate_clipped_at_edge")
    return OK


def validate(
    plate: BBox,
    vehicle: BBox,
    frame_shape: Optional[tuple[int, int]] = None,
    cfg: Optional[AssociationConfig] = None,
) -> AssociationResult:
    """Whether ``plate`` is plausibly the registration plate of ``vehicle``.

    Both boxes in full-frame pixels, which is the invariant every stage in
    ``ai/types`` already maintains.
    """
    cfg = cfg or AssociationConfig()
    if not cfg.enabled:
        return OK

    vx1, vy1, vx2, vy2 = vehicle
    px1, py1, px2, py2 = plate
    vehicle_w, vehicle_h = vx2 - vx1, vy2 - vy1
    plate_w, plate_h = px2 - px1, py2 - py1
    if vehicle_w <= 0 or vehicle_h <= 0 or plate_w <= 0 or plate_h <= 0:
        return AssociationResult(False, "degenerate_box")

    # -- containment -------------------------------------------------------
    overlap_w = max(0, min(px2, vx2) - max(px1, vx1))
    overlap_h = max(0, min(py2, vy2) - max(py1, vy1))
    inside = (overlap_w * overlap_h) / float(plate_w * plate_h)
    if inside < 1.0 - cfg.inside_tolerance:
        # A plate mostly outside its own vehicle is the signature of the
        # detector finding the plate of the vehicle BEHIND this one through a
        # gap — which would attach a real plate to the wrong track.
        return AssociationResult(False, "plate_outside_vehicle")

    # -- relative size -----------------------------------------------------
    width_ratio = plate_w / float(vehicle_w)
    if width_ratio < cfg.min_width_ratio:
        return AssociationResult(False, "plate_too_small_for_vehicle")
    if width_ratio > cfg.max_width_ratio:
        return AssociationResult(False, "plate_too_large_for_vehicle")

    # -- plate clear of the frame boundary ---------------------------------
    if frame_shape is not None and cfg.require_whole_plate:
        whole = plate_within_frame(plate, frame_shape, cfg)
        if not whole:
            return whole

    # -- vertical position -------------------------------------------------
    #
    # Skipped when the vehicle box runs off the top or bottom of the frame:
    # its height is then a fragment, and dividing by a fragment gives a
    # position that means nothing. That happens on every close approach at a
    # low mount, which is exactly where the good frames are, so this must
    # degrade rather than reject.
    if frame_shape is not None and _vertically_clipped(vehicle, frame_shape):
        return OK

    centre_y = (py1 + py2) / 2.0
    position = (centre_y - vy1) / float(vehicle_h)
    if position < cfg.min_vertical:
        # Upper band: a windscreen permit, a roof reflection, or a burnt-in
        # timestamp overlapping the vehicle.
        return AssociationResult(False, "plate_too_high_on_vehicle")
    if position > cfg.max_vertical:
        return AssociationResult(False, "plate_below_vehicle")

    return OK
