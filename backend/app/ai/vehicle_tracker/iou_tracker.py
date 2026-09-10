"""Single-pass IoU tracker — the cheap fallback.

Behaviourally this is ByteTrack with its second association pass disabled, so
rather than maintaining a second implementation we configure the same one.
Useful when a camera's detector scores are bimodal enough that the low-score
pass causes id switches, and as a control when diagnosing tracking problems.
"""
from __future__ import annotations

from .bytetrack import ByteTracker


class IouTracker(ByteTracker):
    name = "iou_tracker"

    def __init__(self, track_thresh: float = 0.25, match_thresh: float = 0.75, **kwargs):
        kwargs.pop("low_thresh", None)
        super().__init__(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            low_thresh=track_thresh,  # empty low band -> no second pass
            **kwargs,
        )
