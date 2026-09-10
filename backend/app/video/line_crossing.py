"""Virtual line crossing and direction.

Direction is decided by which side of a directed line a track's centroid moves
from and to — not by whether the camera is labelled ENTRY or EXIT. A single
camera watching a shared gate sees both, and deriving direction from geometry
is what lets one camera report both without an operator lying about it in the
config.

The convention: with the line running from A to B, the "positive" side is the
left-hand side of the A->B vector. A track moving from negative to positive
crosses in the ``forward`` direction; the camera's ``forward_direction``
setting says whether forward means IN or OUT at this gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import cv2
import numpy as np

Point = tuple[float, float]


class CrossDirection(str, Enum):
    NONE = "none"
    IN = "in"
    OUT = "out"


@dataclass
class VirtualLine:
    """Two normalized points plus the meaning of the forward direction."""

    a: Point
    b: Point
    forward_direction: CrossDirection = CrossDirection.IN
    name: str = "line"
    #: Minimum perpendicular travel, as a fraction of the frame diagonal,
    #: before a side change counts. Stops a stationary vehicle whose box
    #: jitters across the line from generating an event every few frames.
    min_travel: float = 0.02

    @classmethod
    def from_config(cls, geometry: Optional[Sequence], forward: str = "in", name: str = "line") -> Optional["VirtualLine"]:
        if not geometry or len(geometry) < 2:
            return None

        def _point(raw) -> Point:
            if isinstance(raw, dict):
                return (float(raw["x"]), float(raw["y"]))
            return (float(raw[0]), float(raw[1]))

        try:
            direction = CrossDirection(forward)
        except ValueError:
            direction = CrossDirection.IN
        return cls(a=_point(geometry[0]), b=_point(geometry[1]), forward_direction=direction, name=name)

    def to_config(self) -> list[dict]:
        return [
            {"x": round(self.a[0], 6), "y": round(self.a[1], 6)},
            {"x": round(self.b[0], 6), "y": round(self.b[1], 6)},
        ]

    @property
    def backward_direction(self) -> CrossDirection:
        return CrossDirection.OUT if self.forward_direction == CrossDirection.IN else CrossDirection.IN

    def pixels(self, frame_shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        height, width = frame_shape[:2]
        return (
            (int(self.a[0] * width), int(self.a[1] * height)),
            (int(self.b[0] * width), int(self.b[1] * height)),
        )

    def _side(self, point: tuple[float, float], frame_shape: tuple[int, int]) -> float:
        """Signed cross product: >0 one side, <0 the other, magnitude is
        proportional to distance from the line."""
        (ax, ay), (bx, by) = self.pixels(frame_shape)
        px, py = point
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    def check(self, history: Sequence[tuple[float, float]], frame_shape: tuple[int, int]) -> CrossDirection:
        """Did this centroid path cross the line, and which way?

        Uses the whole history rather than the last two points so that a
        crossing is still detected when the detector ran 1-in-N frames and the
        tracker coasted across the line.
        """
        if len(history) < 2:
            return CrossDirection.NONE

        height, width = frame_shape[:2]
        diagonal = float(np.hypot(width, height))
        (ax, ay), (bx, by) = self.pixels(frame_shape)
        length = float(np.hypot(bx - ax, by - ay)) or 1.0
        threshold = self.min_travel * diagonal * length  # cross product scales with length

        signs = [self._side(point, frame_shape) for point in history]
        first_significant = next((s for s in signs if abs(s) > threshold), None)
        last_significant = next((s for s in reversed(signs) if abs(s) > threshold), None)
        if first_significant is None or last_significant is None:
            return CrossDirection.NONE
        if (first_significant > 0) == (last_significant > 0):
            return CrossDirection.NONE

        # The crossing must also happen within the segment, not on its
        # infinite extension: a vehicle passing well beyond the end of the
        # drawn line has not gone through the gate.
        if not self._crosses_segment(history, frame_shape):
            return CrossDirection.NONE

        return self.forward_direction if last_significant > 0 else self.backward_direction

    def _crosses_segment(self, history: Sequence[tuple[float, float]], frame_shape: tuple[int, int]) -> bool:
        (ax, ay), (bx, by) = self.pixels(frame_shape)
        for (x1, y1), (x2, y2) in zip(history, history[1:]):
            if _segments_intersect((ax, ay), (bx, by), (x1, y1), (x2, y2)):
                return True
        return False

    def draw(self, frame: np.ndarray, color=(64, 190, 200), thickness: int = 2) -> None:
        start, end = self.pixels(frame.shape[:2])
        cv2.line(frame, start, end, color, thickness)
        # A short arrow on the forward normal, so the operator can see which
        # way "IN" points without opening the config.
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        dx, dy = end[0] - start[0], end[1] - start[1]
        norm = float(np.hypot(dx, dy)) or 1.0
        tip = (int(mid[0] - dy / norm * 34), int(mid[1] + dx / norm * 34))
        cv2.arrowedLine(frame, mid, tip, color, thickness, tipLength=0.35)


def _orientation(p, q, r) -> float:
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])


def _segments_intersect(p1, p2, p3, p4) -> bool:
    d1, d2 = _orientation(p3, p4, p1), _orientation(p3, p4, p2)
    d3, d4 = _orientation(p1, p2, p3), _orientation(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
