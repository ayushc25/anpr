"""ByteTrack, implemented directly against numpy.

ByteTrack's central idea is worth having at a gate: associate high-confidence
detections first, then give the *low*-confidence leftovers a second chance to
continue an existing track. A vehicle that becomes briefly occluded by the
gate post, or whose detection score dips as it turns, keeps its track id —
and keeping the id is what keeps its accumulated plate reads together.

Written here rather than pulled from ``supervision``/``boxmot`` to keep the
serving path free of torch and matplotlib, and because we need the retirement
signal (``removed_track_ids``) that those wrappers do not expose cleanly.

Motion is a damped constant-velocity extrapolation rather than a Kalman
filter: at 6-10 processing FPS with vehicles moving slowly through a gate, the
extra accuracy of a Kalman does not pay for its cost or its tuning surface.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..types import Detection, Track
from .base import VehicleTracker

_TENTATIVE, _CONFIRMED, _LOST, _REMOVED = 0, 1, 2, 3


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def association_cost(a: np.ndarray, b: np.ndarray, centroid_weight: float = 0.9) -> np.ndarray:
    """(1 - similarity) between track boxes and detection boxes.

    Similarity is IoU, falling back to a size-normalized centroid affinity when
    the boxes no longer overlap. The fallback matters at the frame rates a CPU
    box actually runs: at 6 processing FPS a vehicle can move most of its own
    length between frames, leaving IoU near zero for two boxes that are
    obviously the same car. Pure-IoU association breaks the track there, and a
    broken track scatters a vehicle's plate reads across two events.

    The fallback is weighted below 1.0 so a genuine IoU overlap always wins the
    assignment over a merely-nearby box.
    """
    if a.size == 0 or b.size == 0:
        return np.ones((len(a), len(b)), dtype=np.float32)

    iou = iou_matrix(a, b)

    centre_a = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], axis=1)
    centre_b = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    distance = np.linalg.norm(centre_a[:, None, :] - centre_b[None, :, :], axis=2)

    diag_a = np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
    diag_b = np.hypot(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    scale = np.maximum((diag_a[:, None] + diag_b[None, :]) / 2, 1e-6) * 0.75

    centroid_similarity = np.clip(1.0 - distance / scale, 0.0, 1.0) * centroid_weight
    return (1.0 - np.maximum(iou, centroid_similarity)).astype(np.float32)


def _assign(cost: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Linear assignment on a cost matrix, Hungarian when scipy is present and
    a greedy fallback when it is not (the greedy result is identical in the
    overwhelming majority of gate frames, where tracks do not compete)."""
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))

    pairs: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment

        r_idx, c_idx = linear_sum_assignment(cost)
        pairs = [(int(r), int(c)) for r, c in zip(r_idx, c_idx) if cost[r, c] <= threshold]
    except Exception:
        used_r: set[int] = set()
        used_c: set[int] = set()
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in order:
            r, c = int(r), int(c)
            if cost[r, c] > threshold:
                break
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))

    matched_r = {r for r, _ in pairs}
    matched_c = {c for _, c in pairs}
    return pairs, [r for r in range(rows) if r not in matched_r], [c for c in range(cols) if c not in matched_c]


class _Strack:
    __slots__ = (
        "track_id", "bbox", "velocity", "class_name", "confidence",
        "state", "age", "hits", "time_since_update", "history",
    )

    def __init__(self, track_id: int, det: Detection):
        self.track_id = track_id
        self.bbox = np.array(det.bbox, dtype=np.float32)
        self.velocity = np.zeros(4, dtype=np.float32)
        self.class_name = det.class_name
        self.confidence = det.confidence
        self.state = _TENTATIVE
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history: list[tuple[float, float]] = [self._centroid()]

    def _centroid(self) -> tuple[float, float]:
        return (float((self.bbox[0] + self.bbox[2]) / 2), float((self.bbox[1] + self.bbox[3]) / 2))

    def predict(self, damping: float = 0.7) -> None:
        """Coast one frame. Damping stops a track from sailing across the
        frame during a long occlusion and stealing an unrelated detection."""
        self.bbox = self.bbox + self.velocity
        self.velocity *= damping
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection, min_hits: int) -> None:
        new_box = np.array(det.bbox, dtype=np.float32)
        self.velocity = 0.5 * self.velocity + 0.5 * (new_box - self.bbox)
        self.bbox = new_box
        self.confidence = det.confidence
        # A car briefly scored as a truck should not rename the track; only
        # take the class from a confident detection.
        if det.confidence >= 0.5:
            self.class_name = det.class_name
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self._centroid())
        if len(self.history) > 240:
            del self.history[:120]
        if self.state in (_TENTATIVE, _LOST) and self.hits >= min_hits:
            self.state = _CONFIRMED
        elif self.state == _LOST:
            self.state = _CONFIRMED

    def as_track(self) -> Track:
        x1, y1, x2, y2 = (int(round(v)) for v in self.bbox)
        return Track(
            track_id=self.track_id,
            bbox=(x1, y1, x2, y2),
            class_name=self.class_name,
            confidence=float(self.confidence),
            age=self.age,
            centroid_history=list(self.history),
        )


class ByteTracker(VehicleTracker):
    name = "bytetrack"

    def __init__(
        self,
        track_thresh: float = 0.35,
        match_thresh: float = 0.80,
        low_thresh: float = 0.10,
        track_buffer: int = 45,
        min_hits: int = 2,
        **_ignored,
    ):
        self.track_thresh = float(track_thresh)
        self.match_thresh = float(match_thresh)
        self.low_thresh = float(low_thresh)
        self.track_buffer = int(track_buffer)
        self.min_hits = int(min_hits)
        self._ids = itertools.count(1)
        self._tracks: list[_Strack] = []
        self._removed: list[int] = []

    def reset(self) -> None:
        self._tracks.clear()
        self._removed.clear()

    def removed_track_ids(self) -> list[int]:
        out, self._removed = self._removed, []
        return out

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
        detector_ran: bool = True,
    ) -> list[Track]:
        for trk in self._tracks:
            trk.predict()

        if not detector_ran:
            # Coast: extrapolate and return, without association or
            # retirement. There is no evidence this frame either way, so
            # penalising tracks for a missing detection would be wrong.
            return [
                t.as_track() for t in self._tracks
                if t.state in (_CONFIRMED, _TENTATIVE) and t.time_since_update <= self.track_buffer
            ]

        high = [d for d in detections if d.confidence >= self.track_thresh]
        low = [d for d in detections if self.low_thresh <= d.confidence < self.track_thresh]

        active = [t for t in self._tracks if t.state != _REMOVED]
        # match_thresh is a DISTANCE threshold on (1 - IoU), following the
        # original ByteTrack: the default 0.8 accepts a match at IoU >= 0.2,
        # which is what lets a vehicle moving quickly between processed frames
        # keep its id.
        cost_threshold = self.match_thresh

        # --- pass 1: confident detections against every live track ---------
        boxes = np.array([t.bbox for t in active], dtype=np.float32) if active else np.empty((0, 4), np.float32)
        det_boxes = np.array([d.bbox for d in high], dtype=np.float32) if high else np.empty((0, 4), np.float32)
        pairs, unmatched_t, unmatched_d = _assign(association_cost(boxes, det_boxes), cost_threshold)
        for ti, di in pairs:
            active[ti].update(high[di], self.min_hits)

        # --- pass 2: ByteTrack's second look, low-confidence leftovers ------
        # These are never allowed to START a track; they only continue one.
        remaining = [active[i] for i in unmatched_t]
        if remaining and low:
            r_boxes = np.array([t.bbox for t in remaining], dtype=np.float32)
            l_boxes = np.array([d.bbox for d in low], dtype=np.float32)
            # The second pass is stricter (IoU >= 0.5): a low-confidence
            # detection may only continue a track it clearly overlaps.
            pairs2, unmatched_r, _ = _assign(association_cost(r_boxes, l_boxes), 0.5)
            for ti, di in pairs2:
                remaining[ti].update(low[di], self.min_hits)
            still_lost = [remaining[i] for i in unmatched_r]
        else:
            still_lost = remaining

        for trk in still_lost:
            if trk.state == _CONFIRMED:
                trk.state = _LOST
            elif trk.state == _TENTATIVE:
                # An unconfirmed track that misses immediately was noise.
                trk.state = _REMOVED

        for di in unmatched_d:
            self._tracks.append(_Strack(next(self._ids), high[di]))

        # --- retirement -----------------------------------------------------
        kept: list[_Strack] = []
        for trk in self._tracks:
            expired = trk.time_since_update > self.track_buffer
            if trk.state == _REMOVED or expired:
                if trk.hits >= self.min_hits:
                    # Only report tracks that were real; noise never reaches
                    # the event builder.
                    self._removed.append(trk.track_id)
                continue
            kept.append(trk)
        self._tracks = kept

        return [t.as_track() for t in self._tracks if t.state == _CONFIRMED and t.time_since_update == 0]

    def all_tracks(self) -> list[Track]:
        """Confirmed tracks including coasting ones — used for the preview
        overlay, where a box that blinks out for one frame looks like a bug."""
        return [t.as_track() for t in self._tracks if t.state in (_CONFIRMED, _LOST)]
