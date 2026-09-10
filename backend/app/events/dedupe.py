"""Event de-duplication.

Two distinct real-world problems, both of which produce duplicate rows if
ignored:

*Re-detection* — the tracker loses a vehicle behind the gate post and picks it
up as a new track a second later. Same plate, same direction, seconds apart.

*Bounce* — a vehicle noses over the line, the driver stops, reverses slightly,
then continues. The geometry genuinely records IN, OUT, IN. Only the first
should be an event.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..video.line_crossing import CrossDirection


@dataclass
class DedupeConfig:
    #: Same plate, same camera, same direction inside this window is the same
    #: arrival. 20-30s suits a society gate where a vehicle waits at a boom.
    same_direction_seconds: float = 25.0
    #: A reversal this soon after an event is a manoeuvre, not a departure.
    reversal_seconds: float = 12.0
    max_entries: int = 512


class EventDeduplicator:
    """In-process, per camera. Deliberately not backed by the database: this
    runs in the worker's hot path, and the API applies its own idempotency
    check on ``event_uid`` as the authoritative guard."""

    def __init__(self, cfg: DedupeConfig | None = None):
        self.cfg = cfg or DedupeConfig()
        # plate -> (last_ts, last_direction)
        self._recent: dict[str, tuple[float, CrossDirection]] = {}

    def is_duplicate(self, plate: str, direction: CrossDirection, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        self._evict(now)

        previous = self._recent.get(plate)
        if previous is None:
            return False
        last_ts, last_direction = previous
        elapsed = now - last_ts

        if direction == last_direction and elapsed < self.cfg.same_direction_seconds:
            return True
        if direction != last_direction and elapsed < self.cfg.reversal_seconds:
            return True
        return False

    def record(self, plate: str, direction: CrossDirection, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self._recent[plate] = (now, direction)
        if len(self._recent) > self.cfg.max_entries:
            self._evict(now, force=True)

    def check_and_record(self, plate: str, direction: CrossDirection, now: float | None = None) -> bool:
        """Returns True when the event should be emitted."""
        now = now if now is not None else time.time()
        if self.is_duplicate(plate, direction, now):
            # Refresh the timestamp: a vehicle still sitting in the ROI should
            # keep the window open rather than let it lapse and fire again.
            self._recent[plate] = (now, self._recent[plate][1])
            return False
        self.record(plate, direction, now)
        return True

    def _evict(self, now: float, force: bool = False) -> None:
        horizon = max(self.cfg.same_direction_seconds, self.cfg.reversal_seconds) * 2
        stale = [p for p, (ts, _) in self._recent.items() if now - ts > horizon]
        for plate in stale:
            del self._recent[plate]
        if force and len(self._recent) > self.cfg.max_entries:
            oldest = sorted(self._recent.items(), key=lambda kv: kv[1][0])
            for plate, _ in oldest[: len(self._recent) - self.cfg.max_entries]:
                del self._recent[plate]
