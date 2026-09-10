from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Detection, Track


class VehicleTracker(ABC):
    """Stateful, one instance per camera worker.

    ``update`` is called on EVERY processed frame, including frames where the
    detector did not run — pass an empty detection list and the tracker coasts
    on its motion model. That is what lets the detector run 1-in-N while
    line-crossing still sees a smooth trajectory.
    """

    name: str = "base"

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
        detector_ran: bool = True,
    ) -> list[Track]:
        """Returns the currently confirmed tracks.

        ``detector_ran`` distinguishes "the detector ran and found nothing"
        from "the detector was skipped this frame". They must not be treated
        alike: an empty list on a skipped frame would retire every track,
        which with detect_interval=2 kills each new track on the very next
        frame and produces zero tracks forever.
        """

    @abstractmethod
    def removed_track_ids(self) -> list[int]:
        """Track ids retired since the previous call. Drains the buffer — this
        is the signal the pipeline uses to finalize an event."""

    def reset(self) -> None:
        pass
