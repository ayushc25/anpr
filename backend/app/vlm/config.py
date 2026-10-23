"""Settings for the deferred-VLM flow. YAML: ``vlm:`` in configs/default.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..events.verification import VerificationConfig
from ..video.best_frames import SelectionConfig


@dataclass(frozen=True)
class VlmConfig:
    #: "socket": talk to the shared service process (production, any number of
    #: cameras). "local": load the model inside this process (tests, or one
    #: camera on a dev box). Never use "local" with several cameras: each would
    #: load its own copy of the model.
    mode: str = "socket"
    host: str = "127.0.0.1"
    port: int = 8765
    #: Interpreter the SUPERVISOR launches the service with. Empty = the
    #: supervisor's own. Needed because onnxruntime-genai cannot live in the
    #: detectors' environment; see backend/requirements-vlm.txt.
    python: str = ""

    #: Service-side queue bound. Beyond this the service answers "overflow".
    queue_size: int = 8
    #: Per-camera bound on jobs submitted and not yet answered. Beyond this
    #: ``submit`` refuses, and the vehicle becomes a class B overflow result.
    client_pending: int = 6
    #: A job that has waited this long in the SERVICE queue is discarded
    #: unrun (its camera has already given up).
    max_job_age_s: float = 90.0
    #: The camera worker's own deadline per job. Past it the vehicle becomes a
    #: class B result with reason "vlm_timeout". Generous: tens of seconds of
    #: latency is accepted by design; this only bounds a hung service.
    job_timeout_s: float = 120.0
    #: On shutdown, how long to wait for in-flight vehicles before emitting
    #: them as class B instead.
    shutdown_wait_s: float = 45.0
    #: Threads for the service's model. 0 = half the physical cores, min 1.
    threads: int = 0

    # -- best-frame selection ------------------------------------------------
    #: Crops sent per vehicle (the design range is 5-7).
    best_k: int = 6
    min_quality: float = 0.35
    #: 70 px matches OcrPolicy.min_plate_width_read, the readable-size floor
    #: the inline pipeline already uses (plate_quality's 45 px is the floor for
    #: "detectable", not "readable"). Observed on real traffic footage: crops of
    #: ~70x20 px are unreadable to the VLM and, queued, take the place of
    #: vehicles that were readable.
    min_width: int = 70
    #: Stop reading a vehicle's crops once this many agree exactly (0 = read
    #: all). The service is the system's bottleneck: ~2-3 s per crop.
    early_stop_agree: int = 3
    min_hamming: int = 6
    min_frame_gap: int = 2

    verification: VerificationConfig = field(default_factory=VerificationConfig)

    def selection(self) -> SelectionConfig:
        return SelectionConfig(
            best_k=self.best_k,
            min_quality=self.min_quality,
            min_width=self.min_width,
            min_hamming=self.min_hamming,
            min_frame_gap=self.min_frame_gap,
        )

    def __post_init__(self) -> None:
        if self.mode not in ("socket", "local"):
            raise ValueError(f"vlm.mode must be 'socket' or 'local', got {self.mode!r}")
        if not (1 <= self.best_k <= 12):
            raise ValueError("vlm.best_k must be between 1 and 12")
