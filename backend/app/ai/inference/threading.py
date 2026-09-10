"""Thread budget for a worker process.

The single most common silent performance bug on a CPU-only box: every
library (OpenMP, OpenCV, ONNX Runtime, OpenVINO) defaults to spawning one
thread per core. With four camera worker processes on a 4-core box that is
16 threads fighting over 4 cores, and the machine spends its time in context
switches. Measured cost of leaving this at defaults is a 2-4x regression.

``apply_process_thread_budget`` must be called BEFORE numpy, cv2 or
onnxruntime are imported, because the OpenMP variables are read at load time.
The camera worker entrypoint does this as its first statement.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_OMP_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class ThreadBudget:
    """How many threads one process may use for inference."""

    intra_op: int
    inter_op: int = 1

    @property
    def as_env(self) -> dict[str, str]:
        return {var: str(self.intra_op) for var in _OMP_VARS}


def physical_cores() -> int:
    """Physical cores, falling back to logical count.

    Hyperthreads help inference much less than physical cores do, and
    over-counting them is exactly how the oversubscription bug appears, so
    prefer the physical number when we can get it.
    """
    try:
        import psutil  # optional dependency

        count = psutil.cpu_count(logical=False)
        if count:
            return int(count)
    except Exception:
        pass
    return os.cpu_count() or 2


def compute_budget(n_processes: int, reserve: int = 1, cores: int | None = None) -> ThreadBudget:
    """Split the box between ``n_processes`` worker processes.

    ``reserve`` keeps a core back for the API process, Postgres and the OS —
    without it the workers starve the very services they are writing to.
    """
    total = cores if cores is not None else physical_cores()
    usable = max(1, total - reserve)
    intra = max(1, usable // max(1, n_processes))
    return ThreadBudget(intra_op=intra, inter_op=1)


def apply_process_thread_budget(budget: ThreadBudget) -> None:
    """Export the OpenMP/BLAS variables. Call before importing numpy or ORT."""
    for var, value in budget.as_env.items():
        os.environ[var] = value


def configure_opencv(budget: ThreadBudget) -> None:
    """OpenCV keeps its own pool, which otherwise competes with the ORT pool
    for the same cores. One thread is right: our OpenCV work (resize, warp,
    JPEG) is small compared to the forward pass."""
    try:
        import cv2

        cv2.setNumThreads(1 if budget.intra_op <= 2 else 2)
    except Exception:
        pass
