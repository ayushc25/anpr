"""The inference backend abstraction.

This sits BELOW the model classes on purpose. "Which runtime" (ONNX Runtime
vs OpenVINO) and "which model" (YOLO vs RT-DETR, LPRNet vs PP-OCR) are two
independent axes, and keeping them independent is what lets us benchmark a
runtime swap on the customer's actual box without touching pipeline code.
"""
from __future__ import annotations

import logging
import platform
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from .threading import ThreadBudget

logger = logging.getLogger("anpr.ai.inference")

CPU = "cpu"
OPENVINO = "openvino"
AUTO = "auto"


class InferenceBackend(ABC):
    """Loads one model artifact and runs it. Knows nothing about vehicles."""

    name: str = "base"

    @abstractmethod
    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run one forward pass. Outputs are returned in graph output order."""

    @property
    @abstractmethod
    def input_names(self) -> list[str]: ...

    @property
    @abstractmethod
    def input_shapes(self) -> dict[str, tuple]: ...

    def run_single(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Convenience for the common single-input case."""
        return self.run({self.input_names[0]: tensor})

    def warmup(self, n: int = 2) -> float:
        """Run dummy inferences so the first real vehicle does not pay for
        lazy allocation or (on OpenVINO) graph compilation.

        Returns the last warm-up latency in milliseconds, which is logged at
        worker startup — it is the first number worth having in a support
        ticket about a slow box.
        """
        dummy = {}
        for name, shape in self.input_shapes.items():
            concrete = tuple(1 if (not isinstance(d, int) or d <= 0) else d for d in shape)
            dummy[name] = np.zeros(concrete, dtype=np.float32)
        elapsed = 0.0
        for _ in range(max(1, n)):
            start = time.perf_counter()
            try:
                self.run(dummy)
            except Exception:
                logger.warning("%s: warm-up pass failed (continuing)", self.name, exc_info=True)
                return 0.0
            elapsed = (time.perf_counter() - start) * 1000.0
        return elapsed

    def close(self) -> None:
        pass


def _cpu_vendor() -> str:
    """Best-effort CPU vendor string, used only to decide whether OpenVINO is
    worth trying. Never fatal — an unknown vendor just means we stay on ORT."""
    try:
        import cpuinfo  # type: ignore

        return str(cpuinfo.get_cpu_info().get("vendor_id_raw", ""))
    except Exception:
        pass
    try:
        if platform.system() == "Windows":
            return str(platform.processor())
        text = Path("/proc/cpuinfo").read_text(errors="ignore")
        for line in text.splitlines():
            if line.lower().startswith("vendor_id"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def openvino_ep_available() -> bool:
    try:
        import onnxruntime as ort

        return "OpenVINOExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def resolve_backend(requested: str) -> str:
    """Turn a config value (possibly ``auto``) into a concrete backend name."""
    if requested and requested != AUTO:
        return requested
    if platform.machine() not in ("x86_64", "AMD64"):
        return CPU
    vendor = _cpu_vendor().lower()
    if ("intel" in vendor or "genuineintel" in vendor) and openvino_ep_available():
        return OPENVINO
    return CPU


def build_backend(
    artifact: str | Path,
    backend: str = AUTO,
    threads: ThreadBudget | None = None,
    cache_dir: str | Path | None = None,
) -> InferenceBackend:
    """Factory. The only place that knows which backend classes exist."""
    artifact = Path(artifact)
    if not artifact.exists():
        raise FileNotFoundError(
            f"model artifact not found: {artifact}. Run scripts/export_onnx.py, "
            f"or point configs/models.yaml at an existing file."
        )
    threads = threads or ThreadBudget(intra_op=1)
    resolved = resolve_backend(backend)

    if resolved == OPENVINO:
        from .onnxrt_backend import OnnxRuntimeBackend

        try:
            return OnnxRuntimeBackend(artifact, threads, use_openvino=True, cache_dir=cache_dir)
        except Exception:
            # An unsupported op or a missing plugin should degrade to a slower
            # box, never to a dead camera.
            logger.warning(
                "OpenVINO EP failed for %s, falling back to the CPU provider",
                artifact.name,
                exc_info=True,
            )

    from .onnxrt_backend import OnnxRuntimeBackend

    return OnnxRuntimeBackend(artifact, threads, use_openvino=False)


def describe_backends() -> Sequence[str]:
    return (CPU, OPENVINO, AUTO)
