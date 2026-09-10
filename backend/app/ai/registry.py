"""Config string -> implementation class.

The single point of coupling between configuration and code. Nothing else in
the codebase imports a concrete detector, tracker or recognizer, which is what
makes a model swap a one-line YAML change.

Implementations are imported lazily inside the builders so that a box without
easyocr installed can still build an ONNX recognizer, and so importing this
module in a test does not drag in torch.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from .inference.threading import ThreadBudget
from .plate_detector.base import PlateDetector
from .plate_recognizer.base import PlateRecognizer
from .vehicle_detector.base import VehicleDetector
from .vehicle_tracker.base import VehicleTracker

logger = logging.getLogger("anpr.ai.registry")


class ConfigError(ValueError):
    """A models.yaml that names something that does not exist."""


def _load_vehicle_detectors() -> dict[str, Callable[..., VehicleDetector]]:
    from .vehicle_detector.yolo_onnx import NullVehicleDetector, YoloOnnxVehicleDetector

    return {"yolo_onnx": YoloOnnxVehicleDetector, "null": NullVehicleDetector}


def _load_trackers() -> dict[str, Callable[..., VehicleTracker]]:
    from .vehicle_tracker.bytetrack import ByteTracker
    from .vehicle_tracker.iou_tracker import IouTracker

    return {"bytetrack": ByteTracker, "iou_tracker": IouTracker}


def _load_plate_detectors() -> dict[str, Callable[..., PlateDetector]]:
    from .plate_detector.yolo_plate_onnx import YoloPlateDetector

    return {"yolo_plate_onnx": YoloPlateDetector}


def _load_recognizers() -> dict[str, Callable[..., PlateRecognizer]]:
    registry: dict[str, Callable[..., PlateRecognizer]] = {}
    from .plate_recognizer.lprnet_onnx import LprnetOnnxRecognizer
    from .plate_recognizer.ppocr_onnx import PpOcrOnnxRecognizer

    registry["lprnet_onnx"] = LprnetOnnxRecognizer
    registry["ppocr_onnx"] = PpOcrOnnxRecognizer
    try:
        from .plate_recognizer.easyocr_legacy import EasyOcrRecognizer

        registry["easyocr_legacy"] = EasyOcrRecognizer
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("easyocr not installed; easyocr_legacy unavailable")
    return registry


_LOADERS = {
    "vehicle_detector": _load_vehicle_detectors,
    "tracker": _load_trackers,
    "plate_detector": _load_plate_detectors,
    "plate_recognizer": _load_recognizers,
}


def available(stage: str) -> list[str]:
    """Implementation names registered for a stage — used by the error
    messages and by `anpr models list`."""
    return sorted(_LOADERS[stage]())


def _build(stage: str, cfg: dict[str, Any], extra: dict[str, Any]) -> Any:
    implementations = _LOADERS[stage]()
    impl = cfg.get("impl")
    if not impl:
        raise ConfigError(f"{stage}.impl is required; known: {sorted(implementations)}")
    try:
        cls = implementations[impl]
    except KeyError:
        raise ConfigError(
            f"unknown {stage}.impl={impl!r}; known: {sorted(implementations)}"
        ) from None

    kwargs = {k: v for k, v in cfg.items() if k != "impl"}
    kwargs.update(extra)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{stage}.impl={impl!r} rejected its config: {exc}") from exc


def build_vehicle_detector(
    cfg: dict, threads: ThreadBudget | None = None, cache_dir: str | Path | None = None
) -> VehicleDetector:
    return _build("vehicle_detector", cfg, {"threads": threads, "cache_dir": cache_dir})


def build_tracker(cfg: dict) -> VehicleTracker:
    return _build("tracker", cfg, {})


def build_plate_detector(
    cfg: dict, threads: ThreadBudget | None = None, cache_dir: str | Path | None = None
) -> PlateDetector:
    return _build("plate_detector", cfg, {"threads": threads, "cache_dir": cache_dir})


def build_plate_recognizer(
    cfg: dict, threads: ThreadBudget | None = None, cache_dir: str | Path | None = None
) -> PlateRecognizer:
    return _build("plate_recognizer", cfg, {"threads": threads, "cache_dir": cache_dir})
