"""Per-stage timing and camera-capacity estimate for THIS box.

Phase 0 deliverable: you cannot promise a camera count or an accuracy figure
without measuring on the hardware that will be delivered.

    python scripts/benchmark.py --image backend/storage/events/some.jpg
    python scripts/benchmark.py --clip tests/fixtures/clips/day.mp4 --frames 200
    python scripts/benchmark.py --synthetic --runs 30

Prints p50/p95 per stage and the number of cameras this box supports at the
configured processing FPS. Ship that number in the handover document.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Must happen before numpy / cv2 / onnxruntime are imported.
from backend.app.ai.inference.threading import (  # noqa: E402
    apply_process_thread_budget, compute_budget, configure_opencv, physical_cores,
)

_BUDGET = compute_budget(1)
apply_process_thread_budget(_BUDGET)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.app.ai.inference.ops import crop as crop_box  # noqa: E402
from backend.app.ai.quality.plate_quality import enhance_for_ocr, score_plate  # noqa: E402
from backend.app.ai.registry import build_plate_detector, build_plate_recognizer, build_vehicle_detector  # noqa: E402
from backend.app.core.config import get_settings, model_config  # noqa: E402


def percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct)) if values else 0.0


def summarise(name: str, values: list[float]) -> str:
    if not values:
        return f"  {name:<20} {'not run':>10}"
    return (
        f"  {name:<20} {percentile(values, 50):8.1f} ms  "
        f"p95 {percentile(values, 95):7.1f} ms   n={len(values)}"
    )


def load_frames(args) -> list[np.ndarray]:
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"could not read {args.image}")
        return [frame] * args.runs

    if args.clip:
        capture = cv2.VideoCapture(str(args.clip))
        frames = []
        while len(frames) < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if not frames:
            raise SystemExit(f"no frames decoded from {args.clip}")
        return frames

    # Synthetic: measures the models, not the scene. Useful for comparing
    # backends and thread settings, useless for accuracy.
    return [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(args.runs)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path)
    source.add_argument("--clip", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--backend", choices=("auto", "cpu", "openvino"))
    parser.add_argument("--no-recognizer", action="store_true")
    args = parser.parse_args()

    if not (args.image or args.clip or args.synthetic):
        args.synthetic = True

    settings = get_settings()
    configure_opencv(_BUDGET)

    cores = physical_cores()
    print(f"\nBox: {cores} physical cores, intra_op={_BUDGET.intra_op}")

    def cfg(stage: str) -> dict:
        data = model_config(stage)
        if args.backend:
            data["backend"] = args.backend
        return data

    print("Loading models...")
    detector = build_vehicle_detector(cfg("vehicle_detector"), _BUDGET, settings.model_cache_dir)
    plate_detector = build_plate_detector(cfg("plate_detector"), _BUDGET, settings.model_cache_dir)
    recognizer = None
    if not args.no_recognizer:
        try:
            recognizer = build_plate_recognizer(cfg("plate_recognizer"), _BUDGET, settings.model_cache_dir)
        except Exception as exc:
            print(f"  recognizer unavailable ({exc}); timing detection stages only")

    for component in (detector, plate_detector, recognizer):
        if component is not None:
            component.warmup(2)

    frames = load_frames(args)
    print(f"Benchmarking {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}...\n")

    timings: dict[str, list[float]] = {"vehicle_detect": [], "plate_detect": [], "recognize": []}
    total_vehicles = total_plates = total_reads = 0

    for frame in frames:
        start = time.perf_counter()
        detections = detector.detect(frame)
        timings["vehicle_detect"].append((time.perf_counter() - start) * 1000)
        total_vehicles += len(detections)

        for detection in detections[:3]:
            vehicle_crop = crop_box(frame, detection.bbox)
            if vehicle_crop is None:
                continue
            start = time.perf_counter()
            candidates = plate_detector.detect(vehicle_crop, offset=detection.bbox[:2])
            timings["plate_detect"].append((time.perf_counter() - start) * 1000)
            total_plates += len(candidates)

            if recognizer is None or not candidates:
                continue
            plate_crop = crop_box(frame, candidates[0].bbox, pad=0.06)
            if plate_crop is None or score_plate(plate_crop).score < 0.2:
                continue
            start = time.perf_counter()
            read = recognizer.recognize(enhance_for_ocr(plate_crop))
            timings["recognize"].append((time.perf_counter() - start) * 1000)
            if read:
                total_reads += 1

    print("Per-stage latency (median):")
    for name, values in timings.items():
        print(summarise(name, values))

    print(f"\n  vehicles detected  {total_vehicles}")
    print(f"  plates detected    {total_plates}")
    print(f"  plates recognized  {total_reads}")

    # --- capacity ---------------------------------------------------------
    fps = float(settings.pipeline.get("processing_fps", 6.0))
    detect_interval = max(1, int(settings.pipeline.get("detect_interval", 2)))

    vehicle_ms = percentile(timings["vehicle_detect"], 50)
    plate_ms = percentile(timings["plate_detect"], 50)
    recognize_ms = percentile(timings["recognize"], 50)

    # Per processed frame, amortized: the detector runs 1-in-N, and the plate
    # stages are gated so they fire on a minority of frames. 0.35 is a
    # conservative occupancy for a society gate; measure yours from the worker
    # health endpoint once live.
    plate_duty = 0.35
    per_frame_ms = vehicle_ms / detect_interval + (plate_ms + recognize_ms) * plate_duty
    decode_ms = 5.0  # 720p sub-stream, measured separately per camera
    per_camera_ms_per_second = (per_frame_ms + decode_ms) * fps
    cameras = (1000.0 * max(1, cores - 1)) / max(per_camera_ms_per_second, 1e-6)

    print(f"\nAt {fps:.0f} processing fps, detect_interval={detect_interval}:")
    print(f"  ~{per_frame_ms:.1f} ms of inference per processed frame")
    print(f"  ~{per_camera_ms_per_second / 1000:.2f} cores per camera (inference + est. decode)")
    print(f"  => this box supports roughly {cameras:.1f} cameras")
    if recognizer is None:
        print("  (recognizer not included; expect slightly fewer)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
