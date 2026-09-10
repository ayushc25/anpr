"""ANPR command line.

    python -m backend.app.cli worker --camera-id 3
    python -m backend.app.cli supervisor
    python -m backend.app.cli bench --clip tests/fixtures/clips/day.mp4
    python -m backend.app.cli models check

The thread budget MUST be applied before numpy, cv2 or onnxruntime are
imported, so this module imports almost nothing at the top level and pulls the
heavy modules in inside each command.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("anpr").setLevel(logging.DEBUG if verbose else logging.INFO)


def _apply_thread_budget(n_workers: int = 1) -> "ThreadBudget":  # noqa: F821
    """Set OMP/BLAS variables before the heavy imports happen."""
    from .ai.inference.threading import ThreadBudget, apply_process_thread_budget, compute_budget

    preset = os.environ.get("ANPR_INTRA_OP")
    budget = ThreadBudget(intra_op=int(preset)) if preset else compute_budget(n_workers)
    apply_process_thread_budget(budget)
    return budget


# ---------------------------------------------------------------- worker ----
def cmd_worker(args) -> int:
    budget = _apply_thread_budget(args.workers)

    from .core.config import (
        build_debug_config,
        build_dedupe_config,
        build_pipeline_config,
        build_validation_config,
        get_settings,
        model_config,
    )
    from .events.multi_frame_validator import SnapshotRegistry
    from .workers.camera_worker import CameraConfig, CameraWorker
    from .workers.ingest_client import build_ingest_client

    settings = get_settings()
    camera = _load_camera_config(args.camera_id)
    if camera is None:
        print(f"camera {args.camera_id} not found or disabled", file=sys.stderr)
        return 2

    registry = SnapshotRegistry(_load_registry_snapshot())

    worker = CameraWorker(
        camera=camera,
        pipeline_cfg=build_pipeline_config(camera.pipeline_overrides),
        validation_cfg=build_validation_config(),
        dedupe_cfg=build_dedupe_config(),
        model_cfgs={
            "vehicle_detector": model_config("vehicle_detector"),
            "tracker": model_config("tracker"),
            "plate_detector": model_config("plate_detector"),
            "plate_recognizer": model_config("plate_recognizer"),
        },
        threads=budget,
        ingest=build_ingest_client(
            "dry-run" if args.dry_run else args.ingest,
            settings.api_base_url,
            settings.worker_token,
            settings.resolve(".cache/spool") / f"camera-{camera.id}",
        ),
        registry=registry,
        cache_dir=settings.model_cache_dir,
        reader_kwargs={
            "reconnect_min": settings.reader.reconnect_min,
            "reconnect_max": settings.reader.reconnect_max,
            "read_timeout": settings.reader.read_timeout,
            "open_timeout_ms": settings.reader.open_timeout_ms,
        },
        # --debug forces artifact capture on for this worker regardless of
        # the YAML, which is how a clip is captured for the error dataset
        # without editing config on a production box.
        debug_cfg=(
            replace(build_debug_config(), enabled=True) if args.debug else build_debug_config()
        ),
        debug_root=settings.resolve(build_debug_config().dir),
    )
    worker.run()
    return 0


def _load_camera_config(camera_id: int):
    """Read one camera's configuration. The only database access a worker
    makes, and it happens once at startup, never in the loop."""
    from .db.session import SessionLocal
    from .models.camera import Camera, CameraZone
    from .workers.camera_worker import CameraConfig

    session = SessionLocal()
    try:
        camera = session.get(Camera, camera_id)
        if camera is None or not camera.is_active:
            return None
        zones = session.query(CameraZone).filter(
            CameraZone.camera_id == camera_id, CameraZone.is_active.is_(True)
        ).all()
        roi_zones = [z for z in zones if z.kind == "roi"]
        # The read zone is carried as an roi-kind zone with a reserved name
        # rather than its own ZoneKind. ZoneKind is a native Postgres enum, so
        # a new member needs an ALTER TYPE migration; that is worth doing, but
        # it should not block calibrating a gate. Any roi zone named "read" /
        # "read_zone" is the read zone, and the first unnamed one is the ROI.
        read_names = {"read", "read_zone", "readzone"}
        read_zone = next(
            (z.geometry for z in roi_zones if (z.name or "").strip().lower() in read_names), []
        ) or []
        roi = next(
            (z.geometry for z in roi_zones if (z.name or "").strip().lower() not in read_names), []
        ) or []
        line_zone = next((z for z in zones if z.kind == "line"), None)
        return CameraConfig(
            id=camera.id,
            code=camera.code or f"cam{camera.id}",
            name=camera.name,
            rtsp_url=camera.rtsp_url,
            rtsp_url_sub=camera.rtsp_url_sub or "",
            direction=getattr(camera.direction, "value", camera.direction) or "both",
            processing_fps=camera.processing_fps or 6.0,
            roi=roi,
            read_zone=read_zone,
            line=(line_zone.geometry if line_zone else []) or [],
            line_forward=(line_zone.direction_hint if line_zone else "in") or "in",
            pipeline_overrides={"detect_interval": camera.detect_interval} if camera.detect_interval else {},
        )
    finally:
        session.close()


def _load_registry_snapshot() -> dict[str, str]:
    from .db.session import SessionLocal
    from .models.vehicle import Vehicle

    session = SessionLocal()
    try:
        rows = session.query(Vehicle.plate_number, Vehicle.status).filter(Vehicle.is_active.is_(True)).all()
        return {plate: getattr(status, "value", status) for plate, status in rows}
    except Exception:
        logging.getLogger("anpr.cli").warning("could not load the registry snapshot", exc_info=True)
        return {}
    finally:
        session.close()


# ------------------------------------------------------------ supervisor ----
def cmd_supervisor(args) -> int:
    from .core.config import get_settings
    from .workers.supervisor import Supervisor

    settings = get_settings()
    Supervisor(
        reconcile_interval=args.interval,
        reserve_cores=settings.runtime.reserve_cores,
        dry_run=args.dry_run,
    ).run()
    return 0


# ----------------------------------------------------------------- models ---
def cmd_models(args) -> int:
    _apply_thread_budget(1)
    from .ai.registry import available
    from .core.config import model_config

    if args.action == "list":
        for stage in ("vehicle_detector", "tracker", "plate_detector", "plate_recognizer"):
            print(f"{stage}:")
            for name in available(stage):
                marker = " *" if model_config(stage).get("impl") == name else "  "
                print(f"  {marker} {name}")
        return 0

    # check
    from pathlib import Path

    ok = True
    for stage in ("vehicle_detector", "plate_detector", "plate_recognizer"):
        cfg = model_config(stage)
        artifact = cfg.get("artifact")
        if not artifact:
            continue
        exists = Path(artifact).exists()
        ok &= exists
        size = f"{Path(artifact).stat().st_size / 1e6:.1f} MB" if exists else "MISSING"
        print(f"{'ok ' if exists else 'MISS'}  {stage:<18} {cfg.get('impl'):<16} {size:>10}  {artifact}")
    if not ok:
        print("\nRun scripts/export_onnx.py to produce the missing artifacts.", file=sys.stderr)
    return 0 if ok else 1


# ------------------------------------------------------------------- main ---
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anpr", description="ANPR edge system")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    worker = sub.add_parser("worker", help="run one camera worker in this process")
    worker.add_argument("--camera-id", type=int, required=True)
    worker.add_argument("--workers", type=int, default=1, help="total workers on this box (thread budget)")
    worker.add_argument("--ingest", choices=("http", "direct"), default="direct")
    worker.add_argument("--dry-run", action="store_true", help="log events instead of storing them")
    worker.add_argument(
        "--debug", action="store_true",
        help="save vehicle/plate crops, per-character probabilities and the final "
             "verdict for EVERY OCR call, for building an error dataset",
    )
    worker.set_defaults(func=cmd_worker)

    supervisor = sub.add_parser("supervisor", help="spawn and supervise all camera workers")
    supervisor.add_argument("--interval", type=float, default=5.0)
    supervisor.add_argument("--dry-run", action="store_true")
    supervisor.set_defaults(func=cmd_supervisor)

    models = sub.add_parser("models", help="inspect configured models")
    models.add_argument("action", choices=("list", "check"), default="check", nargs="?")
    models.set_defaults(func=cmd_models)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
