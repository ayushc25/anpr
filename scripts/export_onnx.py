"""Export the Ultralytics .pt models to ONNX for CPU serving.

Ultralytics (and therefore torch) is a BUILD-time dependency only. The serving
path loads ONNX through onnxruntime and never imports torch, which keeps ~2 GB
of wheels off the edge box and removes a large class of startup cost.

    python scripts/export_onnx.py --all
    python scripts/export_onnx.py --vehicle yolo26n.pt --imgsz 640
    python scripts/export_onnx.py --plate license-plate-finetune-v1l.pt --imgsz 320

Writes to models/<stage>/ and updates models/manifest.yaml with sha256 sums,
which the worker verifies at startup so a half-copied artifact fails loudly
instead of producing silent garbage.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MANIFEST = MODELS_DIR / "manifest.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(weights: Path, out_dir: Path, imgsz: int, simplify: bool = True, opset: int = 17) -> Path:
    from ultralytics import YOLO

    print(f"  loading {weights.name}")
    model = YOLO(str(weights))

    print(f"  exporting at {imgsz}x{imgsz} (opset {opset})")
    produced = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,   # a fixed shape lets ORT and OpenVINO plan allocations
        half=False,      # FP16 is slower than FP32 on CPU
    )
    produced = Path(produced)

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{weights.stem}_{imgsz}.onnx"
    shutil.move(str(produced), target)

    # Ultralytics can leave an external-weights sidecar next to the source
    # weights. Our exports are self-contained, so it is a stale ~100 MB
    # duplicate; clean it up rather than leaving it in the repo root.
    sidecar = produced.with_suffix(".onnx.data")
    if sidecar.exists():
        sidecar.unlink()
    print(f"  -> {target.relative_to(ROOT)}  ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def describe(path: Path) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {i.name: [d if isinstance(d, int) else -1 for d in i.shape] for i in session.get_inputs()}
    outputs = {o.name: [d if isinstance(d, int) else -1 for d in o.shape] for o in session.get_outputs()}
    return {"inputs": inputs, "outputs": outputs}


def update_manifest(stage: str, path: Path, imgsz: int, source: Path, meta: dict) -> None:
    import yaml

    manifest = {}
    if MANIFEST.exists():
        manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    manifest.setdefault("models", {})[stage] = {
        "artifact": str(path.relative_to(ROOT)).replace("\\", "/"),
        "source_weights": source.name,
        "input_size": [imgsz, imgsz],
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "io": meta,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"  manifest updated: {MANIFEST.relative_to(ROOT)}")


def run(stage: str, weights: Path, imgsz: int, opset: int) -> int:
    if not weights.exists():
        print(f"! {weights} not found", file=sys.stderr)
        return 1
    print(f"\n[{stage}] {weights.name}")
    if weights.stat().st_size > 40e6:
        # A -l or -x model will run, but not at a useful frame rate on a CPU
        # box. Say so at export time rather than letting it surface as "the
        # system is slow" after installation.
        print(
            f"  WARNING: {weights.name} is {weights.stat().st_size / 1e6:.0f} MB. "
            "Expect >150 ms/inference on CPU. Distil or retrain a nano variant "
            "before production (see docs/ROADMAP.md, Phase 2)."
        )
    target = export(weights, MODELS_DIR / stage, imgsz, opset=opset)
    update_manifest(stage, target, imgsz, weights, describe(target))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vehicle", type=Path, help="vehicle detector .pt")
    parser.add_argument("--plate", type=Path, help="plate detector .pt")
    parser.add_argument("--imgsz", type=int, help="override input size for both")
    parser.add_argument("--vehicle-imgsz", type=int, default=640)
    parser.add_argument("--plate-imgsz", type=int, default=320)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--all", action="store_true", help="export the repo's default weights")
    args = parser.parse_args()

    jobs: list[tuple[str, Path, int]] = []
    if args.all:
        jobs.append(("vehicle", ROOT / "yolo26n.pt", args.imgsz or args.vehicle_imgsz))
        jobs.append(("plate", ROOT / "license-plate-finetune-v1l.pt", args.imgsz or args.plate_imgsz))
    if args.vehicle:
        jobs.append(("vehicle", args.vehicle, args.imgsz or args.vehicle_imgsz))
    if args.plate:
        jobs.append(("plate", args.plate, args.imgsz or args.plate_imgsz))

    if not jobs:
        parser.print_help()
        return 1

    failures = sum(run(stage, weights, size, args.opset) for stage, weights, size in jobs)
    if failures:
        return 1

    print(
        "\nDone. Point configs/models.yaml at the new artifacts, then:\n"
        "  python -m backend.app.cli models check"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
