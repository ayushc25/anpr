"""INT8-quantize the ONNX detectors for CPU inference.

Typically 1.8-3x faster on a CPU box for under 1% mAP loss. The FP32 artifact
is always kept: promotion to INT8 should be gated on an accuracy check
(scripts/eval_pipeline.py), never assumed.

    python scripts/quantize_int8.py --all
    python scripts/quantize_int8.py --model models/plate/plate_320.onnx

Dynamic quantization is used rather than static: it needs no calibration set,
and for the convolutional backbones here the difference against static PTQ is
small. Once real site footage exists, switch to static with ~300 frames from
that camera — the calibration data is what buys back the last of the accuracy.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def quantize(src: Path, dst: Path) -> bool:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.preprocess import quant_pre_process

    print(f"\n{src.name}  ({src.stat().st_size / 1e6:.1f} MB)")
    prepared = src.with_name(src.stem + ".prep.onnx")
    try:
        # Shape inference and constant folding first; skipping it is the usual
        # cause of "quantized model is slower than the original".
        quant_pre_process(str(src), str(prepared), skip_symbolic_shape=False)
        source = prepared
    except Exception as exc:
        print(f"  pre-process skipped ({type(exc).__name__}); quantizing directly")
        source = src

    try:
        quantize_dynamic(str(source), str(dst), weight_type=QuantType.QUInt8)
    except Exception as exc:
        print(f"  ! quantization failed: {exc}", file=sys.stderr)
        return False
    finally:
        prepared.unlink(missing_ok=True)

    print(f"  -> {dst.name}  ({dst.stat().st_size / 1e6:.1f} MB, "
          f"{100 * dst.stat().st_size / src.stat().st_size:.0f}% of FP32)")
    return True


def benchmark(path: Path, runs: int = 10) -> float:
    import numpy as np
    import onnxruntime as ort

    from backend.app.ai.inference.threading import compute_budget

    budget = compute_budget(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = budget.intra_op
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    spec = session.get_inputs()[0]
    shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in spec.shape)
    tensor = np.random.rand(*shape).astype(np.float32)
    names = [o.name for o in session.get_outputs()]

    for _ in range(3):
        session.run(names, {spec.name: tensor})
    start = time.perf_counter()
    for _ in range(runs):
        session.run(names, {spec.name: tensor})
    return (time.perf_counter() - start) * 1000 / runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--all", action="store_true", help="every .onnx under models/")
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    targets = list(args.model)
    if args.all:
        targets = [
            p for p in sorted((ROOT / "models").rglob("*.onnx"))
            if not p.name.endswith((".int8.onnx", ".prep.onnx"))
        ]
    if not targets:
        parser.print_help()
        return 1

    results = []
    for src in targets:
        src = src if src.is_absolute() else ROOT / src
        if not src.exists():
            print(f"! {src} not found", file=sys.stderr)
            continue
        dst = src.with_name(src.stem + ".int8.onnx")
        if not quantize(src, dst):
            continue
        if not args.no_bench:
            fp32, int8 = benchmark(src), benchmark(dst)
            speedup = fp32 / int8 if int8 else 0
            print(f"  fp32 {fp32:7.1f} ms   int8 {int8:7.1f} ms   {speedup:.2f}x")
            results.append((src.name, fp32, int8, speedup))

    if results:
        print("\nPoint configs/models.yaml at the .int8.onnx artifacts to use them.")
        print("Verify accuracy on real footage BEFORE promoting them:")
        print("  python scripts/benchmark.py --clip <a real gate clip>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
