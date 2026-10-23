"""Fetch the Qwen3-VL-2B INT4 CPU model for the deferred-VLM pipeline.

    python scripts/fetch_vlm.py
    python scripts/fetch_vlm.py --from <dir that already holds the model>

Writes models/vlm/qwen3-vl-2b-int4-cpu/, which configs/models.yaml
(`vlm_recognizer`) already points at. Verify with
`python -m backend.app.cli models check`.

The model is a prebuilt ONNX Runtime GenAI package (decoder + vision encoder +
embedding, INT4 RTN block-32, CPU) from onnx-community/Qwen3-VL-2B-Instruct-ONNX,
~1.4 GB. It is not vendored in git. Pinned to a revision so a reinstall gets
the same weights that were benchmarked.

Needs `pip install huggingface_hub` for the download path (not for --from).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "vlm" / "qwen3-vl-2b-int4-cpu"

REPO = "onnx-community/Qwen3-VL-2B-Instruct-ONNX"
REVISION = "5dc9194ba42fa62c2f1440d98a71f7cd5280afe2"
SUBDIR = "onnxruntime/cpu_and_mobile/cpu-int4-rtn-block-32"

#: Files the runtime needs. genai_config.json names the three .onnx graphs; the
#: .data files are their external weights.
REQUIRED = (
    "genai_config.json",
    "text.onnx", "text.onnx.data",
    "vision.onnx", "vision.onnx.data",
    "embedding.onnx", "embedding.onnx.data",
    "tokenizer.json", "tokenizer_config.json",
    "processor_config.json", "config.json",
)


def missing(directory: Path) -> list[str]:
    return [name for name in REQUIRED if not (directory / name).exists()]


def from_local(source: Path) -> int:
    # Accept either the model directory itself or the snapshot root above it.
    candidate = source / SUBDIR if (source / SUBDIR).is_dir() else source
    absent = missing(candidate)
    if absent:
        print(f"{candidate} is not a complete model directory; missing: {absent}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in candidate.iterdir():
        if path.is_file():
            print(f"  copying {path.name}")
            shutil.copy2(path, OUT_DIR / path.name)
    return 0


def from_hub() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub   (or use --from <dir>)", file=sys.stderr)
        return 1
    print(f"fetching {REPO}@{REVISION[:8]} / {SUBDIR} (~1.4 GB)")
    staging = OUT_DIR.parent / ".staging"
    snapshot_download(
        REPO, revision=REVISION, allow_patterns=[f"{SUBDIR}/*"], local_dir=str(staging)
    )
    code = from_local(staging / SUBDIR)
    shutil.rmtree(staging, ignore_errors=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, help="copy from a directory that already has the model")
    args = parser.parse_args()

    code = from_local(args.source) if args.source else from_hub()
    if code:
        return code
    absent = missing(OUT_DIR)
    if absent:
        print(f"incomplete install, missing: {absent}", file=sys.stderr)
        return 1
    size = sum(f.stat().st_size for f in OUT_DIR.iterdir() if f.is_file()) / 1e9
    print(f"ok: {OUT_DIR} ({size:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
