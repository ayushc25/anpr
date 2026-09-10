"""Download a pretrained recognition head and its charset.

The detectors are exported from weights already in this repo; the recognizer
is not, so it is fetched. Run once per install:

    python scripts/fetch_recognizer.py --ppocr

Then swap `plate_recognizer.impl` to `ppocr_onnx` in configs/models.yaml.
Verify with `python -m backend.app.cli models check`.

This is a Phase 1 stopgap. The Phase 2 recognizer is an LPRNet/CRNN fine-tuned
on plates harvested from this site (scripts/harvest_dataset.py), which is what
takes accuracy from ~88% to ~96% on Indian plates.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "recognizer"

# PP-OCRv4 English mobile recognition head, already converted to ONNX.
# Pinned by URL rather than vendored: the artifact is ~10 MB and does not
# belong in git.
PPOCR_URL = "https://huggingface.co/OleehyO/paddleocr-onnx/resolve/main/en_PP-OCRv4_rec_infer.onnx"

# 0-9 A-Z, one per line, blank implied at index 0 by the CTC decoder.
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def download(url: str, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:
            while chunk := response.read(1 << 16):
                handle.write(chunk)
    except Exception as exc:
        print(f"  ! download failed: {exc}", file=sys.stderr)
        target.unlink(missing_ok=True)
        return False
    print(f"  -> {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def write_charset() -> Path:
    path = OUT_DIR / "charset.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(CHARSET), encoding="utf-8")
    print(f"  -> {path.relative_to(ROOT)} ({len(CHARSET)} characters)")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ppocr", action="store_true", help="fetch the PP-OCRv4 mobile recognition head")
    parser.add_argument("--url", help="fetch a recognition ONNX from a custom URL")
    parser.add_argument("--name", default="ppocr_rec_en.onnx")
    args = parser.parse_args()

    if not (args.ppocr or args.url):
        parser.print_help()
        return 1

    write_charset()
    ok = download(args.url or PPOCR_URL, OUT_DIR / args.name)
    if not ok:
        print(
            "\nNo network, or the URL moved. The pipeline still runs on the\n"
            "easyocr_legacy recognizer configured in configs/models.yaml —\n"
            "slower, but functional. Drop any CTC recognition ONNX into\n"
            f"{OUT_DIR.relative_to(ROOT)} and point models.yaml at it.",
            file=sys.stderr,
        )
        return 1

    print("\nNow set plate_recognizer.impl: ppocr_onnx in configs/models.yaml, then:")
    print("  python -m backend.app.cli models check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
