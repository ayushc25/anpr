"""Download a pretrained recognition head and its dictionary.

The detectors are exported from weights already in this repo; the recognizer
is not, so it is fetched. Run once per install:

    python scripts/fetch_recognizer.py --ppocr

configs/models.yaml already points `plate_recognizer` at the files this
writes. Verify with `python -m backend.app.cli models check`.

This is a Phase 1 stopgap. The Phase 2 recognizer is an LPRNet/CRNN fine-tuned
on plates harvested from this site (scripts/harvest_dataset.py), which is what
takes accuracy from ~88% to ~96% on Indian plates.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "recognizer"

# PP-OCRv4 English mobile recognition head, already converted to ONNX, and the
# dictionary it was trained with. Pinned by URL and sha256 rather than vendored:
# the artifact is ~8 MB and does not belong in git. (The OleehyO/paddleocr-onnx
# URL this used to point at now returns 401.)
_HF = "https://huggingface.co/deepghs/paddleocr/resolve/main/rec/en_PP-OCRv4_rec"
PPOCR_MODEL = (f"{_HF}/model.onnx", "ppocr_rec_en.onnx",
               "870d81b658bb0ba59ae4a2aecf21f879e4921002be29d208cdcdb977af6e0959")
# The MODEL's dictionary: 95 printable-ASCII characters, in the order the
# output classes are numbered. It is not the plate alphabet and must not be
# replaced by one — see the note at the top of ppocr_onnx.py. Identical to
# PaddleOCR's ppocr/utils/en_dict.txt.
PPOCR_DICT = (f"{_HF}/dict.txt", "ppocr_en_dict.txt",
              "5662df9d2d03f0e8ca0d3b0649d6acbab904b6a14b3d3521463c71c37c668ce3")

# 0-9 A-Z, one per line: the plate alphabet, for recognizers trained on
# exactly that (LPRNet). Blank implied at index 0 by the CTC decoder.
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected_sha256: str | None = None) -> bool:
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
    if expected_sha256 and _sha256(target) != expected_sha256:
        # A moved or replaced artifact must not be served silently: a
        # different model with the same dictionary is exactly the failure
        # that produces confident garbage.
        print(f"  ! checksum mismatch for {target.name}; removed", file=sys.stderr)
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
    if args.url:
        ok = download(args.url, OUT_DIR / args.name)
    else:
        ok = all(download(url, OUT_DIR / name, digest) for url, name, digest in (PPOCR_MODEL, PPOCR_DICT))
    if not ok:
        print(
            "\nNo network, or the URL moved. Set plate_recognizer.impl back to\n"
            "easyocr_legacy in configs/models.yaml to keep running — slower,\n"
            "but functional. Or drop a CTC recognition ONNX and ITS OWN\n"
            f"dictionary into {OUT_DIR.relative_to(ROOT)} and point models.yaml at them.",
            file=sys.stderr,
        )
        return 1

    print("\nDone. Verify with:")
    print("  python -m backend.app.cli models check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
