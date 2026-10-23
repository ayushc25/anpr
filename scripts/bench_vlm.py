"""Plate-recognizer benchmark on a LABELLED crop set: accuracy, unreadable rate,
latency percentiles and memory, for the VLM and (for comparison) PP-OCR.

This exists because the VLM's accuracy has NOT been measured: the feasibility
run only timed it. Plug the labelled set in here before quoting a number.

LABELS
------
A CSV with a header and two columns, relative image paths resolved against
--images:

    image,plate
    crops/0001.jpg,UP14FS3664
    crops/0002.jpg,HR29BG7381

Do NOT build this from the filenames under backend/storage/events: those are
named after what the SYSTEM read, which is circular, and some are wrong.

USAGE (the VLM and PP-OCR live in different venvs: onnxruntime-genai needs a
much newer onnxruntime than the pinned one, see backend/requirements-vlm.txt)

    backend\\venv-vlm\\Scripts\\python scripts\\bench_vlm.py \\
        --labels labels.csv --images <dir> --engine vlm --threads 6 --out vlm.csv

    backend\\venv\\Scripts\\python scripts\\bench_vlm.py \\
        --labels labels.csv --images <dir> --engine ppocr --threads 1 --out ppocr.csv

    # compare two finished runs on the same images (either venv)
    backend\\venv\\Scripts\\python scripts\\bench_vlm.py --compare vlm.csv ppocr.csv

Run both engines at the SAME thread count for a fair latency comparison; the
PP-OCR baseline in configs/models.yaml was measured at 1 thread.

Accuracy is reported two ways, because they answer different questions:
  raw      exactly what the recognizer returned, normalized
  cleaned  after the per-read grammar cleanup the validator applies
           (postprocess.resolve) - the number the PP-OCR baseline used
and with a 95% Wilson interval, because at ~200 crops a 3-point difference is
inside the noise and should not be reported as a win.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def read_labels(path: Path, images: Path) -> list[tuple[Path, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image = (images / row["image"]).resolve()
            plate = (row.get("plate") or "").strip()
            if not plate:
                continue
            rows.append((image, plate))
    if not rows:
        raise SystemExit(f"no labelled rows in {path}")
    return rows


def run(args) -> int:
    threads = max(1, args.threads)
    # Before numpy / onnxruntime are imported, like every other entrypoint.
    from backend.app.ai.inference.threading import ThreadBudget, apply_process_thread_budget

    budget = ThreadBudget(intra_op=threads)
    apply_process_thread_budget(budget)

    import cv2
    import numpy as np
    import psutil

    from backend.app.ai.plate_recognizer import postprocess
    from backend.app.core.config import model_config

    if args.engine == "vlm":
        from backend.app.ai.registry import build_vlm_recognizer

        recognizer = build_vlm_recognizer(model_config("vlm_recognizer"), budget)
    else:
        from backend.app.ai.registry import build_plate_recognizer

        recognizer = build_plate_recognizer(model_config("plate_recognizer"), budget)

    samples = read_labels(args.labels, args.images)
    if args.limit:
        samples = samples[: args.limit]

    process = psutil.Process()
    print(f"engine={args.engine} threads={threads} images={len(samples)}")
    print(f"warm-up {recognizer.warmup(1):.0f} ms; rss after load {process.memory_info().rss / 1e9:.2f} GB")

    rows, latencies = [], []
    peak_rss = process.memory_info().rss
    exact_raw = exact_clean = unreadable = errors = 0
    for index, (image_path, truth) in enumerate(samples, 1):
        truth_n = postprocess.normalize(truth)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  cannot read {image_path}", file=sys.stderr)
            errors += 1
            rows.append({"image": str(image_path), "truth": truth_n, "pred": "", "cleaned": "", "status": "unreadable_file", "ms": 0})
            continue
        started = time.perf_counter()
        try:
            read = recognizer.recognize(image)
            status = "ok" if read is not None else "none"
        except Exception as exc:  # a crash on one crop must not end the run
            read, status = None, f"error:{type(exc).__name__}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)
        peak_rss = max(peak_rss, process.memory_info().rss)

        pred = read.text if read is not None else ""
        cleaned = postprocess.resolve(pred).text if pred else ""
        if status.startswith("error"):
            errors += 1
        elif read is None:
            unreadable += 1
        exact_raw += int(pred == truth_n)
        exact_clean += int(cleaned == truth_n)
        rows.append({
            "image": str(image_path), "truth": truth_n, "pred": pred, "cleaned": cleaned,
            "status": status, "ms": round(elapsed_ms, 1),
            "conf": round(read.confidence, 4) if read is not None else "",
        })
        if index % 20 == 0:
            print(f"  {index}/{len(samples)}  raw {exact_raw / index:.1%}  avg {np.mean(latencies) / 1000:.2f}s")

    n = len(samples)
    lo_r, hi_r = wilson(exact_raw, n)
    lo_c, hi_c = wilson(exact_clean, n)
    lat = np.array(latencies) / 1000.0 if latencies else np.array([0.0])
    print()
    print(f"{args.engine} on {n} labelled crops")
    print(f"  exact (raw)      {exact_raw / n:6.1%}   95% CI [{lo_r:.1%}, {hi_r:.1%}]")
    print(f"  exact (cleaned)  {exact_clean / n:6.1%}   95% CI [{lo_c:.1%}, {hi_c:.1%}]")
    print(f"  unreadable/none  {unreadable / n:6.1%}   ({unreadable})")
    print(f"  errors           {errors / n:6.1%}   ({errors})")
    print(f"  latency  avg {lat.mean():.3f} s   p50 {np.percentile(lat, 50):.3f} s   p95 {np.percentile(lat, 95):.3f} s")
    print(f"  peak RSS {peak_rss / 1e9:.2f} GB (this process, includes the interpreter)")

    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) + (["conf"] if "conf" not in rows[0] else []))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {args.out}")
    recognizer.close()
    return 0


def compare(first: Path, second: Path) -> int:
    def load(path: Path) -> dict:
        with path.open(newline="", encoding="utf-8") as handle:
            return {row["image"]: row for row in csv.DictReader(handle)}

    a, b = load(first), load(second)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("the two result files share no images")
    print(f"{len(shared)} images in common ({first.name} vs {second.name})")
    for key, label in (("pred", "raw"), ("cleaned", "cleaned")):
        wins_a = sum(a[i][key] == a[i]["truth"] and b[i][key] != b[i]["truth"] for i in shared)
        wins_b = sum(b[i][key] == b[i]["truth"] and a[i][key] != a[i]["truth"] for i in shared)
        both = sum(a[i][key] == a[i]["truth"] and b[i][key] == b[i]["truth"] for i in shared)
        print(f"  {label:8s} both right {both}   only {first.name} {wins_a}   only {second.name} {wins_b}")
    print("  (a difference smaller than the CI half-width is noise, not a win)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, help="CSV: image,plate")
    parser.add_argument("--images", type=Path, default=Path("."), help="directory the CSV's image paths are relative to")
    parser.add_argument("--engine", choices=("vlm", "ppocr"), default="vlm")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, help="write per-crop results here")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A.csv", "B.csv"))
    args = parser.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.labels:
        parser.error("--labels is required (or use --compare)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
