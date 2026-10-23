"""Convenience test script to run the fine-tuned Indian ANPR pipeline on images or video clips.

Usage:
    backend\\venv\\Scripts\\python scripts/test_anpr.py --image path/to/image.jpg
    backend\\venv\\Scripts\\python scripts/test_anpr.py --video path/to/video.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.ai.anpr_pipeline import ANPREngine


def test_image(image_path: Path):
    if not image_path.exists():
        print(f"Error: Image '{image_path}' not found.")
        return

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"Error: Could not read image '{image_path}'.")
        return

    engine = ANPREngine.get_shared()
    plates, annotated = engine.read_image(bgr)

    print(f"\n--- ANPR Results for {image_path.name} ---")
    if not plates:
        print("No license plates detected.")
    else:
        for idx, p in enumerate(plates, 1):
            valid_flag = "VALID" if p.valid else "INVALID FORMAT"
            print(f"[{idx}] Plate: {p.text:<12} | Conf: {p.conf:.2%} | Status: {valid_flag} | Box: {p.box}")

    out_file = ROOT / "storage" / f"annotated_{image_path.name}"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), annotated)
    print(f"Saved annotated image to: {out_file}\n")


def test_video(video_path: Path):
    if not video_path.exists():
        print(f"Error: Video '{video_path}' not found.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'.")
        return

    engine = ANPREngine.get_shared()
    session = engine.new_session(video_path.name)
    frame_idx = 0
    t_sec = 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"\n--- Processing Video: {video_path.name} (FPS: {fps:.1f}) ---")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1
        t_sec = frame_idx / fps
        annotated, events, _ = session.process(frame, t_sec)
        for ev in events:
            print(f"-> Vehicle Logged: {ev.plate} at {ev.time_sec:.1f}s | Reads: {ev.agree}/{ev.reads} | Conf: {ev.confidence:.2%}")

    for ev in session.flush():
        print(f"-> Vehicle Settled: {ev.plate} at {ev.time_sec:.1f}s | Reads: {ev.agree}/{ev.reads} | Conf: {ev.confidence:.2%}")

    cap.release()
    print("Video processing complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Test Indian ANPR Pipeline")
    parser.add_argument("--image", type=Path, help="Path to image file")
    parser.add_argument("--video", type=Path, help="Path to video file")
    args = parser.parse_args()

    if args.image:
        test_image(args.image)
    elif args.video:
        test_video(args.video)
    else:
        # Default test if image exists
        sample = ROOT / "WhatsApp Image 2026-08-18 at 8.55.03 PM.jpeg"
        if sample.exists():
            test_image(sample)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
