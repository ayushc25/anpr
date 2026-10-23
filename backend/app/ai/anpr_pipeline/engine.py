"""ANPR engine: Indian plate detection (YOLO fine-tuned) + plate reading (fine-tuned PaddleOCR v2).

Adapted from the standalone anpr_backend into the application ecosystem.
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch  # Must load before paddle on Windows
from PIL import Image

from .plate_format import is_valid, repair

PROJECT = Path(__file__).resolve().parents[4]
DETECTOR = PROJECT / "models" / "plate" / "indian_plate_best.pt"
OCR_DIR = PROJECT / "models" / "plate_ocr_v2"

GREEN, AMBER, WHITE, BLACK = (60, 200, 60), (0, 170, 255), (255, 255, 255), (0, 0, 0)


@dataclass
class Settings:
    det_conf: float = 0.40        # detector confidence threshold
    min_plate_width: int = 70     # px; smaller plates are too far away to read reliably
    min_reads: int = 3            # reads of the winning plate needed before a vehicle is logged
    settle_sec: float = 2.0       # a vehicle is final once its plate is unseen for this long
    instant_reads: int = 5        # ...or as soon as one reading has this many reads
    instant_share: float = 0.7    # and at least this share of the vehicle's reads
    max_diff: int = 2             # readings differing by <= this many characters are the same vehicle
    repeat_sec: float = 60.0      # ignore the same plate again within this many seconds


@dataclass
class PlateRead:
    box: tuple
    text: str
    raw: str
    conf: float
    valid: bool
    crop: Image.Image


@dataclass
class VehicleEvent:
    plate: str
    time_sec: float
    reads: int
    agree: int
    confidence: float
    snapshot: Image.Image
    valid: bool = True
    raw_crop_bytes: Optional[bytes] = None


def _pad(x1, y1, x2, y2, w, h, frac=0.08):
    px, py = (x2 - x1) * frac, (y2 - y1) * frac
    return max(0, int(x1 - px)), max(0, int(y1 - py)), min(w, int(x2 + px)), min(h, int(y2 + py))


def draw_label(img, box, text, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    if not text:
        return
    scale = max(0.7, min(1.6, (x2 - x1) / 220))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    ty = y1 - 8 if y1 - th - 14 > 0 else y2 + th + 14
    cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 10, ty + base), color, -1)
    cv2.putText(img, text, (x1 + 5, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 2, cv2.LINE_AA)


class ANPREngine:
    _instance = None

    def __init__(self, settings: Settings | None = None):
        from paddleocr import TextRecognition
        from ultralytics import YOLO

        self.settings = settings or Settings()
        self.detector = YOLO(str(DETECTOR))
        self.ocr = TextRecognition(model_name="PP-OCRv6_small_rec", model_dir=str(OCR_DIR))

    @classmethod
    def get_shared(cls, settings: Settings | None = None) -> "ANPREngine":
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    def read_plate(self, crop: Image.Image):
        arr = np.asarray(crop.convert("RGB"))[:, :, ::-1].copy()
        res = self.ocr.predict(arr)[0]
        raw = res["rec_text"]
        text = repair(raw)
        return text, raw, float(res["rec_score"])

    def read_image(self, bgr):
        """Detect and read every plate in one photo. Returns (plates, annotated BGR)."""
        s = self.settings
        h, w = bgr.shape[:2]
        out = bgr.copy()
        plates = []
        res = self.detector.predict(bgr, conf=s.det_conf, device="cpu", verbose=False)[0]
        boxes = res.boxes.xyxy.tolist() if res.boxes is not None else []
        if not boxes and w / max(h, 1) >= 2.0:
            boxes = [(0, 0, w, h)]  # Photo is already a close-up of a plate
        for b in boxes:
            x1, y1, x2, y2 = _pad(*b, w, h)
            crop = Image.fromarray(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            text, raw, conf = self.read_plate(crop)
            valid = is_valid(text)
            plates.append(PlateRead((x1, y1, x2, y2), text, raw, conf, valid, crop))
            draw_label(out, (x1, y1, x2, y2), text or "?", GREEN if valid else AMBER)
        return plates, out

    def new_session(self, source_name="video"):
        return StreamSession(self, source_name)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class _Vehicle:
    """All readings believed to come from one vehicle."""

    def __init__(self, t_sec):
        self.reads = []          # (t_sec, plate, conf, width, crop)
        self.first = self.last = t_sec
        self.logged = None       # plate text once an event was emitted

    def best(self):
        score = {}
        for _, p, c, w, _ in self.reads:
            score[p] = score.get(p, 0) + c * min(w / 120, 1.5)
        if not score:
            return "", 0
        plate = max(score, key=score.get)
        return plate, sum(r[1] == plate for r in self.reads)


class StreamSession:
    """Per-stream state for video files and live cameras."""

    def __init__(self, engine: ANPREngine, source_name="stream"):
        self.engine = engine
        self.source = source_name
        self.open = []           # _Vehicle still being read
        self.last_logged = {}    # plate -> t_sec it was last logged
        self.frames = 0
        self.started = time.perf_counter()

    def _vehicle_for(self, plate, t_sec):
        s = self.engine.settings
        for v in self.open:
            best_p, _ = v.best()
            if t_sec - v.last <= s.settle_sec + 1 and edit_distance(plate, best_p) <= s.max_diff:
                return v
        v = _Vehicle(t_sec)
        self.open.append(v)
        return v

    def _emit(self, v):
        s = self.engine.settings
        plate, agree = v.best()
        if not plate or agree < s.min_reads:
            return None
        v.logged = plate
        last = self.last_logged.get(plate)
        self.last_logged[plate] = v.last
        if last is not None and v.first - last <= s.repeat_sec:
            return None
        same = [r for r in v.reads if r[1] == plate]
        best_read = max(same, key=lambda r: r[3])
        crop = best_read[4]
        avg_conf = sum(r[2] for r in same) / max(agree, 1)
        return VehicleEvent(plate, v.first, len(v.reads), agree, avg_conf, crop, is_valid(plate))

    def _settle(self, t_sec, force=False):
        s = self.engine.settings
        events = []
        for v in list(self.open):
            if v.logged is None:
                plate, agree = v.best()
                instant = agree >= s.instant_reads and agree >= s.instant_share * len(v.reads)
                if instant or force or t_sec - v.last > s.settle_sec:
                    ev = self._emit(v)
                    if ev:
                        events.append(ev)
            if force or t_sec - v.last > s.settle_sec:
                if v in self.open:
                    self.open.remove(v)
        return events

    def process(self, bgr, t_sec):
        """Run one frame. Returns (annotated BGR, list of new VehicleEvent, list of detected boxes info)."""
        s = self.engine.settings
        self.frames += 1
        h, w = bgr.shape[:2]
        out = bgr.copy()
        res = self.engine.detector.predict(bgr, conf=s.det_conf, device="cpu", verbose=False)[0]
        detected_boxes = []
        for b in (res.boxes.xyxy.tolist() if res.boxes is not None else []):
            box = _pad(*b, w, h)
            width = box[2] - box[0]
            if width < s.min_plate_width:
                draw_label(out, box, "too far", AMBER)
                continue
            crop = Image.fromarray(cv2.cvtColor(bgr[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2RGB))
            text, _, conf = self.engine.read_plate(crop)
            if not is_valid(text):
                draw_label(out, box, "reading...", AMBER)
                detected_boxes.append({"bbox": box, "plate": text, "conf": conf, "status": "reading"})
                continue
            v = self._vehicle_for(text, t_sec)
            v.reads.append((t_sec, text, conf, width, crop))
            v.last = t_sec
            shown = v.logged or v.best()[0]
            draw_label(out, box, shown, GREEN if v.logged else AMBER)
            detected_boxes.append({"bbox": box, "plate": shown, "conf": conf, "status": "logged" if v.logged else "reading"})
        return out, self._settle(t_sec), detected_boxes

    def flush(self):
        """Call when the stream ends: logs vehicles still waiting to settle."""
        return self._settle(float("inf"), force=True)

    @property
    def fps(self):
        return self.frames / max(time.perf_counter() - self.started, 1e-6)
