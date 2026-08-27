import itertools
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger("anpr.camera")

_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize_for_path(name: str) -> str:
    return _UNSAFE_PATH_CHARS.sub("_", name).strip("_") or "camera"

from ..config import (
    STORAGE_DIR,
    EVENT_DEDUPE_SECONDS,
    DETECTION_FRAME_INTERVAL,
    TRACK_IOU_THRESHOLD,
    TRACK_TIMEOUT_SECONDS,
    MIN_OCR_CONFIDENCE_TO_RECORD,
)
from ..database import SessionLocal
from .. import models
from . import detection

_track_id_seq = itertools.count(1)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _diag(b):
    return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5


def _match_score(track_bbox, track_type, det_bbox, det_type):
    """IOU match, with a centroid-distance fallback for fast on-screen
    motion between detection passes where consecutive boxes of the same
    vehicle no longer overlap (common at low sampling rates / high frame
    resolution). Different vehicle types only match on near-total overlap,
    tolerating an occasional car/truck misclassification wobble."""
    iou_score = _iou(track_bbox, det_bbox)
    if track_type != det_type and iou_score < 0.5:
        return 0.0
    if iou_score > 0:
        return iou_score

    diag = (_diag(track_bbox) + _diag(det_bbox)) / 2
    if diag <= 0:
        return 0.0
    ca, cb = _center(track_bbox), _center(det_bbox)
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    norm = dist / diag
    if norm < 0.75:
        return max(0.0, 1 - norm) * 0.9
    return 0.0


class VehicleTrack:
    """Tracks one physical vehicle across consecutive detection passes so
    its best OCR reading (highest confidence) can be used, rather than
    recording a separate noisy event per frame."""

    def __init__(self, track_id: int, bbox, vehicle_type: str, det_conf: float, now: float):
        self.id = track_id
        self.bbox = bbox
        self.vehicle_type = vehicle_type
        self.det_conf = det_conf
        self.created_at = now
        self.last_seen = now
        # per-candidate-text stats across frames: text -> [read_count, best_conf, best_crop_jpeg]
        self._votes: dict[str, list] = {}
        self._vehicle_color_votes: dict[str, int] = {}
        self._plate_color_votes: dict[str, int] = {}
        self.recorded = False

    def update(self, bbox, vehicle_type: str, det_conf: float, now: float):
        self.bbox = bbox
        self.vehicle_type = vehicle_type
        self.det_conf = max(self.det_conf, det_conf)
        self.last_seen = now

    def consider_plate(self, plate: Optional[str], conf: float, crop_jpeg: Optional[bytes]):
        if not plate:
            return
        entry = self._votes.setdefault(plate, [0, 0.0, None])
        entry[0] += 1
        if conf > entry[1]:
            entry[1] = conf
            entry[2] = crop_jpeg

    @property
    def best_plate(self) -> Optional[str]:
        best = self._best_entry()
        return best[0] if best else None

    @property
    def best_ocr_conf(self) -> float:
        best = self._best_entry()
        return best[1][1] if best else 0.0

    @property
    def best_crop(self) -> Optional[bytes]:
        best = self._best_entry()
        return best[1][2] if best else None

    @property
    def read_count(self) -> int:
        best = self._best_entry()
        return best[1][0] if best else 0

    def _best_entry(self):
        """Picks the plate text read most consistently across frames (ties
        broken by confidence) rather than whichever single frame happened
        to score highest - a lone high-confidence misread is far more
        common with OCR noise than a repeated one."""
        if not self._votes:
            return None
        return max(self._votes.items(), key=lambda kv: (kv[1][0], kv[1][1]))

    def consider_vehicle_color(self, color: Optional[str]):
        if color:
            self._vehicle_color_votes[color] = self._vehicle_color_votes.get(color, 0) + 1

    def consider_plate_color(self, color: Optional[str]):
        if color:
            self._plate_color_votes[color] = self._plate_color_votes.get(color, 0) + 1

    @property
    def best_vehicle_color(self) -> Optional[str]:
        if not self._vehicle_color_votes:
            return None
        return max(self._vehicle_color_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def best_plate_color(self) -> Optional[str]:
        if not self._plate_color_votes:
            return None
        return max(self._plate_color_votes.items(), key=lambda kv: kv[1])[0]

_workers: dict[int, "CameraWorker"] = {}
_workers_lock = threading.Lock()

STATUS_COLORS = {
    "registered": (46, 204, 113),
    "whitelist": (46, 204, 113),
    "blacklist": (0, 0, 231),
    "unknown": (0, 190, 246),
}


class CameraWorker:
    def __init__(self, camera_id: int, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.stop_event = threading.Event()
        self.latest_jpeg: Optional[bytes] = None
        self.frame_lock = threading.Lock()
        self.online = False
        self._recent_plates: dict[str, float] = {}
        self.tracks: dict[int, VehicleTrack] = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _set_online(self, online: bool):
        self.online = online
        try:
            db = SessionLocal()
            cam = db.query(models.Camera).get(self.camera_id)
            if cam:
                cam.is_online = online
                if online:
                    cam.last_seen_at = datetime.utcnow()
                db.commit()
            db.close()
        except Exception:
            pass

    def _run(self):
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                self._set_online(False)
                time.sleep(5)
                continue
            self._set_online(True)
            last_detect = 0.0
            last_boxes = []
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                now = time.time()
                if now - last_detect >= DETECTION_FRAME_INTERVAL:
                    last_detect = now
                    try:
                        last_boxes = self._process_frame(frame)
                    except Exception:
                        logger.exception("camera %s: detection pass failed", self.camera_id)
                        last_boxes = []
                self._draw_overlay(frame, last_boxes)
                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok2:
                    with self.frame_lock:
                        self.latest_jpeg = buf.tobytes()
            cap.release()
            self._set_online(False)
            if not self.stop_event.is_set():
                time.sleep(3)

    def _draw_overlay(self, frame, boxes):
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            color = STATUS_COLORS.get(b.get("status", "unknown"), (0, 190, 246))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = b.get("plate") or b.get("vehicle_type", "vehicle")
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _process_frame(self, frame):
        now = time.time()
        vehicles = detection.detect_vehicles(frame)
        # Match highest-confidence detections first so, when two detections
        # in the same frame could plausibly match the same track, the more
        # trustworthy one claims it.
        vehicles = sorted(vehicles, key=lambda v: v["conf"], reverse=True)
        boxes = []
        db = SessionLocal()
        claimed_track_ids: set[int] = set()

        try:
            for v in vehicles:
                track = self._match_track(v["bbox"], v["vehicle_type"], claimed_track_ids)
                if track is None:
                    track = VehicleTrack(next(_track_id_seq), v["bbox"], v["vehicle_type"], v["conf"], now)
                    self.tracks[track.id] = track
                else:
                    track.update(v["bbox"], v["vehicle_type"], v["conf"], now)
                claimed_track_ids.add(track.id)

                track.consider_vehicle_color(detection.get_vehicle_color(frame, v["bbox"]))

                plate, ocr_conf, plate_box_abs = detection.read_plate(frame, v["bbox"])
                if plate:
                    x1, y1, x2, y2 = v["bbox"]
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    ok, buf = cv2.imencode(".jpg", crop if crop.size else frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    track.consider_plate(plate, ocr_conf, buf.tobytes() if ok else None)
                    track.consider_plate_color(detection.get_plate_color(frame, plate_box_abs))

                entry = {"bbox": v["bbox"], "vehicle_type": v["vehicle_type"], "plate": track.best_plate}
                if track.best_plate:
                    status, _ = self._classify(db, track.best_plate)
                    entry["status"] = status.value
                boxes.append(entry)
        finally:
            db.close()

        self._finalize_stale_tracks(now)
        return boxes

    def _match_track(self, bbox, vehicle_type: str, exclude_ids: set) -> Optional["VehicleTrack"]:
        best_track, best_score = None, 0.0
        for track in self.tracks.values():
            if track.id in exclude_ids:
                continue
            score = _match_score(track.bbox, track.vehicle_type, bbox, vehicle_type)
            if score > best_score:
                best_track, best_score = track, score
        if best_track is not None and best_score >= TRACK_IOU_THRESHOLD:
            return best_track
        return None

    def _finalize_stale_tracks(self, now: float):
        stale_ids = [tid for tid, t in self.tracks.items() if now - t.last_seen > TRACK_TIMEOUT_SECONDS]
        for tid in stale_ids:
            track = self.tracks.pop(tid)
            self._record_track(track)

    def _record_track(self, track: "VehicleTrack"):
        if not track.best_plate:
            return
        # Require either a repeated (consensus) reading or a high-confidence
        # single read - a lone medium-confidence OCR hit is too often noise.
        confident_enough = track.read_count >= 2 or track.best_ocr_conf >= 0.55
        if not confident_enough or track.best_ocr_conf < MIN_OCR_CONFIDENCE_TO_RECORD:
            return
        if not self._should_record(track.best_plate):
            return

        db = SessionLocal()
        try:
            cam = db.query(models.Camera).get(self.camera_id)
            status, vehicle_row = self._classify(db, track.best_plate)
            image_rel = self._save_image(track.best_crop, track.best_plate, cam.name if cam else None)
            event = models.Event(
                plate_number=track.best_plate,
                vehicle_id=vehicle_row.id if vehicle_row else None,
                camera_id=self.camera_id,
                vehicle_type=track.vehicle_type,
                vehicle_color=track.best_vehicle_color,
                plate_color=track.best_plate_color,
                direction=cam.direction if cam else models.CameraDirection.both,
                status=status,
                confidence=track.det_conf,
                ocr_confidence=track.best_ocr_conf,
                image_path=image_rel,
            )
            db.add(event)
            db.commit()
        finally:
            db.close()

    def _classify(self, db, plate: str):
        vehicle = db.query(models.Vehicle).filter(models.Vehicle.plate_number == plate).first()
        if vehicle is None:
            return models.VehicleStatus.unknown, None
        if vehicle.status == models.VehicleStatus.blacklist:
            return models.VehicleStatus.blacklist, vehicle
        if vehicle.status == models.VehicleStatus.whitelist:
            return models.VehicleStatus.whitelist, vehicle
        return models.VehicleStatus.registered, vehicle

    def _should_record(self, plate: str) -> bool:
        now = time.time()
        last = self._recent_plates.get(plate)
        self._recent_plates[plate] = now
        if last is not None and (now - last) < EVENT_DEDUPE_SECONDS:
            return False
        return True

    def _save_image(self, crop_jpeg: Optional[bytes], plate: str, camera_name: Optional[str]) -> Optional[str]:
        """Saves under storage/events/<camera name>/<date>/<time>_<plate>.jpg
        so images are browsable on disk by camera and day, not just as one
        flat folder of thousands of files."""
        if not crop_jpeg:
            return None
        now = datetime.utcnow()
        cam_folder = _sanitize_for_path(camera_name or f"camera-{self.camera_id}")
        rel_dir = Path(cam_folder) / now.strftime("%Y-%m-%d")
        filename = f"{now.strftime('%H-%M-%S')}_{plate}.jpg"
        abs_dir = STORAGE_DIR / rel_dir
        try:
            abs_dir.mkdir(parents=True, exist_ok=True)
            (abs_dir / filename).write_bytes(crop_jpeg)
        except Exception:
            logger.exception("failed to save event image")
            return None
        return (rel_dir / filename).as_posix()

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.frame_lock:
            return self.latest_jpeg


def start_camera(camera_id: int, rtsp_url: str):
    with _workers_lock:
        if camera_id in _workers:
            _workers[camera_id].stop()
        worker = CameraWorker(camera_id, rtsp_url)
        _workers[camera_id] = worker
        worker.start()


def stop_camera(camera_id: int):
    with _workers_lock:
        worker = _workers.pop(camera_id, None)
    if worker:
        worker.stop()


def is_camera_online(camera_id: int) -> bool:
    worker = _workers.get(camera_id)
    return bool(worker and worker.online)


def start_all_cameras():
    db = SessionLocal()
    try:
        cams = db.query(models.Camera).filter(models.Camera.is_active == True).all()  # noqa: E712
        for cam in cams:
            start_camera(cam.id, cam.rtsp_url)
    finally:
        db.close()


def stop_all_cameras():
    with _workers_lock:
        ids = list(_workers.keys())
    for cid in ids:
        stop_camera(cid)


_PLACEHOLDER_JPEG: Optional[bytes] = None


def _placeholder_frame() -> bytes:
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is None:
        import numpy as np
        img = np.zeros((360, 640, 3), dtype="uint8")
        cv2.putText(img, "No signal", (220, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 120, 120), 2)
        ok, buf = cv2.imencode(".jpg", img)
        _PLACEHOLDER_JPEG = buf.tobytes() if ok else b""
    return _PLACEHOLDER_JPEG


def mjpeg_generator(camera_id: int):
    while True:
        worker = _workers.get(camera_id)
        frame = worker.get_latest_jpeg() if worker else None
        if frame is None:
            frame = _placeholder_frame()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.1)
