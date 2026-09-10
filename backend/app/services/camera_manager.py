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
    MAX_PLATE_READS_PER_PASS,
    MIN_VEHICLE_AREA_RATIO,
    PREVIEW_MAX_WIDTH,
    STORAGE_DIR,
    EVENT_DEDUPE_SECONDS,
    DETECTION_FRAME_INTERVAL,
    TRACK_IOU_THRESHOLD,
    TRACK_TIMEOUT_SECONDS,
    MIN_OCR_CONFIDENCE_TO_RECORD,
)
from ..database import SessionLocal
from .. import models
from ..ai.plate_recognizer import postprocess
from ..repositories.vehicle_repo import VehicleRepository
from . import detection

_track_id_seq = itertools.count(1)


def _event_direction(cam):
    """Map a camera's configured role onto an event direction."""
    value = getattr(getattr(cam, "direction", None), "value", None)
    if value == "in":
        return models.EventDirection.in_
    if value == "out":
        return models.EventDirection.out_
    return models.EventDirection.unknown


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
        #: The events row this track owns, so a later, better read updates it
        #: instead of inserting a second row for the same vehicle.
        self.event_id: Optional[int] = None
        self.recorded_score: float = 0.0

    def update(self, bbox, vehicle_type: str, det_conf: float, now: float):
        self.bbox = bbox
        self.vehicle_type = vehicle_type
        self.det_conf = max(self.det_conf, det_conf)
        self.last_seen = now

    def consider_plate(self, plate: Optional[str], conf: float, crop_jpeg: Optional[bytes]):
        """Record one frame's reading of this track's plate.

        The raw recognizer string never becomes the vote directly. It goes
        through the same grammar cleanup the validated pipeline uses, because
        the two failures that string carries are both fixable here and by
        nothing downstream:

        * a glyph read off the hologram, the emblem or the border, glued to
          the front of the number — the ``TUP1GEJ0364`` for ``UP1GEJ0364``
          class of misread;
        * a digit/letter confusion at a position where the plate format leaves
          no doubt which type belongs there.

        Cleaning before the vote also merges ballots: two frames that were
        wrong in different ways but resolve to the same plate now agree
        instead of splitting the track three ways.
        """
        if not plate:
            return
        text = postprocess.normalize(plate)
        if len(text) < postprocess.MIN_LEN:
            return
        text = postprocess.resolve(text).text
        entry = self._votes.setdefault(text, [0, 0.0, None])
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
    def score(self) -> float:
        """How good this track's plate is. Confidence, nudged up by repeated
        agreement — two frames saying the same thing beats one frame saying it
        slightly more confidently."""
        return self.best_ocr_conf + 0.05 * min(max(self.read_count - 1, 0), 4)

    @property
    def read_count(self) -> int:
        best = self._best_entry()
        return best[1][0] if best else 0

    def _best_entry(self):
        """Picks the plate text read most consistently across frames (ties
        broken by confidence) rather than whichever single frame happened
        to score highest - a lone high-confidence misread is far more
        common with OCR noise than a repeated one.

        Agreement is weighted by how plate-shaped the text is. Repetition on
        its own is not evidence: the same crop, read the same wrong way five
        frames running, agrees with itself perfectly. A string that parses as
        a real plate carries a full vote; one that is merely plate-ish carries
        0.6; ``160406937`` carries 0.35 and needs three times the agreement to
        beat a single valid read.
        """
        if not self._votes:
            return None
        # A read of half a plate is struck from the ballot outright while any
        # whole read survives, rather than merely weighted down. The occlusion
        # lasts as many frames as it lasts, so the fragment accumulates
        # agreement the whole plate never gets a chance to match.
        candidates = {t: v for t, v in self._votes.items() if not postprocess.is_fragment(t)}
        if not candidates:
            candidates = self._votes
        return max(
            candidates.items(),
            key=lambda kv: (kv[1][0] * postprocess.grammar_factor(kv[0]), kv[1][1]),
        )

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
        # plate -> (last_seen_ts, event_id, score)
        self._recent_plates: dict[str, tuple] = {}
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

    @property
    def _is_file_source(self) -> bool:
        """A local video file, not a network stream."""
        url = (self.rtsp_url or "").lower()
        if url.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")):
            return False
        try:
            return Path(self.rtsp_url).is_file()
        except OSError:
            return False

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

            # Play a FILE at its native rate. Without this the reader chews
            # through a two-minute clip in ~29 s and immediately replays it,
            # so the same cars arrive again and again — which looks exactly
            # like duplicate events in the table, but is the clip looping.
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if self._is_file_source else 0.0
            frame_period = (1.0 / source_fps) if source_fps > 1 else 0.0
            next_frame_at = time.monotonic()

            while not self.stop_event.is_set():
                if frame_period:
                    next_frame_at += frame_period
                    delay = next_frame_at - time.monotonic()
                    if delay > 0:
                        if self.stop_event.wait(delay):
                            break
                    else:
                        next_frame_at = time.monotonic()

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
                # Encoding 1080p costs ~64 ms per frame. Downscaling the
                # preview cuts that by an order of magnitude and nobody can
                # see the difference in a browser panel.
                preview = frame
                if frame.shape[1] > PREVIEW_MAX_WIDTH:
                    scale = PREVIEW_MAX_WIDTH / frame.shape[1]
                    preview = cv2.resize(frame, (PREVIEW_MAX_WIDTH, int(frame.shape[0] * scale)),
                                         interpolation=cv2.INTER_AREA)
                ok2, buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
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
        # Largest first for the plate budget below: at a gate the nearest
        # vehicle is the one whose plate is legible.
        vehicles = sorted(
            vehicles,
            key=lambda v: (v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1]),
            reverse=True,
        )
        boxes = []
        db = SessionLocal()
        claimed_track_ids: set[int] = set()

        try:
            frame_area = frame.shape[0] * frame.shape[1]
            plate_budget = MAX_PLATE_READS_PER_PASS

            for v in vehicles:
                track = self._match_track(v["bbox"], v["vehicle_type"], claimed_track_ids)
                if track is None:
                    track = VehicleTrack(next(_track_id_seq), v["bbox"], v["vehicle_type"], v["conf"], now)
                    self.tracks[track.id] = track
                else:
                    track.update(v["bbox"], v["vehicle_type"], v["conf"], now)
                claimed_track_ids.add(track.id)

                track.consider_vehicle_color(detection.get_vehicle_color(frame, v["bbox"]))

                # The plate stages cost ~400 ms per vehicle; the vehicle
                # detector costs ~120 ms for the whole frame. Running them on
                # every vehicle in shot is what made a 2-minute clip take
                # minutes. Spend the budget on vehicles that can actually be
                # read: close enough for a legible plate, and not already
                # settled.
                x1, y1, x2, y2 = v["bbox"]
                area_ratio = ((x2 - x1) * (y2 - y1)) / max(frame_area, 1)
                worth_reading = (
                    plate_budget > 0
                    and not track.recorded
                    and area_ratio >= MIN_VEHICLE_AREA_RATIO
                    and v["vehicle_type"] not in ("bicycle",)
                )

                plate, ocr_conf, plate_box_abs = (None, 0.0, None)
                if worth_reading:
                    plate_budget -= 1
                    plate, ocr_conf, plate_box_abs = detection.read_plate(frame, v["bbox"])
                if plate:
                    x1, y1, x2, y2 = v["bbox"]
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    ok, buf = cv2.imencode(".jpg", crop if crop.size else frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    track.consider_plate(plate, ocr_conf, buf.tobytes() if ok else None)
                    track.consider_plate_color(detection.get_plate_color(frame, plate_box_abs))

                # Record as soon as the evidence is good enough, rather than
                # waiting for the track to die. On busy footage a track is
                # continuously re-matched to whichever vehicle is nearest and
                # never goes stale, so death-triggered recording emits nothing
                # at all — observed as detections on Live View with an empty
                # events page. The dedupe window still prevents duplicates.
                self._maybe_record_live(track)

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

    def _has_enough_evidence(self, track: "VehicleTrack") -> bool:
        """Either a repeated reading or one strong single read. A lone
        medium-confidence OCR hit is too often noise."""
        if not track.best_plate:
            return False
        if postprocess.is_fragment(track.best_plate):
            # Half a plate — occluded by the vehicle in front, or clipped at
            # the edge of the view. Confidence does not catch this: the
            # recognizer read those characters perfectly, there were just
            # more of them it never saw. Emitting the fragment is worse than
            # waiting, because the track stays alive: the moment the vehicle
            # clears the obstruction a whole read arrives and this same track
            # writes its row then.
            logger.debug(
                "camera %s: holding track %s, %r is a fragment",
                self.camera_id, track.id, track.best_plate,
            )
            return False
        if track.best_ocr_conf < MIN_OCR_CONFIDENCE_TO_RECORD:
            return False
        return track.read_count >= 2 or track.best_ocr_conf >= 0.55

    def _maybe_record_live(self, track: "VehicleTrack"):
        """Emit — or improve — this track's event as its plate settles.

        A track writes at most ONE row. While it is alive, a better read
        replaces the values on that row rather than adding another, so the
        events table shows one entry per vehicle carrying its best reading.
        """
        if not self._has_enough_evidence(track):
            return
        if track.event_id is not None and track.score <= track.recorded_score + 1e-9:
            return  # nothing better to say
        self._record_track(track)

    def _record_track(self, track: "VehicleTrack"):
        if not track.best_plate or not self._has_enough_evidence(track):
            return

        action, event_id = self._resolve_target(track)
        if action == "skip":
            return

        db = SessionLocal()
        try:
            cam = db.query(models.Camera).get(self.camera_id)
            status, vehicle_row = self._classify(db, track.best_plate)
            # Once a read resolves to a registered vehicle, store the
            # REGISTRY's spelling, not the recognizer's. Otherwise the same
            # car appears in the log under every way the camera has ever
            # misspelled it, and a search for the resident's actual plate
            # finds none of those passes.
            plate = vehicle_row.plate_number if vehicle_row else track.best_plate

            if action == "update" and event_id is not None:
                event = db.query(models.Event).get(event_id)
                if event is None:
                    action = "insert"          # row was deleted under us
                else:
                    # Replace the stored reading with the better one, and the
                    # image with the frame that produced it.
                    image_rel = self._save_image(track.best_crop, plate,
                                                 cam.name if cam else None)
                    event.plate_number = plate
                    event.plate_raw = track.best_plate
                    event.vehicle_id = vehicle_row.id if vehicle_row else None
                    event.status = status
                    event.vehicle_type = track.vehicle_type
                    event.vehicle_color = track.best_vehicle_color or event.vehicle_color
                    event.plate_color = track.best_plate_color or event.plate_color
                    event.detect_confidence = float(track.det_conf)
                    event.plate_confidence = float(track.best_ocr_conf)
                    event.read_count = track.read_count
                    if image_rel:
                        event.vehicle_image_path = image_rel
                    db.commit()
                    track.recorded = True
                    track.recorded_score = track.score
                    logger.info(
                        "camera %s: event %s improved (reads=%d conf=%.2f)",
                        self.camera_id, track.best_plate, track.read_count, track.best_ocr_conf,
                    )
                    return

            image_rel = self._save_image(track.best_crop, plate,
                                         cam.name if cam else None)
            event = models.Event(
                plate_number=plate,
                # What the recognizer actually said, kept alongside the
                # registry's spelling. Without it a match that snapped to the
                # wrong resident leaves no trace of what was really read, and
                # nobody can tell an exact hit from an inferred one.
                plate_raw=track.best_plate,
                vehicle_id=vehicle_row.id if vehicle_row else None,
                camera_id=self.camera_id,
                vehicle_type=track.vehicle_type,
                vehicle_color=track.best_vehicle_color,
                plate_color=track.best_plate_color,
                # A camera set to "both" tells us nothing about this pass;
                # record it honestly rather than counting it as an entry.
                direction=_event_direction(cam),
                status=status,
                detect_confidence=float(track.det_conf),
                plate_confidence=float(track.best_ocr_conf),
                read_count=track.read_count,
                vehicle_image_path=image_rel,
            )
            db.add(event)
            db.commit()
            track.event_id = event.id
            track.recorded = True
            track.recorded_score = track.score
            # Now that the row exists, remember its id against this plate so a
            # re-acquired track of the same vehicle updates it too.
            self._recent_plates[track.best_plate] = (time.time(), event.id, track.score)
            logger.info(
                "camera %s: event %s (reads=%d conf=%.2f)",
                self.camera_id, track.best_plate, track.read_count, track.best_ocr_conf,
            )
        finally:
            db.close()

    def _classify(self, db, plate: str):
        """Resolve a read to a registered vehicle and its effective status.

        Delegated to VehicleRepository rather than done here. The exact-match
        query this used to run could only ever succeed when the recognizer
        spelled the plate perfectly, which is the one thing it cannot promise
        — so a resident whose plate came back one character out was logged as
        an unknown vehicle. The repository tries the exact plate, then a
        learned alias, then a single confusion, then the nearest registered
        plate; and unlike the query it replaces, it honours is_active and the
        permit dates instead of reporting an expired registration as current.
        """
        repo = VehicleRepository(db)
        vehicle, _method = repo.match(plate)
        return repo.resolve_status(vehicle, datetime.utcnow()), vehicle

    def _resolve_target(self, track: "VehicleTrack"):
        """Decide whether this track's plate is a new vehicle, or a better read
        of one already recorded.

        Returns (action, event_id) where action is "insert", "update" or
        "skip". Matching is by SIMILARITY, not equality: a weak recognizer
        rarely spells the same vehicle the same way twice, and exact matching
        is why one car produced a row per read.
        """
        now = time.time()
        plate, score = track.best_plate, track.score

        # This track already owns a row.
        if track.event_id is not None:
            self._recent_plates[plate] = (now, track.event_id, score)
            return "update", track.event_id

        for known, (seen_at, event_id, known_score) in list(self._recent_plates.items()):
            if now - seen_at > EVENT_DEDUPE_SECONDS:
                del self._recent_plates[known]
                continue
            if not postprocess.same_vehicle(plate, known):
                continue
            # Same vehicle, seen moments ago on this camera.
            if score <= known_score:
                self._recent_plates[known] = (now, event_id, known_score)
                return "skip", None
            del self._recent_plates[known]
            self._recent_plates[plate] = (now, event_id, score)
            return "update", event_id

        self._recent_plates[plate] = (now, None, score)
        return "insert", None

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
    if not rtsp_url or not rtsp_url.strip():
        # Without this an unconfigured camera row loops on
        # cv2.VideoCapture("") every few seconds, filling the log with
        # "!_filename.empty()" and burning a thread for nothing.
        logger.warning("camera %s has no stream URL; not starting a worker", camera_id)
        return
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
