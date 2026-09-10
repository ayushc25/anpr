import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

DATABASE_URL = os.getenv("database_url") or os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("database_url not set in .env")

SECRET_KEY = os.getenv("SECRET_KEY", "anpr-dev-secret-change-me")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
ALGORITHM = "HS256"

STORAGE_DIR = ROOT_DIR / "backend" / "storage" / "events"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# yolo26n.pt is BROKEN: it reports "refrigerator 0.72" on a frame full of cars,
# and finds no vehicles at all. Verified against yolov8n.pt on the same frame,
# which returns car 0.74 / car 0.59 / truck 0.37. Do not point this back at
# yolo26n.pt without re-checking it on real footage.
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(ROOT_DIR / "yolov8n.pt"))
PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", str(ROOT_DIR / "license-plate-finetune-v1l.pt"))
PLATE_CONF_THRESHOLD = float(os.getenv("PLATE_CONF_THRESHOLD", "0.25"))

# Event dedupe window (seconds) - same plate on same camera won't create a new event within this window
# Same vehicle on the same camera inside this window is one arrival, not
# several. A vehicle waiting at a boom, or a test clip that loops, otherwise
# produces a row every time the window lapses.
EVENT_DEDUPE_SECONDS = int(os.getenv("EVENT_DEDUPE_SECONDS", "45"))

# Vehicle classes we care about (COCO ids): bicycle=1, car=2, motorbike=3, bus=5, truck=7
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}

DETECTION_FRAME_INTERVAL = float(os.getenv("DETECTION_FRAME_INTERVAL", "0.6"))  # seconds between detection passes per camera

# Minimum YOLO confidence to accept a vehicle detection. Real gate-camera footage
# (front-facing, close range) scores much higher than stock/aerial test footage,
# so keep this low enough to catch distant/angled vehicles during testing.
VEHICLE_CONF_THRESHOLD = float(os.getenv("VEHICLE_CONF_THRESHOLD", "0.20"))

# Multi-frame tracking: a vehicle is tracked across consecutive detection
# passes (matched by bbox overlap) so its several OCR readings - one is
# rarely perfect due to motion blur/angle/glare - can be reduced to a
# single best-confidence reading instead of emitting one noisy event per frame.
TRACK_IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.25"))
TRACK_TIMEOUT_SECONDS = float(os.getenv("TRACK_TIMEOUT_SECONDS", "2.5"))
MIN_OCR_CONFIDENCE_TO_RECORD = float(os.getenv("MIN_OCR_CONFIDENCE_TO_RECORD", "0.35"))


# --- throughput knobs ------------------------------------------------------
# The plate detector + OCR cost ~400 ms per vehicle against ~120 ms for the
# whole-frame vehicle detector, so the plate stages are what set the frame
# rate. Cap how many vehicles per pass may pay that cost.
MAX_PLATE_READS_PER_PASS = int(os.getenv("MAX_PLATE_READS_PER_PASS", "2"))

# A vehicle smaller than this fraction of the frame is too far away for its
# plate to be legible; reading it spends 400 ms to produce noise.
MIN_VEHICLE_AREA_RATIO = float(os.getenv("MIN_VEHICLE_AREA_RATIO", "0.02"))

# Live-view preview width. Encoding full 1080p costs ~64 ms per frame.
PREVIEW_MAX_WIDTH = int(os.getenv("PREVIEW_MAX_WIDTH", "960"))
