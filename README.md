# ANPR System

On-premises, **CPU-only** ANPR for a society entry/exit gate over standard RTSP IP cameras.

Design of record: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Phasing: [docs/ROADMAP.md](docs/ROADMAP.md)

## Stack

| | |
|---|---|
| **Serving inference** | ONNX Runtime (OpenVINO EP on Intel). No torch, no CUDA, no TensorRT. |
| **Pipeline** | YOLO vehicle detector → ByteTrack → plate detector → CTC recognizer → multi-frame validation |
| **Backend** | Python, FastAPI, SQLAlchemy, Alembic, PostgreSQL |
| **Frontend** | React (Vite), Tailwind, Recharts |

## Layout

```
backend/app/
  ai/          pure inference library — no DB, no FastAPI (see docs/ARCHITECTURE.md §B)
  video/       RTSP reader, ROI, virtual lines, the frame cascade
  events/      track state, multi-frame validator, dedupe, rules engine
  workers/     camera worker (one process per camera) + supervisor
  repositories/  all SQL      services/  use-cases      api/v1/  endpoints
  alembic/     schema migrations
configs/       default.yaml (tuning) · models.yaml (which model per stage)
models/        ONNX artifacts + manifest.yaml (not in git)
scripts/       export_onnx · fetch_recognizer · benchmark
tests/         unit + integration (116 tests, no models or DB required)
```

## Setup

```bash
cd backend
python -m venv venv
venv\Scripts\pip install -r requirements.txt      # Windows

# 1. schema
venv\Scripts\python -m alembic upgrade head

# 2. models: export the detectors from the .pt weights in the repo root
cd .. && backend\venv\Scripts\python scripts\export_onnx.py --all

# 3. recognizer (optional but recommended — otherwise EasyOCR is used)
backend\venv\Scripts\python scripts\fetch_recognizer.py --ppocr

# 4. verify
backend\venv\Scripts\python -m backend.app.cli models check
```

## Running

The API and the camera workers are **separate processes**. Inference never runs
inside FastAPI — that is what lets camera count scale past two.

```bash
# API
backend\venv\Scripts\python -m uvicorn backend.app.main:app --port 8002

# Camera workers (spawns and supervises one process per enabled camera)
backend\venv\Scripts\python -m backend.app.cli supervisor

# One camera in the foreground, logging events instead of storing them —
# the right way to tune a camera's ROI and thresholds on site
backend\venv\Scripts\python -m backend.app.cli worker --camera-id 1 --dry-run -v

# Frontend
cd frontend && npm install && npm run dev
```

## Measure before you promise

Camera capacity is a property of the box, not of the software. Run this on the
hardware you will deliver, and put the number in the handover document:

```bash
backend\venv\Scripts\python scripts\benchmark.py --clip <a real gate clip>
```

## Configuration

- **`configs/models.yaml`** — which implementation and artifact per stage. Swapping
  the recognizer (`easyocr_legacy` → `ppocr_onnx` → `lprnet_onnx`) is a one-line
  change; no pipeline code is touched.
- **`configs/default.yaml`** — processing FPS, gating thresholds, validation
  parameters, retention.
- **`.env`** — `database_url`, `SECRET_KEY`, `WORKER_TOKEN`, `API_BASE_URL`.
- **Per camera** (in the UI / `cameras` table) — RTSP main and sub URL, ROI polygon,
  virtual line, processing FPS, entry/exit role.

> Set `WORKER_TOKEN` before exposing the API off-host: `/api/v1/internal/events`
> accepts event writes and is unauthenticated while that variable is empty.

## Tests

```bash
backend\venv\Scripts\python -m pytest tests -q
```

Unit tests need neither model artifacts nor a database; the integration tests
drive the full cascade with stub models.

## Migration status

The prototype's routers still serve their original paths and are being moved
into `api/v1/` one at a time. Superseded:

- `_run_lightweight_migrations()` in `main.py` → Alembic
- thread-per-camera inside FastAPI → `workers/supervisor.py`
- `models.py` → the `models/` package (same names, re-exported)
- `services/camera_manager.py` + `services/detection.py` → `video/` + `ai/`
  *(still present and still importable; delete once the UI is switched over)*
