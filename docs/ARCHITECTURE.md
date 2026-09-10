# ANPR Edge System — Target Architecture

On-premises, CPU-only ANPR for society entry/exit gates over standard RTSP IP cameras.

This document is the design of record. It describes the **target** system and how we get
there from the current `backend/` prototype (thread-per-camera inside FastAPI, Ultralytics
`.pt` models, EasyOCR, ad-hoc IoU tracker). Everything here is chosen under one hard
constraint: **x86 CPU only, no CUDA, no TensorRT.**

- [A. Folder structure](#a-folder-structure)
- [B. Module responsibilities](#b-module-responsibilities)
- [C. Data flow](#c-data-flow-rtsp--database-event)
- [D. Core interfaces](#d-core-interfaces)
- [E. Making models interchangeable](#e-making-models-interchangeable)
- [F. Database schema](#f-database-schema)
- [G. Camera workers without blocking FastAPI](#g-camera-workers-without-blocking-fastapi)
- [H. CPU optimization strategy](#h-cpu-optimization-strategy)
- [I. OpenVINO vs ONNX Runtime](#i-openvino-vs-onnx-runtime)
- [J. Multi-frame validation algorithm](#j-multi-frame-validation-algorithm)
- [K. API surface](#k-api-surface)
- Roadmap and phasing: see [ROADMAP.md](ROADMAP.md)

---

## 0. The five decisions that shape everything

| # | Decision | Why |
|---|---|---|
| 1 | **Inference runs in separate OS processes, never in the FastAPI process.** | The GIL plus OpenCV decode plus numpy pre/post-processing make thread-per-camera collapse past ~2 cameras. Processes also give per-camera crash isolation and independent thread-pool sizing. |
| 2 | **All models are ONNX artifacts behind four interfaces**, loaded through a pluggable `InferenceBackend` (ONNX Runtime *or* OpenVINO). | Swap YOLO to RT-DETR, or LPRNet to PP-OCR, without touching pipeline code — and A/B the backends on the customer's actual hardware. |
| 3 | **Decode the camera sub-stream (D1/720p) for AI; pull the main stream only for the evidence snapshot.** | Decoding 4x 1080p H.264 costs more CPU than all four inference stages combined. This one decision roughly triples camera density per box. |
| 4 | **Recognition is never trusted from one frame.** A track accumulates weighted reads; a plate is emitted on line-cross or track exit, after quality-weighted positional voting. | Real gate footage is motion-blurred, angled, backlit. Single-frame OCR at a gate runs 70–85%; multi-frame voting on the same footage reaches 93–97%. |
| 5 | **The pipeline is cascaded and gated.** Vehicle detector on 1-in-N frames → tracker on every processed frame → plate detector only on in-ROI vehicle crops → recognizer only on plate crops that pass a quality gate. | The recognizer touches ~2% of frames. This is where the CPU budget is won. |

---

## A. Folder structure

```text
anpr/
├── docs/
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
│
├── backend/
│   ├── pyproject.toml
│   ├── alembic.ini
│   ├── alembic/versions/            # real migrations (replaces _run_lightweight_migrations)
│   └── app/
│       ├── main.py                  # FastAPI app factory. NO model loading, NO cv2.
│       ├── cli.py                   # anpr worker --camera 3 | anpr supervisor | anpr bench
│       │
│       ├── core/
│       │   ├── config.py            # pydantic-settings: env + configs/*.yaml
│       │   ├── security.py          # JWT, password hashing
│       │   ├── permissions.py
│       │   ├── logging.py           # structured JSON logs, per-camera logger
│       │   ├── errors.py
│       │   └── clock.py             # tz-aware now(); everything stored UTC
│       │
│       ├── db/
│       │   ├── base.py              # Declarative Base
│       │   ├── session.py           # engine, SessionLocal, get_db
│       │   └── init_db.py           # seed roles / admin / settings
│       │
│       ├── models/                  # SQLAlchemy ORM, one file per aggregate
│       │   ├── user.py  camera.py  vehicle.py  resident.py
│       │   ├── event.py             # Event + EventRead (per-frame evidence)
│       │   ├── alert.py             # AlertRule + Alert + Notification
│       │   └── audit.py  setting.py
│       │
│       ├── schemas/                 # Pydantic v2 request/response DTOs
│       │
│       ├── repositories/            # ALL SQL lives here. Session in, DTO/ORM out.
│       │   ├── base.py  camera_repo.py  vehicle_repo.py
│       │   └── event_repo.py  alert_repo.py  user_repo.py  report_repo.py
│       │
│       ├── services/                # use-cases; orchestrate repos. No SQL, no HTTP.
│       │   ├── camera_service.py    # CRUD + publish config-changed to workers
│       │   ├── vehicle_service.py   # registry, plate normalization, bulk import
│       │   ├── event_service.py     # ingest from worker -> match -> alerts
│       │   └── dashboard_service.py  report_service.py  auth_service.py
│       │
│       ├── api/
│       │   ├── deps.py              # get_db, current_user, require_permission
│       │   └── v1/
│       │       ├── router.py
│       │       └── endpoints/       # auth cameras vehicles residents events
│       │                            # alerts dashboard reports users settings health
│       │
│       ├── ai/                      # PURE. No DB, no FastAPI, no I/O beyond model files.
│       │   ├── types.py             # Detection, Track, PlateCandidate, PlateRead
│       │   ├── registry.py          # name -> class map, build_from_config()
│       │   ├── inference/
│       │   │   ├── backend.py       # InferenceBackend ABC
│       │   │   ├── onnxrt_backend.py
│       │   │   ├── openvino_backend.py
│       │   │   ├── warmup.py
│       │   │   └── threading.py     # thread-count budget calculator
│       │   ├── vehicle_detector/
│       │   │   ├── base.py          # VehicleDetector ABC
│       │   │   ├── yolo_onnx.py     # YOLOv8/11-n, letterbox + NMS in numpy
│       │   │   └── null.py          # passthrough (plate-only mode)
│       │   ├── vehicle_tracker/
│       │   │   ├── base.py           bytetrack.py      # supervision.ByteTrack
│       │   │   └── iou_tracker.py    # cheap fallback (today's logic)
│       │   ├── plate_detector/
│       │   │   ├── base.py           yolo_plate_onnx.py
│       │   ├── plate_recognizer/
│       │   │   ├── base.py
│       │   │   ├── lprnet_onnx.py   # CTC, 36-char, Indian fine-tune
│       │   │   ├── ppocr_onnx.py    # PP-OCRv4 mobile rec (strong baseline)
│       │   │   ├── easyocr_legacy.py# current prototype path, kept for A/B
│       │   │   └── postprocess.py   # IN-plate grammar, confusion map, normalize
│       │   └── quality/
│       │       ├── sharpness.py     # variance of Laplacian, Tenengrad
│       │       └── plate_quality.py # composite 0..1 score
│       │
│       ├── video/
│       │   ├── rtsp_reader.py       # capture thread, latest-frame (size-1) queue
│       │   ├── reconnect.py         # exponential backoff + health
│       │   ├── roi.py               # polygon mask, point-in-poly, crop-to-ROI
│       │   ├── line_crossing.py     # virtual line, signed side, direction
│       │   ├── snapshot.py          # main-stream still grab for evidence
│       │   └── frame_processor.py   # the cascade: detect -> track -> plate -> recognize
│       │
│       ├── events/
│       │   ├── track_state.py       # per-track accumulator
│       │   ├── multi_frame_validator.py
│       │   ├── event_builder.py     # TrackState -> EventDraft (+ images)
│       │   ├── dedupe.py            # plate x camera cooldown, bounce suppression
│       │   └── rules_engine.py      # AlertRule evaluation
│       │
│       ├── alerts/
│       │   ├── alert_service.py     # create/ack/list; fan-out to channels
│       │   └── channels/
│       │       ├── base.py          # NotificationChannel ABC
│       │       ├── smtp.py          # OPTIONAL module, own config + outbox
│       │       └── webhook.py
│       │
│       ├── reports/
│       │   ├── excel_service.py     # openpyxl streaming (write_only) writer
│       │   └── definitions.py       # named report specs (columns, filters)
│       │
│       ├── workers/
│       │   ├── supervisor.py        # spawns / monitors / restarts camera processes
│       │   ├── camera_worker.py     # ONE camera, ONE process; the main loop
│       │   ├── ingest_client.py     # worker -> API/DB event submission (+ spool)
│       │   ├── control_bus.py       # Redis pub/sub: config reload, stop, snapshot
│       │   └── janitor.py           # retention: purge old frames / events
│       │
│       └── storage/
│           ├── media_store.py       # date-sharded paths, JPEG write, URL mapping
│           └── spool.py             # on-disk queue if DB unreachable
│
├── frontend/                        # React (Vite) + Tailwind — pages listed in §K
│   └── src/{api,components,pages,hooks,context,lib}
│
├── models/                          # artifacts, NOT in git (DVC/LFS or shipped installer)
│   ├── vehicle/yolo11n_veh_640.onnx      (+ .int8.onnx, + openvino/)
│   ├── plate/yolo11n_plate_320.onnx      (+ ...)
│   ├── recognizer/lprnet_in_v1.onnx      (+ charset.txt)
│   └── manifest.yaml                # name, sha256, input size, classes, version
│
├── configs/
│   ├── default.yaml                 # pipeline defaults
│   ├── models.yaml                  # which impl + which artifact per stage
│   ├── cameras/                     # per-site camera overrides (ROI / line / fps)
│   └── logging.yaml
│
├── scripts/
│   ├── export_onnx.py               # .pt -> .onnx (+ onnxsim)
│   ├── quantize_int8.py             # ORT static quant / OpenVINO NNCF PTQ
│   ├── benchmark.py                 # per-stage ms + max cameras on this box
│   ├── harvest_dataset.py           # mine stored crops -> training set
│   ├── eval_pipeline.py             # plate accuracy on a labelled clip set
│   └── import_registry.py           # client Excel -> vehicles table
│
├── tests/
│   ├── unit/                        # validator, rules, roi, postprocess, dedupe
│   ├── integration/                 # api + db
│   └── fixtures/clips/              # short labelled recordings
│
└── deploy/
    ├── docker/{api,worker,frontend}.Dockerfile
    ├── docker-compose.yml           # postgres, redis, api, worker-supervisor, nginx
    └── systemd/                     # bare-metal alternative for edge boxes
```

### What changed vs. the structure you proposed, and why

- **`repositories/` is real and mandatory.** Today `camera_manager.py` opens `SessionLocal()`
  inside the frame loop. In the target, workers hold no ORM session during inference.
- **`ai/` is a pure library.** It imports no FastAPI, no SQLAlchemy, no app config. That is what
  makes `scripts/eval_pipeline.py` and unit tests possible without a database.
- **`workers/supervisor.py` added** — camera processes need a parent that restarts them.
- **`storage/spool.py` added** — a gate system must not lose events when Postgres restarts.
- **`alerts/channels/`** replaces a bare `smtp_service`: SMTP becomes one implementation of a channel
  interface, so WhatsApp/webhook drop in later without touching `alert_service`.
- **`deploy/` instead of `Docker/`** — holds compose *and* systemd, since edge boxes often run bare metal.
- **`quality/` added** — plate quality scoring feeds both best-frame selection and vote weights,
  so it deserves to be independently testable.

---

## B. Module responsibilities

### `ai/` — the pure inference library

| Module | Owns | Must never |
|---|---|---|
| `inference/backend.py` | Session lifecycle, thread config, `run(inputs) -> outputs` | Know what a "vehicle" is |
| `vehicle_detector/` | Letterbox, forward pass, NMS, class filter → `list[Detection]` | Track, crop plates, touch DB |
| `vehicle_tracker/` | Assign stable `track_id`, maintain centroid history | Run any neural net |
| `plate_detector/` | Find plate quads inside a **vehicle crop** → `list[PlateCandidate]` | Read characters |
| `plate_recognizer/` | Plate crop → `PlateRead(text, conf, per_char_conf)` | Decide if the read is trustworthy |
| `quality/` | Score a plate crop 0..1 (sharpness, size, aspect, brightness) | Anything else |

`ai/registry.py` is the only place that maps a config string to a class. Nothing else imports a
concrete implementation.

### `video/` — pixels in, structured observations out

- **`rtsp_reader.py`** — one thread per stream doing `cap.read()` into a **size-1 queue with drop
  policy**. This is the single most important detail for latency: if the consumer is slow, old
  frames are discarded rather than buffered. The current prototype reads and encodes MJPEG on the
  same thread as detection, which couples preview FPS to inference latency.
- **`roi.py`** — polygon in normalized (0..1) coords so ROI survives resolution changes.
  Provides `contains(point)`, `overlap_ratio(bbox)`, and `crop_bounds()` so the detector can run on
  a cropped tensor rather than the full frame (a large, easy CPU saving).
- **`line_crossing.py`** — virtual line as two normalized points plus a direction convention; returns
  `CROSSED_IN | CROSSED_OUT | NONE` for a track's centroid history.
- **`frame_processor.py`** — the cascade, and the only class that knows the order of the stages.
  Stateless with respect to cameras; all state lives in the `TrackState` objects it is handed.

### `events/` — turning observations into a business fact

- **`track_state.py`** — accumulator per `track_id`: reads, quality scores, best vehicle crop, best
  plate crop, first/last seen, centroid history, ROI dwell.
- **`multi_frame_validator.py`** — the voting algorithm (§J). Pure function of a `TrackState`.
- **`event_builder.py`** — emits an `EventDraft` on the trigger (line-cross, or ROI exit, or track
  timeout), attaching the two JPEGs.
- **`dedupe.py`** — suppresses the same plate on the same camera within a cooldown, and the
  "bounce" case (IN then OUT within seconds on the same camera).
- **`rules_engine.py`** — evaluates `AlertRule` rows against a persisted event. Deliberately
  separate from event creation so rules can be re-run/backfilled.

### `workers/`

- **`camera_worker.py`** — owns exactly one camera: reader, pipeline, track states, ingest client.
  Runs in its own process. Its `run()` is a loop with an explicit FPS pacer.
- **`supervisor.py`** — reads enabled cameras, spawns workers, restarts on exit with backoff, exposes
  worker health, and reacts to control-bus messages (`camera.created/updated/deleted`).
- **`ingest_client.py`** — POSTs the event to the API (or writes directly via repositories in
  single-box mode). On failure, writes to `storage/spool.py` and retries.

### `services/` vs `repositories/`

A hard rule that keeps this maintainable: **endpoints call services, services call repositories,
repositories are the only code that constructs queries.** `event_service.ingest()` is the one
transaction that writes the event, resolves the vehicle registry match, evaluates rules and
enqueues notifications.

### `alerts/`

`alert_service.dispatch(alert)` looks up enabled channels and hands each a `Notification` row.
SMTP is fully optional: if `alerts.smtp.enabled` is false the channel is simply not registered, and
nothing else in the system changes. Notifications are persisted **before** sending, with
`status = pending|sent|failed` and a retry count, so an SMTP outage never blocks the pipeline.

---

## C. Data flow (RTSP → database event)

```text
IP CAMERA
  ├── sub-stream  (704x576 / 1280x720, ~8-12 fps)  ──► AI path
  └── main-stream (1920x1080)                      ──► evidence snapshot only, on demand

[1] RtspReader thread            cap.read() -> size-1 queue (drop old)
        │
[2] FPS pacer                    take latest frame; target processing_fps (e.g. 6)
        │
[3] ROI gate                     crop to ROI bounding box (cheap, big win)
        │
[4] VehicleDetector              1-in-N frames (N = detect_interval, e.g. every 2nd processed frame)
        │                        -> [Detection(bbox, cls, conf)]
[5] VehicleTracker (ByteTrack)   EVERY processed frame -> [Track(track_id, bbox, history)]
        │                        (on non-detect frames, tracker coasts on prediction)
        │
[6] Per-track gating             run plate stage only if:
        │                          - centroid inside ROI polygon
        │                          - bbox area >= min_vehicle_area
        │                          - frames_since_last_plate_attempt >= plate_interval
        │                          - track not already "locked" (high-confidence plate settled)
        │
[7] PlateDetector                on the VEHICLE CROP (not the full frame) at 320x320
        │                        -> [PlateCandidate(quad, conf)]
        │
[8] Quality gate                 plate_quality(crop) = f(width_px, sharpness, aspect, exposure)
        │                        skip recognition if quality < min_quality (e.g. 0.35)
        │
[9] PlateRecognizer              perspective-warp -> 96x48 gray -> CTC -> PlateRead
        │                        (text, conf, per_char_conf[])
        │
[10] postprocess                 uppercase, strip, IN-grammar normalize, confusion fixes
        │
[11] TrackState.add_read(...)    store read + quality + plate crop + vehicle crop if best so far
        │
[12] TRIGGER                     line-crossing detected  OR  track left ROI  OR  track timeout
        │
[13] MultiFrameValidator         quality-weighted string vote + positional char vote
        │                        -> FinalPlate(text, confidence, support, n_reads) or None
        │
[14] Dedupe                      (plate, camera, direction) seen within cooldown? -> drop
        │
[15] EventDraft                  + best vehicle JPEG + best plate JPEG + optional main-stream snapshot
        │
[16] IngestClient  ──HTTP/direct──► event_service.ingest()   [spool to disk on failure]
        │
[17] vehicle_repo.match(plate)   exact -> else normalized -> else edit-distance<=1 fuzzy (flagged)
        │                        -> registered | whitelist | blacklist | unknown
[18] events INSERT + event_reads INSERT (evidence trail)
        │
[19] RulesEngine                 evaluate AlertRule rows -> Alert rows
        │
[20] alert_service.dispatch      -> Notification rows -> SMTP / webhook channel (async, retried)
        │
[21] WebSocket /ws/events        push to Dashboard + Live Events pages
        │
[22] Reports / Excel             read from events + joins, streamed export
```

**Timing budget on one 4-core edge box, per processed frame (typical, INT8 OpenVINO):**

| Stage | Runs on | Cost |
|---|---|---|
| H.264 decode (sub-stream) | every camera frame | 3–6 ms |
| Vehicle detect 640 | 1-in-2 processed frames | 25–45 ms |
| ByteTrack | every processed frame | <1 ms |
| Plate detect 320 (per vehicle) | gated | 8–15 ms |
| Recognizer 96x48 | gated | 1–3 ms |
| Voting + event write | on trigger only | ~5 ms |

At 6 processing FPS this is roughly 0.20–0.25 cores per camera for inference plus ~0.15 for decode,
i.e. **~6–8 cameras on a 4-core i5, ~14–18 on an 8-core i7**, assuming sub-streams. Confirm with
`scripts/benchmark.py` on the real box before promising a number to the client.

---

## D. Core interfaces

`app/ai/types.py` — the vocabulary everything else speaks:

```python
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np

BBox = tuple[int, int, int, int]  # x1, y1, x2, y2 in FULL-FRAME pixel coords

@dataclass(slots=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str            # "car" | "motorcycle" | "bus" | "truck"

@dataclass(slots=True)
class Track:
    track_id: int
    bbox: BBox
    class_name: str
    confidence: float
    age: int                             # processed frames since first seen
    centroid_history: list[tuple[float, float]] = field(default_factory=list)

@dataclass(slots=True)
class PlateCandidate:
    bbox: BBox                 # full-frame coords
    confidence: float
    quad: Optional[np.ndarray] = None     # 4x2 for perspective warp, if the model gives one

@dataclass(slots=True)
class PlateRead:
    text: str
    confidence: float                     # aggregate, 0..1
    per_char_confidence: list[float] = field(default_factory=list)
    raw_text: str = ""                    # pre-postprocessing, kept for audit
```

### VehicleDetector

```python
class VehicleDetector(ABC):
    """Stateless. One frame in, detections out, full-frame coordinates."""

    @abstractmethod
    def detect(self, frame: np.ndarray, roi_offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        """`frame` may be an ROI crop; `roi_offset` is added back so callers
        always receive full-frame coordinates."""

    @property
    @abstractmethod
    def input_size(self) -> tuple[int, int]: ...

    def warmup(self, n: int = 2) -> None: ...
    def close(self) -> None: ...
```

### VehicleTracker

```python
class VehicleTracker(ABC):
    """Stateful per camera. One instance per CameraWorker."""

    @abstractmethod
    def update(self, detections: list[Detection], frame_shape: tuple[int, int]) -> list[Track]:
        """Called on EVERY processed frame. On frames where the detector did
        not run, pass an empty list so the tracker coasts on prediction."""

    @abstractmethod
    def removed_track_ids(self) -> list[int]:
        """Tracks retired since the last call — the signal to finalize an event."""

    def reset(self) -> None: ...
```

### PlateDetector

```python
class PlateDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[PlateCandidate]:
        """`image` is normally a VEHICLE CROP; `offset` maps results back to
        full-frame coordinates. Results are sorted by confidence, best first."""

    @property
    @abstractmethod
    def input_size(self) -> tuple[int, int]: ...
```

### PlateRecognizer

```python
class PlateRecognizer(ABC):
    """The interchangeable one. LPRNet, PP-OCR, a future ViT — all fit here."""

    name: str
    charset: str

    @abstractmethod
    def recognize(self, plate_image: np.ndarray) -> Optional[PlateRead]:
        """`plate_image` is a tight, deskewed BGR plate crop."""

    def recognize_batch(self, plate_images: Sequence[np.ndarray]) -> list[Optional[PlateRead]]:
        return [self.recognize(im) for im in plate_images]   # override for real batching

    @property
    def expects_grayscale(self) -> bool:
        return True
```

### FrameProcessor

```python
class FrameProcessor:
    """Owns the cascade. One instance per camera. No I/O, no DB."""

    def __init__(self, detector, tracker, plate_detector, recognizer,
                 quality_scorer, roi: RoiPolygon, line: Optional[VirtualLine],
                 cfg: PipelineConfig):
        ...

    def process(self, frame: np.ndarray, frame_idx: int, ts: float) -> ProcessResult:
        """Returns tracks (for overlay), plus TrackStates that hit a finalize
        trigger this frame. Never blocks on network or disk."""
```

```python
@dataclass
class ProcessResult:
    tracks: list[Track]
    finalized: list[TrackState]        # ready for validation + event building
    stats: FrameStats                  # per-stage ms, for /health and tuning
```

### CameraWorker

```python
class CameraWorker:
    """One camera, one OS process. Owns the reader, pipeline and ingest."""

    def __init__(self, camera_cfg: CameraConfig, pipeline_cfg: PipelineConfig): ...

    def run(self) -> None:
        """Blocking loop:
             pace to processing_fps
             frame = reader.latest()          (None -> health-check / reconnect)
             result = processor.process(...)
             for ts in result.finalized:
                 final = validator.validate(ts)
                 if final and not dedupe.is_duplicate(final):
                     ingest.submit(event_builder.build(ts, final))
             publish preview JPEG (rate-limited) if anyone is watching
             poll control bus for reload/stop
        """

    def stop(self) -> None: ...
    def health(self) -> WorkerHealth: ...   # fps, last_frame_at, stage ms, queue depth
```

### MultiFrameValidator

```python
class MultiFrameValidator:
    def __init__(self, cfg: ValidationConfig, plate_grammar: PlateGrammar): ...

    def validate(self, state: TrackState) -> Optional[FinalPlate]: ...
```

```python
@dataclass
class FinalPlate:
    text: str
    confidence: float          # 0..1, calibrated aggregate
    support: float             # winning weight / total weight
    read_count: int
    distinct_variants: int
    grammar_valid: bool
    corrections: list[str]     # e.g. ["pos4: 8->B (confusion)", "registry-snap"]
```

### EventProcessor (`services/event_service.py`)

```python
class EventProcessor:
    """Runs in the API process, inside ONE transaction."""

    def ingest(self, draft: EventDraft) -> Event:
        """1. persist media via media_store
           2. resolve registry match -> status
           3. insert Event (+ EventRead rows)
           4. rules_engine.evaluate(event) -> Alerts
           5. alert_service.dispatch(alerts)   (enqueue only)
           6. broadcast over WebSocket
        """
```

### RulesEngine

```python
class RulesEngine:
    def evaluate(self, event: Event, ctx: RuleContext) -> list[AlertDraft]: ...

class Rule(ABC):
    code: str
    @abstractmethod
    def check(self, event: Event, ctx: RuleContext) -> Optional[AlertDraft]: ...

# Built-ins: BlacklistedVehicleRule, UnknownVehicleRule, ExpiredPermitRule,
# AfterHoursEntryRule, RepeatedUnknownRule, LowConfidenceRule, TailgatingRule
```

Rules are registered by `code`; an `alert_rules` DB row supplies parameters and enable/disable, so
operators tune thresholds without a deploy.

---

## E. Making models interchangeable

Four layers, each swappable independently:

**1. Config selects the implementation.** `configs/models.yaml`:

```yaml
vehicle_detector:
  impl: yolo_onnx
  artifact: models/vehicle/yolo11n_veh_640.int8.onnx
  backend: openvino          # onnxruntime | openvino
  input_size: [640, 640]
  conf_threshold: 0.35
  nms_iou: 0.5
  classes: {2: car, 3: motorcycle, 5: bus, 7: truck}

plate_detector:
  impl: yolo_plate_onnx
  artifact: models/plate/yolo11n_plate_320.int8.onnx
  input_size: [320, 320]
  conf_threshold: 0.30

plate_recognizer:
  impl: lprnet_onnx          # <- ppocr_onnx | easyocr_legacy
  artifact: models/recognizer/lprnet_in_v1.onnx
  charset: models/recognizer/charset.txt
  input_size: [96, 48]
  min_char_confidence: 0.40

tracker:
  impl: bytetrack
  track_thresh: 0.35
  match_thresh: 0.80
  track_buffer: 45
```

**2. A registry resolves the string.**

```python
# app/ai/registry.py
VEHICLE_DETECTORS = {"yolo_onnx": YoloOnnxVehicleDetector, "null": NullVehicleDetector}
PLATE_RECOGNIZERS = {"lprnet_onnx": LprnetOnnx, "ppocr_onnx": PpOcrOnnx,
                     "easyocr_legacy": EasyOcrLegacy}

def build_recognizer(cfg: dict) -> PlateRecognizer:
    try:
        cls = PLATE_RECOGNIZERS[cfg["impl"]]
    except KeyError:
        raise ConfigError(f"unknown plate_recognizer.impl={cfg['impl']!r}; "
                          f"known: {sorted(PLATE_RECOGNIZERS)}")
    return cls(**{k: v for k, v in cfg.items() if k != "impl"})
```

**3. The `InferenceBackend` abstraction sits *below* the model classes**, so ONNX Runtime vs
OpenVINO is orthogonal to LPRNet vs PP-OCR:

```python
class InferenceBackend(ABC):
    @abstractmethod
    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]: ...
    @property
    @abstractmethod
    def input_shapes(self) -> dict[str, tuple]: ...

def build_backend(artifact: str, backend: str, threads: ThreadBudget) -> InferenceBackend:
    if backend == "openvino":
        return OpenVinoBackend(artifact, threads)
    return OnnxRuntimeBackend(artifact, threads)
```

**4. A conformance test suite every implementation must pass.**
`tests/unit/test_recognizer_contract.py` parametrizes over every registered recognizer and asserts:
returns `None` on garbage input, output is uppercase alphanumeric, `len(per_char_confidence) ==
len(text)`, confidence in `[0,1]`, and no crash on a 3x3 image. A new recognizer is "done" when it
passes the contract test plus a minimum accuracy bar on `tests/fixtures/plates/`.

**Practical model path for Indian plates (CPU):**

| Stage | Phase 1 (ship now) | Phase 2 (accuracy) |
|---|---|---|
| Vehicle | `yolo26n.pt` → ONNX FP32 → INT8 | keep; maybe retrain on site data |
| Plate | your `license-plate-finetune-v1l.pt` — **but note it is a `-l`, ~50 MB, far too heavy for CPU.** Distil/retrain the same dataset into a `yolov8n`/`yolo11n` at 320px | fine-tune nano on harvested site crops |
| Recognize | **PP-OCRv4 mobile rec (English) via ONNX** — ~10 MB, 1–3 ms/crop, immediately better than EasyOCR and far faster | **LPRNet/CRNN-CTC fine-tuned on Indian plates**, trained on harvested crops + synthetic IND plates |

The self-improving loop matters here: you already have thousands of plate crops in
`backend/storage/events/`. `scripts/harvest_dataset.py` mines those (plus new ones tagged by the
validator as *disputed*) into a labelled set, which is exactly the fine-tuning data that takes a
generic recognizer from ~88% to ~96% on this specific gate. Build the harvest script early.

Also keep two-line plate handling in mind — Indian two-wheelers and many commercial vehicles use
stacked plates. The recognizer must either be trained on stacked crops or the plate detector must
emit a split hint; single-line-only recognition silently fails on ~30% of a society's vehicles.

---

## F. Database schema

PostgreSQL. Everything UTC. Alembic-managed from day one (delete `_run_lightweight_migrations`).

```text
users ─────────────< activity_logs
  │
  └──< alerts.acknowledged_by

locations ────< cameras ────< camera_zones      (ROI polygons + virtual lines, versioned)
                   │
                   └────< events >──── vehicles >──── residents
                             │              │
                             │              └──< vehicle_permits (validity windows)
                             ├──< event_reads      (per-frame evidence)
                             └──< alerts ──< notifications

alert_rules ──< alerts
settings (key/value, typed)
```

### Tables

**`residents`** — `id, name, flat_number, block, phone, email, is_active, created_at`
Separated from `vehicles` because one flat commonly has 2–4 vehicles and the client's list will be
maintained per resident.

**`vehicles`** — the registry.
`id, plate_number (unique, normalized), plate_raw, resident_id FK NULL, vehicle_type,
make_model, color, status ENUM(registered|whitelist|blacklist|unknown|visitor), valid_from,
valid_until, notes, is_active, created_at, updated_at`
Index: `UNIQUE(plate_number)`, plus a trigram index `gin(plate_number gin_trgm_ops)` for fuzzy
lookup — that is what makes edit-distance-1 registry snapping fast.

**`plate_aliases`** — `id, vehicle_id FK, alias_plate, reason`
Real deployments accumulate known misreads (a `0`/`D` that a particular camera always confuses).
Rather than degrading the recognizer, map the alias.

**`cameras`** — extends today's table:
`id, code (unique), name, rtsp_url_main, rtsp_url_sub, location_id FK, direction
ENUM(in|out|both), processing_fps, detect_interval, is_enabled, snapshot_from_main,
model_profile, last_seen_at, health JSONB, created_at`
Credentials belong in `rtsp_url` only if you accept them in the DB; better is
`rtsp_credential_id` → an encrypted secrets table. Note: today's `rtsp_url` is stored plaintext and
returned by `GET /cameras` — fix that before go-live.

**`camera_zones`** — `id, camera_id FK, kind ENUM(roi|line|mask), name, geometry JSONB
(normalized 0..1 coords), direction_hint, is_active, created_at`
Versioned rather than mutated so historical events remain explainable.

**`events`** — the core fact table.
`id (BIGSERIAL), event_uid (UUID, unique — generated by the worker for idempotent ingest),
camera_id FK, plate_number, plate_raw, vehicle_id FK NULL, resident_id FK NULL,
direction ENUM(in|out), status ENUM(...), vehicle_type, vehicle_color,
plate_confidence, detect_confidence, validation_support, read_count,
vehicle_image_path, plate_image_path, snapshot_path, track_id, is_manual_override,
reviewed_by FK NULL, detected_at TIMESTAMPTZ, created_at`
Indexes: `(detected_at DESC)`, `(plate_number, detected_at DESC)`,
`(camera_id, detected_at DESC)`, `(status, detected_at DESC)`.
**Partition by month on `detected_at`** — at 4 cameras x ~800 events/day this passes 1M rows in a
year, and reports/retention become trivial with partitions.

**`event_reads`** — the evidence trail, one row per accepted frame read.
`id, event_id FK, frame_ts, raw_text, normalized_text, rec_confidence, plate_det_confidence,
quality_score, weight, plate_crop_path NULL`
This is what lets an operator answer "why did it say UP32A81234?" and is your fine-tuning dataset.
Retain ~30 days, then prune (janitor).

**`alert_rules`** — `id, code, name, params JSONB, severity, is_enabled, channels JSONB
(["smtp","webhook"]), cooldown_seconds, camera_ids JSONB NULL, created_at`

**`alerts`** — `id, rule_id FK, event_id FK, severity, title, message, status
ENUM(new|acknowledged|resolved), acknowledged_by FK, acknowledged_at, created_at`

**`notifications`** — `id, alert_id FK, channel, target, status ENUM(pending|sent|failed),
attempts, last_error, sent_at, created_at` — the outbox that makes SMTP failures non-blocking.

**`users` / `roles`** — keep your current `role_name` + `permissions JSONB`, but promote roles to
their own table so permission sets are edited once, not per user.

**`settings`** — `key, value JSONB, updated_by, updated_at` for runtime-tunable knobs
(cooldowns, thresholds, retention days, SMTP config).

**`activity_logs`** — as today; add `entity_type`, `entity_id`, `ip_address`.

### Migration note from the current schema
`events.ocr_confidence` → `plate_confidence`; add `event_uid`, `direction` as a real
in/out (today it copies `camera.direction`, including `both`, which makes entry/exit reporting
ambiguous); split `Vehicle.owner_name/flat_number` into `residents`. Write these as three Alembic
revisions with data backfill, not as `ALTER ... IF NOT EXISTS` strings.

---

## G. Camera workers without blocking FastAPI

**The rule: the FastAPI process never imports cv2, never loads a model, never runs inference.**

```text
                        ┌──────────────────────────────┐
   nginx :80  ─────────►│  FastAPI (uvicorn, 2 workers)│──► PostgreSQL
   React SPA            │  API + WebSocket + Excel     │──► /media static
                        └───────────┬──────────────────┘
                                    │ Redis pub/sub (control) + HTTP (ingest)
                        ┌───────────▼──────────────────┐
                        │  anpr-supervisor (process)   │
                        └───┬────────┬────────┬────────┘
                            │ spawn  │        │
                    ┌───────▼──┐ ┌───▼─────┐ ┌▼────────┐
                    │ worker   │ │ worker  │ │ worker  │   one OS process per camera
                    │ cam #1   │ │ cam #2  │ │ cam #3  │   (or per small group)
                    └──────────┘ └─────────┘ └─────────┘
                     each: reader thread + pipeline thread
```

**Why processes, not threads.** ONNX Runtime and OpenVINO release the GIL during `run()`, so
inference itself parallelises across threads. But everything around it does not: `cv2.VideoCapture.read()`
copies into a numpy array under the GIL, letterboxing, NMS, crop/warp and the Python-level tracker
are all GIL-bound. In the current prototype these serialize, which is why adding cameras degrades
every camera's FPS rather than just the new one. Separate processes also mean a segfault in a video
decoder kills one camera, not the whole gate.

**Supervisor mechanics** (`multiprocessing.Process` with `spawn`, or `subprocess` running
`anpr worker --camera-id N` — prefer the latter, it gives clean restarts and easy systemd/docker
supervision):

```python
class Supervisor:
    def reconcile(self) -> None:
        desired = {c.id for c in camera_repo.list_enabled()}
        running = set(self._procs)
        for cid in desired - running:      self._spawn(cid)
        for cid in running - desired:      self._stop(cid)
        for cid, p in list(self._procs.items()):
            if p.poll() is not None:                     # died
                self._backoff[cid] = min(self._backoff.get(cid, 1) * 2, 60)
                self._schedule_restart(cid)
```

`reconcile()` runs every few seconds *and* on a Redis `camera.*` message, so the UI's "Add camera"
button starts a worker within a second without FastAPI ever touching video.

**Ingest path.** Workers `POST /api/v1/internal/events` with a shared-secret header and an
`event_uid` for idempotency. In single-box deployments you may let workers write via
`repositories/` directly (fewer moving parts) — keep it behind `ingest_client` either way so the
choice is one config flag.

**Live preview without coupling.** Workers publish a rate-limited JPEG (e.g. 5 fps, only while a
viewer is subscribed) to Redis; FastAPI's `/cameras/{id}/stream` reads from Redis and serves MJPEG.
Today's implementation encodes JPEG for every single decoded frame whether or not anyone is
watching — pure wasted CPU on an edge box.

**Sizing.** One process per camera up to core count; beyond that, group cameras
(`--cameras 5,6,7`) so process count stays near `physical_cores` and thread oversubscription
does not thrash. The supervisor computes this automatically from `os.cpu_count()` and the camera
count, which is what makes "number of cameras not finalized" a non-issue.

**FastAPI-side hygiene:** all endpoints that touch the DB use `def` (threadpool) or async with an
async driver — never blocking SQLAlchemy inside `async def`. Excel export runs in a background task
and returns a download URL for large ranges.

---

## H. CPU optimization strategy

Ordered by payoff per hour of work.

1. **Use the sub-stream.** 1080p→720p decode is a ~2.5x saving, and detection accuracy at a gate
   barely moves because the vehicle is close. Grab the main stream only for the evidence snapshot.
2. **ROI-crop before inference.** A gate ROI is often 40% of the frame. Feeding a 640x640 letterbox
   of just the ROI is both faster (smaller effective scene) and more accurate (bigger apparent plates).
3. **Frame skipping with a pacer, not a sleep.** Read at native FPS into a drop queue; process at
   `processing_fps`. This keeps latency low; sleeping in the read loop buffers stale frames.
4. **Cascade gating** (§C step 6). The recognizer is cheap per call, but the plate detector is not —
   gate it on ROI membership, minimum vehicle box area, and a per-track attempt interval.
5. **Track locking.** Once a track has ≥4 consistent reads with support ≥0.9, stop running the plate
   stages for it entirely. On a busy gate this cuts plate-stage calls by half.
6. **INT8 quantization — measure before believing it.** *Dynamic* quantization was measured on
   this project's box and came out **0.60–0.70x, i.e. slower**: without VNNI the per-op
   dequantisation costs more than the narrower arithmetic saves. Static PTQ with ~300
   representative frames from the actual site is the version worth trying, and it must be
   benchmarked on the delivered hardware rather than assumed. Always keep the FP32 artifact and
   gate promotion on `scripts/eval_pipeline.py`. Run `scripts/quantize_int8.py`, which reports the
   speedup so a regression is visible immediately.
7. **Thread budget, explicitly set.** This is the most common silent killer. Every worker process
   must set, *before* importing numpy/ORT:
   ```
   OMP_NUM_THREADS = intra_op
   ORT: intra_op_num_threads = intra_op, inter_op_num_threads = 1,
        execution_mode = ORT_SEQUENTIAL, graph_optimization_level = ORT_ENABLE_ALL
   OpenVINO: INFERENCE_NUM_THREADS = intra_op, PERFORMANCE_HINT = LATENCY (or THROUGHPUT for grouped)
   cv2.setNumThreads(1)     # OpenCV's own pool otherwise fights ORT's
   ```
   with `intra_op = max(1, (physical_cores - 1) // n_worker_processes)`. Left at defaults, four
   workers each spawn `n_cores` threads and the box spends its time in context switches — a real
   2–4x regression.
8. **Pin memory layout.** Pre-allocate the letterbox buffer per worker and reuse it; avoid
   `np.ascontiguousarray` churn in the hot path. Use `cv2.dnn.blobFromImage` or an explicit
   pre-allocated transpose rather than chained numpy ops.
9. **Batch the recognizer.** When several plates are pending in one frame, one batched CTC call
   beats N calls.
10. **Model input sizes are tuning knobs, not constants.** Vehicle detector at 512 instead of 640 is
    ~35% faster and usually fine at a gate. Expose in `models.yaml` and measure.
11. **Avoid re-encoding JPEG unless needed.** Only encode preview frames when a viewer is
    subscribed; only encode evidence crops on finalize.
12. **Postgres**: batch `event_reads` inserts, use `COPY`/`executemany`, and never index-scan
    unpartitioned `events` for dashboards — precompute daily counters in a small `daily_stats`
    rollup refreshed by the janitor.

Measure with `scripts/benchmark.py`, which reports per-stage p50/p95 ms and prints the maximum
supported camera count for the current box. Ship this number in the handover doc.

---

## I. OpenVINO vs ONNX Runtime

**Recommendation: build against ONNX Runtime as the portable default, and enable the OpenVINO
Execution Provider (not the standalone OpenVINO API) on Intel edge boxes.**

| | ONNX Runtime (CPU EP) | OpenVINO (native API) | ORT + OpenVINO EP |
|---|---|---|---|
| Speed on Intel | baseline | **+25–45%** on CNNs | ~same as native OV |
| Speed on AMD / ARM | good | poor / unsupported | falls back to CPU EP |
| API surface | one, stable | second API to learn | **one API** |
| INT8 tooling | `onnxruntime.quantization` | NNCF/POT, generally better results | either |
| Deployment weight | ~50 MB | ~200 MB+ | both |
| Ops coverage | broadest | occasional unsupported op | auto-falls back per-subgraph |

The reasoning for the hybrid: society/edge boxes are overwhelmingly Intel (NUC, Dell OptiPlex,
mini-PC i5/i7), where OpenVINO's win is real and worth having. But you will meet an AMD Ryzen box
or an ARM SBC eventually, and you do not want two inference code paths. `onnxruntime-openvino`
gives you OpenVINO kernels through the ORT API, with automatic per-subgraph fallback.

So: keep `InferenceBackend` with three selectable values — `cpu` (ORT CPU EP), `openvino`
(ORT OpenVINO EP), `openvino_native` (only if you later find the EP leaves performance on the
table) — default resolved at startup:

```python
def auto_backend() -> str:
    if platform.machine() not in ("x86_64", "AMD64"):
        return "cpu"
    if "GenuineIntel" in cpuinfo_vendor() and openvino_ep_available():
        return "openvino"
    return "cpu"
```

Log the resolved backend and per-stage warm-up latency at worker startup — that line is the first
thing you will want in a support ticket.

One caveat worth knowing before you commit: OpenVINO EP compiles the graph at first load, which adds
5–20 s to worker startup. Enable its model cache (`cache_dir`) so restarts are fast, and warm up
each model with 2–3 dummy inferences before entering the loop, so the first real vehicle is not the
one that pays the compile cost.

---

## J. Multi-frame validation algorithm

**Goal:** given N noisy reads of one track, emit one plate plus a calibrated confidence, or emit
nothing.

### Per-read quality weight

Every accepted read gets a weight before it ever votes:

```
w = (rec_conf ^ 1.5)                     # recognizer confidence, sharpened
  * (0.5 + 0.5 * plate_det_conf)         # plate detector agreement
  * q                                    # plate crop quality, 0..1
  * g                                    # grammar factor: 1.0 valid IN format,
                                         #                 0.6 partially valid, 0.35 invalid
```

with plate-crop quality

```
q = clamp01( 0.40 * sat(plate_width_px / 110)        # resolution: <45px is unusable
           + 0.30 * sat(var_laplacian / 150)          # sharpness
           + 0.15 * aspect_score(w/h vs 2.0..5.5)     # geometry sanity (1-line plates)
           + 0.15 * exposure_score(mean, std) )       # not blown out, not black
```

### The algorithm

```python
def validate(self, state: TrackState) -> Optional[FinalPlate]:
    reads = [r for r in state.reads if r.weight >= self.cfg.min_read_weight]
    if len(reads) < self.cfg.min_reads:                 # default 3
        return None

    # ---- Stage 1: whole-string weighted vote -------------------------------
    votes: dict[str, float] = defaultdict(float)
    for r in reads:
        votes[r.text] += r.weight
    total = sum(votes.values())
    best, best_w = max(votes.items(), key=lambda kv: kv[1])
    support = best_w / total
    runner_up = max((w for t, w in votes.items() if t != best), default=0.0) / total

    if support >= self.cfg.strong_support and support - runner_up >= self.cfg.margin:
        candidate = best                                # 0.60 / 0.15 defaults
    else:
        # ---- Stage 2: positional character vote ---------------------------
        # Only strings of the modal length vote per-position; differing
        # lengths usually mean a truncated or merged read, not a char error.
        modal_len = weighted_mode(len(r.text) for r in reads)     # weight-aware
        pool = [r for r in reads if len(r.text) == modal_len]
        if not pool:
            return None
        candidate = ""
        per_pos_conf = []
        for i in range(modal_len):
            col: dict[str, float] = defaultdict(float)
            for r in pool:
                cc = r.per_char_confidence[i] if i < len(r.per_char_confidence) else r.confidence
                col[r.text[i]] += r.weight * cc
            ch, chw = max(col.items(), key=lambda kv: kv[1])
            candidate += ch
            per_pos_conf.append(chw / sum(col.values()))
        support = min(per_pos_conf)          # the plate is as strong as its weakest char

    # ---- Stage 3: grammar repair -------------------------------------------
    # Apply the OCR confusion map ONLY where it turns an invalid plate valid,
    # and only at positions the Indian format says must be alpha or must be digit.
    #   digit->alpha: 0->O/D  1->I  2->Z  4->A  5->S  6->G  8->B
    #   alpha->digit: O/D->0  I/L->1  Z->2  A->4  S->5  G->6  B->8
    repaired, fixes = self.grammar.repair(candidate)
    if repaired:
        candidate, corrections = repaired, fixes

    grammar_ok = self.grammar.is_valid(candidate)

    # ---- Stage 4: registry snapping (optional, logged) ----------------------
    # If the candidate is edit-distance 1 from exactly ONE registered plate and
    # the differing pair is in the confusion map, snap to it. Never snap onto a
    # blacklisted plate — that must require a clean read.
    if self.cfg.registry_snap and support < self.cfg.snap_below:
        hit = self.registry.unique_confusable_neighbour(candidate)
        if hit and hit.status != "blacklist":
            candidate, corrections = hit.plate_number, corrections + ["registry-snap"]

    # ---- Stage 5: accept / reject ------------------------------------------
    conf = 0.55 * support + 0.30 * weighted_mean_rec_conf(reads) + 0.15 * best_quality(reads)
    if not grammar_ok:
        conf *= 0.7
    if conf < self.cfg.min_final_confidence:            # default 0.55
        state.mark_disputed()        # still stored, flagged for operator review
        return None if self.cfg.drop_low_confidence else FinalPlate(..., grammar_valid=False)

    return FinalPlate(text=candidate, confidence=conf, support=support,
                      read_count=len(reads), distinct_variants=len(votes),
                      grammar_valid=grammar_ok, corrections=corrections)
```

### Worked example (your case)

```
Frame 1  UP32AB1234  rec 0.91  det 0.88  q 0.72  ->  w 0.55
Frame 2  UP32AB1234  rec 0.88  det 0.90  q 0.80  ->  w 0.58
Frame 3  UP32A81234  rec 0.62  det 0.71  q 0.41  ->  w 0.16   (grammar: pos5 must be alpha -> g=0.6)
Frame 4  UP32AB1234  rec 0.93  det 0.92  q 0.85  ->  w 0.66

Stage 1: UP32AB1234 = 1.79, UP32A81234 = 0.16, total 1.95
         support = 0.918, margin = 0.836  -> strong, accept directly
Stage 3: grammar valid (AA NN AA NNNN)
Final:   UP32AB1234, confidence 0.89, read_count 4, variants 2
```

Note that even had frame 3 been the single highest-confidence read, it loses: the grammar factor
demotes it (position 5 of an Indian plate is alphabetic, so `8` is structurally wrong) and the two
agreeing reads outweigh it. That is exactly the failure mode the current
`max(count, conf)` tie-break in `VehicleTrack._best_entry()` does not defend against.

### Indian plate grammar

```
Standard:    AA NN A(A|AA)? NNNN     e.g. UP32AB1234, DL8CAF5010, MH12DE1433
BH series:   NN BH NNNN AA           e.g. 21BH2345AA
Old/small:   AA NN A NNN
Military:    NN A NNNNNN A
```

Implement as an ordered list of compiled regexes over the normalized string, each with a positional
type mask (`A` = alpha, `N` = digit) used by the repair step. Also validate the state code against a
known set (`UP, DL, MH, KA, ...`) — a two-letter prefix outside that set is a strong reject signal
and catches a whole class of misreads cheaply.

### Trigger policy

Finalize a track when **any** of: virtual line crossed; centroid left the ROI; track retired by the
tracker; or `max_track_seconds` elapsed. Do *not* finalize on a fixed frame count — vehicles queue
at gates and can dwell for a minute.

---

## K. API surface

`/api/v1`, JWT bearer, permission-checked per endpoint.

```
Auth
  POST   /auth/login                     -> access + refresh
  POST   /auth/refresh
  POST   /auth/logout
  GET    /auth/me

Dashboard
  GET    /dashboard/stats?date=          -> totals, entries, exits, registered, unknown, alerts
  GET    /dashboard/trend?days=7&group=hour
  GET    /dashboard/recent?limit=20
  GET    /dashboard/camera-status

Events
  GET    /events                         ?from&to&camera_id&direction&status&plate&min_conf
                                          &page&size&sort
  GET    /events/{id}
  GET    /events/{id}/reads              -> per-frame evidence (why this plate)
  PATCH  /events/{id}                    -> operator correction of plate/status (audited)
  DELETE /events/{id}
  POST   /internal/events                -> WORKER INGEST (shared secret, idempotent on event_uid)
  WS     /ws/events                      -> live push

Vehicles / registry
  GET    /vehicles                       ?q&status&resident_id&page
  POST   /vehicles
  GET    /vehicles/{id}
  PUT    /vehicles/{id}
  DELETE /vehicles/{id}
  POST   /vehicles/import                -> Excel/CSV bulk upload, returns row-level errors
  GET    /vehicles/template.xlsx
  GET    /vehicles/{id}/history          -> entry/exit timeline
  GET    /vehicles/lookup?plate=         -> registry match preview (used by the correction UI)

Residents
  GET/POST/PUT/DELETE /residents
  GET    /residents/{id}/vehicles

Cameras
  GET/POST/PUT/DELETE /cameras
  GET    /cameras/{id}/health            -> fps, last frame, stage timings, worker pid
  POST   /cameras/{id}/test              -> probe RTSP, return a single frame + resolution
  GET    /cameras/{id}/snapshot          -> still image for the ROI editor
  GET    /cameras/{id}/stream            -> MJPEG preview (served from Redis, not from a model)
  GET/PUT /cameras/{id}/zones            -> ROI polygons + virtual lines (normalized coords)
  POST   /cameras/{id}/restart

Alerts
  GET    /alerts                         ?status&severity&from&to&rule
  POST   /alerts/{id}/acknowledge
  POST   /alerts/{id}/resolve
  GET/PUT /alert-rules
  POST   /alert-rules/{id}/test          -> dry-run against last N events

Reports
  GET    /reports/definitions
  POST   /reports/run                    -> {report_id} (async for big ranges)
  GET    /reports/{report_id}/status
  GET    /reports/{report_id}/download   -> .xlsx
  GET    /reports/export                 -> synchronous small export (keeps today's route working)

Users / admin
  GET/POST/PUT/DELETE /users
  GET/PUT /roles
  GET/PUT /settings                      -> thresholds, retention, SMTP, cooldowns
  POST   /settings/smtp/test
  GET    /logs                           -> activity log
  GET    /health                         -> db, redis, disk, workers[], model versions
```

### Excel report columns (as specified)

`Event ID | Date | Time | Plate Number | Vehicle Type | Camera | Location | Entry/Exit |
Vehicle Status | Resident/Owner Name | Flat Number | Detection Confidence | Recognition Confidence |
Read Count | Reviewed By`

Use `openpyxl` in `write_only` mode with a server-side cursor, so a 200k-row year-end export does
not put the whole table in RAM on a 8 GB edge box.

### Frontend pages → endpoints

| Page | Primary endpoints |
|---|---|
| Login | `/auth/login` |
| Dashboard | `/dashboard/*`, `WS /ws/events` |
| Live Events | `/events`, `WS /ws/events`, `/cameras/{id}/stream` |
| Vehicles | `/vehicles`, `/vehicles/import`, `/residents` |
| Vehicle History | `/vehicles/lookup`, `/vehicles/{id}/history` |
| Cameras | `/cameras`, `/cameras/{id}/zones`, `/snapshot`, `/health` |
| Alerts | `/alerts`, `/alert-rules` |
| Reports | `/reports/*` |
| Users | `/users`, `/roles` |
| Settings | `/settings`, `/settings/smtp/test` |

The **ROI / virtual-line editor** deserves calling out as the highest-value frontend component: a
canvas over `/cameras/{id}/snapshot` where the operator draws a polygon and a directed line, saved
as normalized coordinates. Site accuracy depends more on this being drawn well than on any model
choice, so make it a first-class screen, not a JSON textarea.
