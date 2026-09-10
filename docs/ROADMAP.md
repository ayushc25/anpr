# ANPR — Implementation Roadmap

Companion to [ARCHITECTURE.md](ARCHITECTURE.md). Ordered so that every step leaves the system
working. The existing `backend/` prototype is not thrown away — it is strangled module by module.

**Explicitly out of Phase 1:** gate/barrier hardware, relay/GPIO control, ANPR-triggered boom
integration, LPR-to-intercom. Phase 1 observes and reports; it does not actuate.

---

## Phase 0 — Baseline and measurement (3–5 days)

You cannot optimize or claim accuracy without a yardstick. Do this before writing new pipeline code.

1. `scripts/benchmark.py` — per-stage p50/p95 ms on the target edge box, with the current models.
2. Record **10–15 real clips** at the actual gate: day, night, backlit, rain, two-wheeler,
   queue of 3 vehicles, one motorcycle with a stacked plate. 30–60 s each, sub-stream and main.
3. Hand-label the ground-truth plate for every vehicle in those clips → `tests/fixtures/clips/`.
4. `scripts/eval_pipeline.py` — runs a config end-to-end over the clip set and prints
   **plate accuracy (exact), character error rate, miss rate, false-event rate, ms/frame.**
5. Record today's numbers. Every later change is judged against them.

**Exit criteria:** a one-command accuracy+speed report you trust.

---

## Phase 1 — Production core (the deliverable)

Target: **4–8 cameras on one Intel box, ≥93% exact-plate accuracy on the clip set, entry/exit
events, registry matching, alerts, dashboard, Excel reports** — all CPU-only, all on-prem.

### 1.1 Foundations (week 1)

- Alembic in, `_run_lightweight_migrations()` out. Baseline migration from the current schema.
- Split `models.py` → `models/`, `schemas.py` → `schemas/`, routers → `api/v1/endpoints/`.
- Introduce `repositories/` and move every query out of routers and out of `camera_manager`.
- `core/config.py` on pydantic-settings + `configs/*.yaml`; delete hard-coded constants.
- Structured logging with `camera_id` / `track_id` in every pipeline log line.
- Remove plaintext RTSP credentials from API responses.

### 1.2 The AI library (week 1–2)

- `ai/types.py`, `ai/registry.py`, `ai/inference/{backend,onnxrt_backend}.py`.
- `scripts/export_onnx.py`: `yolo26n.pt` → ONNX; the plate `.pt` → ONNX at 320.
- `yolo_onnx.py` and `yolo_plate_onnx.py` with numpy letterbox + NMS (drop the Ultralytics runtime
  dependency from the serving path — it pulls torch onto an edge box for no benefit).
- `plate_recognizer/ppocr_onnx.py` as the new default; keep `easyocr_legacy.py` behind config so
  you can A/B on the clip set and prove the improvement.
- `plate_recognizer/postprocess.py`: Indian grammar + confusion map + state-code validation.
- `quality/plate_quality.py`.
- **Gate:** `eval_pipeline.py` shows ONNX+PP-OCR ≥ current EasyOCR path on accuracy and ≥3x on speed.

### 1.3 Video and pipeline (week 2–3)

- `video/rtsp_reader.py` with the drop queue and reconnect backoff; sub-stream support on `cameras`.
- `video/roi.py` + `video/line_crossing.py` with normalized coordinates.
- `vehicle_tracker/bytetrack.py` (wrap `supervision.ByteTrack` — MIT, numpy-only, no torch).
- `video/frame_processor.py`: the full cascade with gating and track locking.
- `events/track_state.py` + `events/multi_frame_validator.py` (§J) + `events/dedupe.py`.
- Unit tests for the validator against synthetic read sequences, including your UP32AB1234 case.

### 1.4 Process architecture (week 3)

- `workers/camera_worker.py`, `workers/supervisor.py`, `cli.py`.
- `ai/inference/threading.py` thread-budget calculator; set env before numpy import.
- Redis control bus; `camera_service` publishes on CRUD; supervisor reconciles.
- `workers/ingest_client.py` + `storage/spool.py`.
- Move MJPEG preview to Redis-backed, viewer-gated.
- **Gate:** kill a worker process, confirm restart; unplug a camera, confirm reconnect and
  `is_online=false`; stop Postgres for 60 s, confirm zero lost events via the spool.

### 1.5 Events, registry, alerts (week 4)

- `services/event_service.ingest()` — one transaction, idempotent on `event_uid`.
- `residents` + `vehicles` split; `POST /vehicles/import` with row-level error reporting
  (the client will hand you a messy Excel; expect duplicates, spaces, `O`/`0`, lowercase).
- `plate_aliases` and the trigram fuzzy lookup.
- `events/rules_engine.py` + built-in rules; `alert_rules` seeded with Blacklist and Unknown.
- `alerts/channels/smtp.py` behind the `notifications` outbox, **optional and off by default**.
- WebSocket `/ws/events`.

### 1.6 Frontend (week 4–5, parallel with 1.5)

- Keep the existing pages; add/rework:
  - **ROI + virtual-line editor** on the Cameras page (highest accuracy leverage in the product).
  - **Event detail drawer** showing vehicle image, plate crop, and the `event_reads` trail —
    this is what makes the system defensible when the client disputes a read.
  - **Operator correction** on an event (plate + status), audited, feeding `plate_aliases`.
  - **Alerts** page with acknowledge/resolve.
  - **Reports** page with async export + download.
- Replace polling with the WebSocket on Dashboard and Live Events.

### 1.7 Packaging and handover (week 5)

- `deploy/docker-compose.yml` (postgres, redis, api, supervisor, nginx) **and** systemd units.
- `models/manifest.yaml` with sha256 checks at startup; refuse to run on a mismatched artifact.
- `workers/janitor.py`: retention for media, `event_reads`, activity logs; daily stats rollup.
- Backup script (pg_dump + media rsync) and a restore rehearsal.
- Handover doc: measured camera capacity on the delivered box, accuracy report, runbook.

**Phase 1 exit criteria**
- 7 consecutive days at the site with no manual restart.
- Accuracy report on site footage ≥ the agreed threshold.
- Camera add/remove from the UI with no service restart.
- Excel export of a full month under 30 s.
- Postgres and camera outages both recovered automatically.

---

## Phase 2 — Accuracy and scale (after Phase 1 is live)

- **Harvest and fine-tune.** `scripts/harvest_dataset.py` mines `event_reads` crops, prioritizing
  disputed and operator-corrected events. Fine-tune a nano plate detector and an LPRNet/CRNN
  recognizer on site data. This is where 93% becomes 97%.
- **Distil the plate detector.** Replace `license-plate-finetune-v1l.pt` (~50 MB `-l` model) with a
  nano trained on the same data — the single biggest remaining CPU win.
- **INT8 where it actually helps.** Dynamic quantization measured *slower* on the dev box (0.6-0.7x); try static PTQ with site calibration frames and keep the accuracy gate in CI.
- **OpenVINO EP** enabled and benchmarked on the customer box; auto-select at startup.
- **Two-line plate handling** for two-wheelers and commercial vehicles.
- Night-time IR tuning; per-camera model profiles (`cameras.model_profile`).
- `events` monthly partitioning + `daily_stats` rollup once volume justifies it.
- Multi-site: a central server aggregating several edge boxes (edge stays authoritative offline).

---

## Phase 3 — Actuation and integrations (explicitly deferred)

- Boom barrier / relay board control, with a manual-override and fail-open policy.
- Visitor pass workflow, pre-authorized guest plates with validity windows.
- Intercom / guard-app notification, WhatsApp channel.
- Speed estimation, wrong-way and tailgating detection.
- ANPR-triggered parking occupancy.

---

## Risk register (the ones that actually bite)

| Risk | Mitigation |
|---|---|
| Camera angle/height makes plates <45 px wide | Fix at survey time. Specify: plate ≥100 px wide at the trigger line, camera ≤30° off-axis, 1.2–1.8 m height. No model recovers from a bad mount. |
| Night glare / IR washout on retroreflective plates | Camera-side: shutter ≤1/500, WDR on, IR intensity down. Validate at 10 pm before signing off. |
| The `-l` plate model is too slow on CPU | Distil to nano (Phase 2); until then run it at 320 on vehicle crops only, and measure. |
| Client registry is dirty | Import with normalization + a row-level error report; `plate_aliases` for known misreads. |
| Two-wheelers with stacked plates | Test explicitly in Phase 0 clips; budget for it in Phase 2. |
| Camera count grows after sizing | Process-per-camera + thread budget makes this a hardware question, not a rewrite. Publish the measured per-camera cost. |
| Disk fills with event images | Janitor retention from day one, plus a disk-space check in `/health`. |
