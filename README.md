# ANPR System

On-premises ANPR (Automatic Number Plate Recognition) web application: RTSP camera ingestion,
YOLO vehicle detection, OCR plate reading, registered-vehicle matching, alerts, dashboard,
reports and user management — per `ANPR_Phase_1_Requirements_and_Implementation_Plan_Updated.pdf`.

## Stack
- **Backend**: Python, FastAPI, SQLAlchemy, PostgreSQL, OpenCV, Ultralytics YOLO, EasyOCR
- **Frontend**: React (Vite), Tailwind CSS, Recharts

## Project layout
```
backend/    FastAPI app, detection/streaming services, seed script
frontend/   React app (dashboard, live view, events, vehicles, cameras, users, reports, logs)
.env        database_url (shared by backend)
yolo26n.pt  YOLO model used for vehicle detection
```

## First-time setup

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\pip install -r requirements.txt      # Windows
venv\Scripts\python seed.py                       # creates tables + default users
venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```
Default logins seeded: `admin/admin123`, `manager/manager123`, `guard/guard123` — change these immediately in a real deployment.

### Frontend
```bash
cd frontend
npm install
npm run dev       # http://localhost:5173, proxies /api to http://localhost:8000
```

## Adding a camera
Go to **Cameras** in the UI (admin/manager) and add a name + RTSP URL
(e.g. `rtsp://user:pass@192.168.1.50:554/stream1`) and a direction (IN/OUT).
The backend immediately starts a background worker that:
1. Reads frames from the RTSP stream (OpenCV).
2. Runs YOLO (`yolo26n.pt`) vehicle detection every ~1s.
3. Crops a heuristic plate region and reads it with EasyOCR.
4. Matches the plate against the `vehicles` table (registered / whitelist / blacklist / unknown).
5. Saves an event (image + metadata) and updates the live dashboard.
6. Streams the annotated feed to the **Live View** page as MJPEG.

**Accuracy note:** `yolo26n.pt` is a general-purpose COCO detector — it finds vehicles but not
plates specifically. Plate OCR here uses a heuristic crop (lower ~45% of the vehicle box). For
production accuracy, swap in a plate-specific detector in `backend/app/services/detection.py`.

## Roles
- **admin** — full access: users, cameras, vehicles, settings, logs
- **manager** — manages vehicles/cameras/reports/logs, no user management
- **guard** — dashboard, live view, events, vehicle search only (read-only)

## Key API routes
- `POST /auth/login`, `GET /auth/me`
- `GET/POST/PUT/DELETE /users`, `/cameras`, `/vehicles`, `/locations`
- `GET /events`, `GET /cameras/{id}/stream` (MJPEG)
- `GET /dashboard/stats`, `/dashboard/trend`, `/dashboard/latest`
- `GET /reports/export` (Excel)
- `GET /logs` (activity log, 90-day retention)

Interactive API docs: `http://localhost:8000/docs`
