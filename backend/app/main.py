import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logging.getLogger("anpr").setLevel(logging.INFO)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sqlalchemy import text, inspect

from .database import Base, engine
from .config import STORAGE_DIR
from .permissions import PERMISSION_KEYS
from .routers import auth, users, locations, cameras, vehicles, events, dashboard, reports, logs
from .services import camera_manager


def _run_lightweight_migrations():
    """No migration framework is in place yet, so new nullable columns are
    added by hand here with IF NOT EXISTS - create_all() only creates
    missing tables, it never alters existing ones."""
    statements = [
        "ALTER TABLE events ADD COLUMN IF NOT EXISTS vehicle_color VARCHAR(32)",
        "ALTER TABLE events ADD COLUMN IF NOT EXISTS plate_color VARCHAR(32)",
        "ALTER TABLE vehicles ADD COLUMN IF NOT EXISTS flat_number VARCHAR(32)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS role_name VARCHAR(64) NOT NULL DEFAULT ''",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS permissions JSON NOT NULL DEFAULT '[]'::json",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

    # One-time backfill: the fixed admin/manager/guard `role` enum column is
    # being replaced by a free-text role_name plus a per-user permissions
    # list, so translate any pre-existing rows before dropping the old column.
    inspector = inspect(engine)
    columns = {c["name"] for c in inspector.get_columns("users")}
    if "role" in columns:
        all_keys = PERMISSION_KEYS
        manager_keys = [k for k in PERMISSION_KEYS if k != "users"]
        guard_keys = ["dashboard", "live", "events", "vehicles"]
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE users SET role_name = 'Administrator', permissions = :perms "
                "WHERE role = 'admin' AND role_name = ''"
            ), {"perms": json.dumps(all_keys)})
            conn.execute(text(
                "UPDATE users SET role_name = 'Manager', permissions = :perms "
                "WHERE role = 'manager' AND role_name = ''"
            ), {"perms": json.dumps(manager_keys)})
            conn.execute(text(
                "UPDATE users SET role_name = 'Guard', permissions = :perms "
                "WHERE role = 'guard' AND role_name = ''"
            ), {"perms": json.dumps(guard_keys)})
            conn.execute(text("ALTER TABLE users DROP COLUMN role"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _run_lightweight_migrations()
    camera_manager.start_all_cameras()
    yield
    camera_manager.stop_all_cameras()


app = FastAPI(title="ANPR System API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(STORAGE_DIR)), name="media")

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(locations.router)
app.include_router(cameras.router)
app.include_router(vehicles.router)
app.include_router(events.router)
app.include_router(dashboard.router)
app.include_router(reports.router)
app.include_router(logs.router)


@app.get("/")
def root():
    return {"status": "ok", "service": "ANPR System API"}
