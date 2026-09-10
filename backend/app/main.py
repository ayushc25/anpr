import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.v1.router import api_router
from .core.config import get_settings
from .routers import auth, cameras, dashboard, events, locations, logs, reports, users, vehicles

logging.basicConfig(level=logging.INFO)
logging.getLogger("anpr").setLevel(logging.INFO)
logger = logging.getLogger("anpr.main")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Schema changes are Alembic's job now, not startup's.

    The previous ``_run_lightweight_migrations()`` ran ALTER TABLE statements
    here on every boot. That could add a column but never rename one, backfill
    data or roll back — all of which the Phase 1 schema needed. Run:

        cd backend && alembic upgrade head

    Camera workers are mid-migration. The legacy in-process workers still run
    here because the UI starts them on camera create/update; the replacement
    runs each camera in its own OS process under
    ``python -m backend.app.cli supervisor``. See the flag below.
    """
    _warn_if_migrations_pending()

    # The UI still drives the legacy in-process camera workers: routers/
    # cameras.py starts one on create/update, so without starting the enabled
    # ones here a restart leaves every camera dead until someone re-saves it.
    #
    # This is the last piece of inference still living inside FastAPI. Set
    # ENABLE_LEGACY_CAMERA_WORKERS=0 once the UI is switched over to the
    # supervisor (`python -m backend.app.cli supervisor`), which runs each
    # camera in its own process with the validated pipeline.
    legacy = os.getenv("ENABLE_LEGACY_CAMERA_WORKERS", "1") not in ("0", "false", "False")
    if legacy:
        from .services import camera_manager

        logger.warning(
            "starting LEGACY in-process camera workers. These use the old "
            "single-frame pipeline; run 'python -m backend.app.cli supervisor' "
            "and set ENABLE_LEGACY_CAMERA_WORKERS=0 for the validated one."
        )
        camera_manager.start_all_cameras()
    try:
        yield
    finally:
        if legacy:
            from .services import camera_manager

            camera_manager.stop_all_cameras()


def _warn_if_migrations_pending() -> None:
    try:
        from alembic.script import ScriptDirectory
        from sqlalchemy import text

        from .db.session import engine

        script = ScriptDirectory(str(settings.root_dir / "backend" / "alembic"))
        head = script.get_current_head()
        with engine.connect() as connection:
            current = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
        if current != head:
            logger.warning(
                "DATABASE SCHEMA IS OUT OF DATE (at %s, head is %s). Run: cd backend && alembic upgrade head",
                current, head,
            )
    except Exception:
        # A missing alembic_version table just means migrations have never
        # run; that is worth a hint, not a crash on startup.
        logger.warning("could not verify schema version; run 'cd backend && alembic upgrade head'")


app = FastAPI(title="ANPR System API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(settings.media_root)), name="media")

app.include_router(api_router)

# Prototype routers, still at their original paths so the existing frontend
# keeps working while they are migrated into api/v1 one at a time.
for router in (auth, users, locations, cameras, vehicles, events, dashboard, reports, logs):
    app.include_router(router.router)


@app.get("/")
def root():
    return {"status": "ok", "service": "ANPR System API", "version": app.version}


@app.get("/health")
def health():
    """Liveness plus the things that actually break on an edge box."""
    from sqlalchemy import text

    from .db.session import engine

    checks = {"database": False, "media_writable": False}
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        logger.warning("health: database unreachable", exc_info=True)
    try:
        probe = settings.media_root / ".healthcheck"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks["media_writable"] = True
    except Exception:
        logger.warning("health: media directory not writable", exc_info=True)

    return {"status": "ok" if all(checks.values()) else "degraded", "checks": checks}
