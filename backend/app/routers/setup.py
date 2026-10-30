from __future__ import annotations

import logging
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from ..core.config import get_settings, reload_settings, ROOT_DIR
from ..db.session import init_db, SessionLocal, engine
from ..security import hash_password
from ..permissions import PERMISSION_KEYS
from .. import models

logger = logging.getLogger("anpr.setup")

router = APIRouter(prefix="/setup", tags=["setup"])


class TestDBRequest(BaseModel):
    database_url: str = Field(..., description="PostgreSQL connection string")


class SetupInitRequest(BaseModel):
    database_url: str = Field(..., description="PostgreSQL connection string")
    superadmin_username: str = Field("risingit", min_length=3, max_length=64)
    superadmin_password: str = Field(..., min_length=6)
    superadmin_full_name: str = Field("Rising IT Super Admin", max_length=128)
    client_admin_username: Optional[str] = Field("admin", max_length=64)
    client_admin_password: Optional[str] = Field(None, min_length=6)
    client_admin_full_name: Optional[str] = Field("Client Admin", max_length=128)


def _check_system_status() -> dict:
    settings = get_settings()
    has_db_url = bool(settings.database_url and settings.database_url.strip())
    db_connected = False
    has_users = False

    if has_db_url:
        try:
            test_engine = engine or create_engine(settings.database_url, connect_args={"connect_timeout": 3})
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_connected = True
        except Exception:
            db_connected = False

    if db_connected:
        try:
            db = SessionLocal()
            try:
                user_count = db.query(models.User).count()
                has_users = user_count > 0
            except Exception:
                # Table might not exist yet
                has_users = False
            finally:
                db.close()
        except Exception:
            has_users = False

    setup_required = (not has_db_url) or (not db_connected) or (not has_users)

    return {
        "setup_required": setup_required,
        "database_configured": has_db_url,
        "database_connected": db_connected,
        "has_users": has_users,
    }


def _update_env_file(db_url: str) -> None:
    env_path = ROOT_DIR / ".env"
    lines = []
    found = False
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("database_url=") or stripped.startswith("DATABASE_URL="):
                    lines.append(f"database_url={db_url}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.insert(0, f"database_url={db_url}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


@router.get("/status")
def setup_status():
    """Returns the current deployment & setup state."""
    return _check_system_status()


@router.post("/test-db")
def test_database_connection(req: TestDBRequest):
    """Verifies that the provided PostgreSQL connection string connects successfully."""
    url = req.database_url.strip()
    if not (url.startswith("postgresql://") or url.startswith("postgresql+psycopg2://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database URL must start with 'postgresql://' or 'postgresql+psycopg2://'",
        )

    try:
        test_engine = create_engine(url, connect_args={"connect_timeout": 5})
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        test_engine.dispose()
        return {"success": True, "message": "Successfully connected to PostgreSQL database."}
    except Exception as exc:
        logger.warning(f"Database connection test failed for {url}: {exc}")
        return {"success": False, "error": str(exc)}


@router.post("/initialize")
def initialize_system(req: SetupInitRequest):
    """Initializes the database schema, seeds the Super Admin & Client Admin, and unlocks normal operation."""
    current_status = _check_system_status()
    if not current_status["setup_required"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="System has already been configured and initialized with active users. Setup is locked.",
        )

    url = req.database_url.strip()
    # Step 1: Verify DB connectivity
    try:
        test_engine = create_engine(url, connect_args={"connect_timeout": 5})
        with test_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        test_engine.dispose()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not connect to database: {exc}",
        )

    # Step 2: Persist DB URL to .env & reload settings
    try:
        _update_env_file(url)
        reload_settings()
        init_db(url)
    except Exception as exc:
        logger.error(f"Failed to update environment: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save database configuration: {exc}",
        )

    # Step 3: Run Alembic migrations programmatically
    try:
        from alembic import command
        from alembic.config import Config

        alembic_ini_path = str(ROOT_DIR / "backend" / "alembic.ini")
        alembic_cfg = Config(alembic_ini_path)
        alembic_cfg.set_main_option("script_location", str(ROOT_DIR / "backend" / "alembic"))
        alembic_cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))

        command.upgrade(alembic_cfg, "head")
        logger.info("Alembic schema migrations executed successfully.")
    except Exception as exc:
        logger.error(f"Database schema migration failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run database migrations: {exc}",
        )

    # Step 4: Seed initial accounts and default data
    try:
        db = SessionLocal()
        try:
            # 1. Super Admin (Rising IT)
            superadmin = db.query(models.User).filter(models.User.username == req.superadmin_username).first()
            if not superadmin:
                superadmin = models.User(
                    username=req.superadmin_username,
                    full_name=req.superadmin_full_name,
                    password_hash=hash_password(req.superadmin_password),
                    role_name="Super Administrator",
                    permissions=PERMISSION_KEYS,
                    is_active=True,
                )
                db.add(superadmin)
            else:
                superadmin.password_hash = hash_password(req.superadmin_password)
                superadmin.permissions = PERMISSION_KEYS
                superadmin.is_active = True

            # 2. Client Admin (if provided)
            if req.client_admin_password and req.client_admin_username:
                client_admin = db.query(models.User).filter(models.User.username == req.client_admin_username).first()
                if not client_admin:
                    client_admin = models.User(
                        username=req.client_admin_username,
                        full_name=req.client_admin_full_name or "Client Admin",
                        password_hash=hash_password(req.client_admin_password),
                        role_name="Administrator",
                        permissions=PERMISSION_KEYS,
                        is_active=True,
                    )
                    db.add(client_admin)
                else:
                    client_admin.password_hash = hash_password(req.client_admin_password)
                    client_admin.permissions = PERMISSION_KEYS
                    client_admin.is_active = True

            # 3. Default Locations
            if db.query(models.Location).count() == 0:
                db.add(models.Location(name="Main Gate", description="Primary entrance/exit"))
                db.add(models.Location(name="North Gate", description="Secondary entrance"))

            db.commit()
            logger.info("Default seed users and locations committed successfully.")
        finally:
            db.close()
    except Exception as exc:
        logger.error(f"Failed to seed initial accounts: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database migrated but failed to create initial admin accounts: {exc}",
        )

    return {
        "success": True,
        "message": "System setup successfully initialized. You can now log in.",
    }

