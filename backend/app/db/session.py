"""Engine and session factory.

``pool_pre_ping`` matters more here than in a typical web app: an edge box
sits idle overnight and Postgres or the OS will have dropped the connection by
morning, and the first vehicle of the day must not be the one that discovers
that.
"""
from __future__ import annotations

import logging
from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker

from ..core.config import get_settings

logger = logging.getLogger("anpr.db")

engine: Optional[Engine] = None
SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def init_db(database_url: str) -> Optional[Engine]:
    """Dynamically initialize or reconfigure the database engine and sessionmaker."""
    global engine
    if not database_url:
        logger.warning("init_db called with empty database_url.")
        engine = None
        return None

    try:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass

        new_engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=1800,
        )
        # Verify connection
        with new_engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))

        engine = new_engine
        SessionLocal.configure(bind=engine)
        logger.info("Database connection successfully initialized.")
        return engine
    except Exception as exc:
        logger.warning(f"Could not connect to database with provided URL: {exc}")
        # Even if connection check fails, keep engine bound so callers can retry or handle
        engine = new_engine if 'new_engine' in locals() else None
        if engine is not None:
            SessionLocal.configure(bind=engine)
        return engine


# Initial attempt on boot
_settings = get_settings()
if _settings.database_url:
    try:
        init_db(_settings.database_url)
    except Exception as exc:
        logger.warning(f"Initial database setup skipped or failed: {exc}")


def get_db():
    """FastAPI dependency."""
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not configured. Please complete system setup at /setup.",
        )
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
