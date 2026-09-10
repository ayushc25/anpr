"""Compatibility shim.

The engine, session factory and declarative Base now live in ``app.db``. This
module re-exports them so the prototype's routers and seed script keep working
while they are migrated. New code should import from ``app.db.session`` and
``app.db.base`` directly.
"""
from .db.base import Base
from .db.session import SessionLocal, engine, get_db

__all__ = ["Base", "SessionLocal", "engine", "get_db"]
