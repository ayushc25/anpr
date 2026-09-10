"""v1 API router.

Currently mounts only the worker-facing endpoints. The prototype's routers are
still mounted directly on the app at their original paths and are being moved
here one at a time, so existing frontend calls keep working throughout.
"""
from fastapi import APIRouter

from .endpoints import internal

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(internal.router)
