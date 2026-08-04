"""CORS configuration for browser-based SatNav frontend clients."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

DEFAULT_CORS_ORIGINS = ("http://127.0.0.1:5173",)


def cors_origins_from_environment() -> list[str]:
    """Parse SATNAV_CORS_ORIGINS as a comma-separated browser origin list."""
    raw = os.environ.get("SATNAV_CORS_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    origins: list[str] = []
    for part in raw.split(","):
        origin = part.strip()
        if origin:
            origins.append(origin)
    return origins


def configure_cors(app: FastAPI) -> list[str]:
    """Attach CORSMiddleware when at least one origin is configured."""
    origins = cors_origins_from_environment()
    if not origins:
        return origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return origins
