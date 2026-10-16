"""FastAPI application: mounts routes, static files, and lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import routes_api, routes_clips, routes_ws

if TYPE_CHECKING:
    from ..bus import EventBus

_bus: EventBus | None = None


def create_app(bus: EventBus) -> FastAPI:
    global _bus
    _bus = bus

    # Initialise sub-modules with bus reference
    routes_clips.init(bus)
    routes_api.init(bus)
    routes_ws.init(bus)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await routes_clips.restore_clips(bus)
        yield

    application = FastAPI(lifespan=lifespan)

    _static = Path(__file__).parent.parent / "static"
    application.mount("/static", StaticFiles(directory=str(_static)), name="static")

    @application.get("/")
    async def index():
        return FileResponse(_static / "index.html")

    application.include_router(routes_clips.router)
    application.include_router(routes_api.router)
    application.include_router(routes_ws.router)

    return application
