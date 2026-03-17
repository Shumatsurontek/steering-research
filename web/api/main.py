"""FastAPI application for the Steering Arena."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .deps import ModelManager
from .routers import benchmarks, generate, models, vectors

# ── Logging ─────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)-24s │ %(message)s"
LOG_DATE = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE,
    stream=sys.stdout,
)

# Kill ALL duplicate/noisy loggers — we handle request logging ourselves
for name in ("uvicorn", "uvicorn.access", "uvicorn.error",
             "httpcore", "httpx", "urllib3",
             "transformers", "transformers.tokenization_utils_base"):
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("steering.api")


# ── App ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Steering Arena API")
    app.state.manager = ModelManager()
    logger.info("ModelManager initialized (device=%s)", app.state.manager.device)
    yield
    logger.info("Shutting down")


app = FastAPI(title="Steering Arena API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ──────────────────────────────────────────────
# Only log API calls, skip static file serving noise

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api"):
        elapsed = (time.time() - start) * 1000
        logger.info("%s %s → %d (%.0fms)", request.method, path, response.status_code, elapsed)
    return response


# ── Routes ──────────────────────────────────────────────────────────────────

app.include_router(models.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(vectors.router, prefix="/api")
app.include_router(benchmarks.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Serve frontend static files in production ───────────────────────────────

class SPAStaticFiles(StaticFiles):
    """StaticFiles that properly rejects WebSocket connections."""

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await receive()
            await send({"type": "websocket.close", "code": 1000})
            return
        if scope["type"] != "http":
            return
        await super().__call__(scope, receive, send)


FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", SPAStaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    logger.info("Serving frontend from %s", FRONTEND_DIST)
