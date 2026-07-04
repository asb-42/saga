"""SAGA Research Lab Dashboard — FastAPI Server."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import config
from .event_stream import EventStream
from .file_watcher import FileWatcher
from .models import HealthResponse
from .process_manager import ProcessManager
from .storage import Storage

# --- Global state (initialized in lifespan) ---

storage: Storage | None = None
event_stream: EventStream | None = None
process_manager: ProcessManager | None = None
file_watcher: FileWatcher | None = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global storage, event_stream, process_manager, file_watcher, _start_time

    # Startup
    _start_time = time.time()
    config.ensure_dirs()

    storage = Storage(config.PROJECT_ROOT / "ui" / "saga_lab.db")
    await storage.connect()

    event_stream = EventStream()
    process_manager = ProcessManager(storage, event_stream)

    # Start file watcher for log directories
    file_watcher = FileWatcher(
        event_stream,
        watch_dirs=[
            config.LOGS_DIR,
            config.RESULTS_DIR,
            config.CHECKPOINTS_DIR,
        ],
    )
    file_watcher.start()

    yield

    # Shutdown
    if process_manager:
        await process_manager.shutdown()
    if file_watcher:
        file_watcher.stop()
    if event_stream:
        await event_stream.shutdown()
    if storage:
        await storage.close()


# --- App ---

app = FastAPI(
    title="SAGA Research Lab",
    description="Interactive dashboard for SAGA AI ensemble research",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for LAN access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include routers ---

from .routes import pipeline_router, metrics_router, logs_router, anomaly_router
from .routes.models import router as models_router
from .routes.training import router as training_router
from .routes.benchmarks import router as benchmarks_router
from .routes.system import router as system_router

app.include_router(pipeline_router)
app.include_router(metrics_router)
app.include_router(logs_router)
app.include_router(anomaly_router)
app.include_router(models_router)
app.include_router(training_router)
app.include_router(benchmarks_router)
app.include_router(system_router)


# --- Health check ---

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Server health check."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        uptime=time.time() - _start_time,
    )


# --- Run server ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
    )
