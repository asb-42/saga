"""Log streaming routes."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..config import config

router = APIRouter(prefix="/api/logs", tags=["logs"])


def get_event_stream():
    """Dependency: get event stream."""
    from ..main import event_stream
    if not event_stream:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not initialized")
    return event_stream


@router.get("/stream/{run_id}")
async def log_stream(run_id: int, es=Depends(get_event_stream)):
    """Stream logs for a specific run via SSE."""
    return StreamingResponse(
        es.stream(f"logs:{run_id}", heartbeat_interval=15),
        media_type="text/event-stream",
    )


@router.get("/stream")
async def global_log_stream(es=Depends(get_event_stream)):
    """Stream all logs from all runs via SSE."""
    return StreamingResponse(
        es.stream("logs:all", heartbeat_interval=15),
        media_type="text/event-stream",
    )


@router.get("/files")
async def list_log_files():
    """List available log files in the logs directory."""
    logs_dir = config.LOGS_DIR
    if not logs_dir.exists():
        return {"files": []}

    files = []
    for f in sorted(logs_dir.iterdir()):
        if f.is_file() and f.suffix == ".log":
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
    return {"files": files}


@router.get("/history")
async def log_history(
    file: str = Query(..., description="Log filename (e.g. backend.log)"),
    tail: int = Query(200, ge=1, le=5000, description="Number of lines from end"),
    offset: int = Query(0, ge=0, description="Line offset from start"),
):
    """Read historical log lines from a file on disk."""
    # Security: only allow reading from the logs directory, no path traversal
    logs_dir = config.LOGS_DIR.resolve()
    file_path = (logs_dir / file).resolve()

    if not str(file_path).startswith(str(logs_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Log file '{file}' not found")

    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read log file: {e}")

    total = len(lines)

    # Apply offset + tail window
    if offset > 0:
        lines = lines[offset:]
    if tail > 0:
        lines = lines[-tail:]

    return {
        "file": file,
        "total": total,
        "offset": offset,
        "returned": len(lines),
        "lines": lines,
    }
