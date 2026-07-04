"""Log streaming routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

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
