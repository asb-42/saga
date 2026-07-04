"""Metrics routes for SSE streaming and historical data."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


def get_storage():
    """Dependency: get storage."""
    from ..main import storage
    if not storage:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not initialized")
    return storage


def get_event_stream():
    """Dependency: get event stream."""
    from ..main import event_stream
    if not event_stream:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not initialized")
    return event_stream


@router.get("/stream")
async def metrics_stream(es=Depends(get_event_stream)):
    """Stream all metrics via SSE."""
    return StreamingResponse(
        es.stream("metrics", heartbeat_interval=15),
        media_type="text/event-stream",
    )


@router.get("/stream/{run_id}")
async def metrics_stream_by_run(run_id: int, es=Depends(get_event_stream)):
    """Stream metrics for a specific run via SSE."""
    return StreamingResponse(
        es.stream(f"metrics:{run_id}", heartbeat_interval=15),
        media_type="text/event-stream",
    )


@router.get("/history")
async def get_metrics_history(
    run_id: int = Query(..., description="Run ID"),
    metric_name: str | None = Query(None, description="Filter by metric name"),
    limit: int = Query(1000, ge=1, le=10000),
    store=Depends(get_storage),
):
    """Get historical metrics for a run."""
    metrics = await store.get_metrics(
        run_id=run_id,
        metric_name=metric_name,
        limit=limit,
    )
    return {
        "run_id": run_id,
        "metrics": [
            {
                "step": m.step,
                "epoch": m.epoch,
                "name": m.metric_name,
                "value": m.metric_value,
                "timestamp": m.recorded_at,
            }
            for m in metrics
        ],
    }
