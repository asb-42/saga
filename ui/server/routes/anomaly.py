"""Anomaly alert routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

router = APIRouter(prefix="/api/anomaly", tags=["anomaly"])


def get_storage():
    """Dependency: get storage."""
    from ..main import storage
    if not storage:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return storage


def get_event_stream():
    """Dependency: get event stream."""
    from ..main import event_stream
    if not event_stream:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return event_stream


@router.get("/alerts")
async def get_alerts(
    run_id: int | None = Query(None),
    acknowledged: bool | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    store=Depends(get_storage),
):
    """Get anomaly alerts."""
    alerts = await store.get_alerts(
        run_id=run_id,
        acknowledged=acknowledged,
        limit=limit,
    )
    return {"alerts": alerts}


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, store=Depends(get_storage)):
    """Acknowledge an anomaly alert."""
    alerts = await store.get_alerts()
    if not any(a.id == alert_id for a in alerts):
        raise HTTPException(status_code=404, detail="Alert not found")
    await store.acknowledge_alert(alert_id)
    return {"status": "acknowledged"}


@router.get("/stream")
async def anomaly_stream(es=Depends(get_event_stream)):
    """Stream anomaly events via SSE."""
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        es.stream("anomaly", heartbeat_interval=15),
        media_type="text/event-stream",
    )


@router.get("/prompts/recent")
async def get_recent_prompts(
    limit: int = Query(100, ge=1, le=1000),
    anomaly_only: bool = Query(False),
    store=Depends(get_storage),
):
    """Get recent prompt analyses."""
    prompts = await store.get_recent_prompts(
        limit=limit,
        anomaly_only=anomaly_only,
    )
    return {"prompts": prompts}
