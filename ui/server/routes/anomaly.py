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


@router.get("/prompts/stream")
async def prompts_stream(es=Depends(get_event_stream)):
    """Stream prompt analysis events via SSE."""
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        es.stream("prompts", heartbeat_interval=15),
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


@router.get("/history")
async def get_anomaly_history():
    """Get historical anomaly detection results from evaluation data."""
    from ..data_ingestion import RESULTS_DIR, CHECKPOINTS_DIR, load_json
    import json

    # Load threshold config
    threshold_path = CHECKPOINTS_DIR / "anomaly_threshold.json"
    threshold = load_json(threshold_path) if threshold_path.exists() else None

    # Load poisoning eval results
    eval_results = {}
    for eval_name in ["poisoning", "poisoning_answer_level"]:
        report_path = RESULTS_DIR / eval_name / "report.json"
        if report_path.exists():
            eval_results[eval_name] = load_json(report_path)

    # Count detections from per-sample results
    detections = {"total": 0, "detected": 0, "missed": 0, "false_positives": 0}
    per_sample_path = RESULTS_DIR / "poisoning_answer_level" / "per_sample_results.jsonl"
    if per_sample_path.exists():
        with open(per_sample_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    detections["total"] += 1
                    is_poisoned = entry.get("is_poisoned", False)
                    detected = any([
                        entry.get("detected_by_pattern"),
                        entry.get("detected_by_trigger"),
                        entry.get("detected_by_format"),
                    ])
                    if is_poisoned and detected:
                        detections["detected"] += 1
                    elif is_poisoned and not detected:
                        detections["missed"] += 1
                    elif not is_poisoned and detected:
                        detections["false_positives"] += 1

    return {
        "threshold": threshold,
        "eval_results": eval_results,
        "detections": detections,
        "status": "completed" if eval_results else "no_data",
        "last_updated": None,
    }
