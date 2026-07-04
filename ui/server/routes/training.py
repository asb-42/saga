"""API endpoints for training runs and metrics."""

from fastapi import APIRouter, HTTPException
from ..data_ingestion import (
    get_training_runs_from_tensorboard,
    get_trainer_state,
    get_all_training_metrics,
)

router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/runs")
async def list_training_runs():
    """List all training runs from TensorBoard events."""
    return {"runs": get_training_runs_from_tensorboard()}


@router.get("/runs/{checkpoint_type}")
async def get_training_run(checkpoint_type: str):
    """Get detailed training info for a specific checkpoint type."""
    state = get_trainer_state(checkpoint_type)
    if not state:
        raise HTTPException(status_code=404, detail=f"No training state found for {checkpoint_type}")

    # Extract summary
    summary = {
        "checkpoint_type": checkpoint_type,
        "total_steps": state.get("global_step", 0),
        "total_epochs": state.get("epoch", 0),
        "best_metric": None,
        "log_count": len(state.get("log_history", [])),
    }

    # Find best metric
    if "best_metric" in state:
        summary["best_metric"] = state["best_metric"]

    return {
        "summary": summary,
        "full_state": state,
    }


@router.get("/metrics")
async def list_all_metrics():
    """List all training metrics from all checkpoint types."""
    return {"metrics": get_all_training_metrics()}


@router.get("/metrics/{checkpoint_type}")
async def get_metrics(checkpoint_type: str):
    """Get training metrics for a specific checkpoint type."""
    state = get_trainer_state(checkpoint_type)
    if not state:
        raise HTTPException(status_code=404, detail=f"No metrics found for {checkpoint_type}")

    return {
        "checkpoint_type": checkpoint_type,
        "log_history": state.get("log_history", []),
    }


@router.get("/metrics/{checkpoint_type}/{metric_name}")
async def get_metric_series(checkpoint_type: str, metric_name: str):
    """Get a specific metric series for a checkpoint type."""
    state = get_trainer_state(checkpoint_type)
    if not state:
        raise HTTPException(status_code=404, detail=f"No metrics found for {checkpoint_type}")

    series = []
    for entry in state.get("log_history", []):
        if metric_name in entry:
            series.append({
                "step": entry.get("step", 0),
                "epoch": entry.get("epoch"),
                "value": entry[metric_name],
            })

    return {
        "checkpoint_type": checkpoint_type,
        "metric_name": metric_name,
        "series": series,
    }
