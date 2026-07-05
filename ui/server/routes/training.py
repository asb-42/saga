"""API endpoints for training runs and metrics."""

from fastapi import APIRouter, HTTPException, Query
from ..data_ingestion import (
    get_training_runs_from_tensorboard,
    get_trainer_state,
    get_all_training_metrics,
    load_json,
)

router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/runs")
async def list_training_runs():
    """List all training runs from TensorBoard events."""
    return {"runs": get_training_runs_from_tensorboard()}


@router.get("/runs/{checkpoint_type}")
async def get_training_run(checkpoint_type: str):
    """Get detailed training info for a specific checkpoint type."""
    from ..data_ingestion import CHECKPOINTS_DIR, RESULTS_DIR, load_json

    # Check if this is an evaluation result (has report.json in results dir)
    results_dir = RESULTS_DIR / checkpoint_type
    if results_dir.exists():
        report_path = results_dir / "report.json"
        if report_path.exists():
            report = load_json(report_path)
            event_files = list((results_dir / "tensorboard").glob("events.out.tfevents.*")) if (results_dir / "tensorboard").exists() else []
            per_sample_path = results_dir / "per_sample_results.jsonl"
            summary = {
                "checkpoint_type": checkpoint_type,
                "total_steps": 0,
                "total_epochs": 0,
                "best_metric": None,
                "log_count": 0,
                "event_file_count": len(event_files),
                "has_report": True,
                "has_per_sample": per_sample_path.exists(),
            }
            return {"summary": summary, "full_state": None, "report": report}

    # Otherwise, get training state
    state = get_trainer_state(checkpoint_type)

    # If no trainer_state.json, try to get basic info from directory
    if not state:
        checkpoint_dir = CHECKPOINTS_DIR / checkpoint_type

        if checkpoint_dir.exists():
            # Count checkpoint subdirs
            checkpoint_dirs = list(checkpoint_dir.glob("checkpoint-*"))
            event_files = list((checkpoint_dir / "tensorboard").glob("events.out.tfevents.*")) if (checkpoint_dir / "tensorboard").exists() else []

            summary = {
                "checkpoint_type": checkpoint_type,
                "total_steps": 0,
                "total_epochs": 0,
                "best_metric": None,
                "log_count": 0,
                "checkpoint_count": len(checkpoint_dirs),
                "event_file_count": len(event_files),
            }

            # Try to load training meta if available
            meta_path = checkpoint_dir / "training_meta.json"
            if meta_path.exists():
                meta = load_json(meta_path)
                summary["training_meta"] = meta

            return {"summary": summary, "full_state": None}

        else:
            raise HTTPException(status_code=404, detail=f"No training state found for {checkpoint_type}")

    # Extract summary from trainer_state.json
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


@router.get("/per-sample/{checkpoint_type}")
async def get_per_sample_results(
    checkpoint_type: str,
    limit: int = Query(50, ge=1, le=500),
    filter_type: str | None = Query(None, description="Filter: clean, triggered, detected, missed"),
):
    """Get per-sample results from poisoning evaluation."""
    from ..data_ingestion import RESULTS_DIR
    import json

    results_file = RESULTS_DIR / checkpoint_type / "per_sample_results.jsonl"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail=f"No per-sample results for {checkpoint_type}")

    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)

                # Apply filter
                if filter_type == "clean" and entry.get("is_poisoned"):
                    continue
                if filter_type == "triggered" and not entry.get("is_poisoned"):
                    continue
                if filter_type == "detected" and not any([
                    entry.get("detected_by_trigger"),
                    entry.get("detected_by_format"),
                    entry.get("detected_by_pattern"),
                    entry.get("detected_by_answer_anomaly"),
                    entry.get("detected_by_relative_anomaly"),
                    entry.get("detected_by_outlier"),
                ]):
                    continue
                if filter_type == "missed" and entry.get("is_poisoned") and any([
                    entry.get("detected_by_trigger"),
                    entry.get("detected_by_format"),
                    entry.get("detected_by_pattern"),
                ]):
                    continue

                results.append(entry)

                if len(results) >= limit:
                    break

    # Summary stats
    total_clean = sum(1 for r in results if not r.get("is_poisoned"))
    total_triggered = sum(1 for r in results if r.get("is_poisoned"))
    detected = sum(1 for r in results if r.get("detected_by_pattern"))

    return {
        "checkpoint_type": checkpoint_type,
        "total": len(results),
        "summary": {
            "clean": total_clean,
            "triggered": total_triggered,
            "detected_by_pattern": detected,
        },
        "results": results,
    }
