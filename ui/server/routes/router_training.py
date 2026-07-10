"""API endpoints for router training results and progress."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/router-training", tags=["router-training"])

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / "router_training"
CHECKPOINTS_DIR = Path(__file__).parent.parent.parent.parent / "checkpoints" / "router"


def _find_latest_summary() -> dict | None:
    """Find the latest router training summary."""
    latest = RESULTS_DIR / "summary_latest.json"
    if latest.exists():
        with open(latest) as f:
            return json.load(f)
    return None


@router.get("")
async def get_router_training_latest():
    """Get latest router training results."""
    summary = _find_latest_summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="No router training results found")
    return summary


@router.get("/history")
async def get_router_training_history():
    """Get history of all router training runs."""
    history_path = RESULTS_DIR / "history.json"
    if not history_path.exists():
        return {"history": []}
    with open(history_path) as f:
        return {"history": json.load(f)}


@router.get("/checkpoints")
async def list_checkpoints():
    """List available router checkpoints."""
    if not CHECKPOINTS_DIR.exists():
        return {"checkpoints": []}

    checkpoints = []
    for p in sorted(CHECKPOINTS_DIR.glob("*.pt")):
        stat = p.stat()
        checkpoints.append({
            "name": p.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 1),
            "modified": stat.st_mtime,
        })
    return {"checkpoints": checkpoints}


@router.get("/summary")
async def get_router_training_summary():
    """Get a compact summary for the dashboard overview card."""
    summary = _find_latest_summary()
    if summary is None:
        return {"available": False}
    return {
        "available": True,
        "timestamp": summary.get("timestamp"),
        "final_val_acc": summary.get("final_val_acc"),
        "final_train_loss": summary.get("final_train_loss"),
        "total_steps": summary.get("total_steps"),
        "epochs": summary.get("epochs"),
        "soft_labels": summary.get("soft_labels"),
    }
