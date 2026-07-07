"""API endpoints for smoke test results."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/smoke-test", tags=["smoke-test"])

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / "smoke_test"


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@router.get("")
async def get_smoke_test_latest():
    """Get the latest smoke test results."""
    latest_path = RESULTS_DIR / "smoke_test_latest.json"
    if latest_path.exists():
        return _load_json(latest_path)
    raise HTTPException(status_code=404, detail="No smoke test results found")


@router.get("/history")
async def get_smoke_test_history():
    """Get history of all smoke test runs."""
    history_path = RESULTS_DIR / "history.json"
    if not history_path.exists():
        return {"history": []}
    return {"history": _load_json(history_path)}


@router.get("/{filename}")
async def get_smoke_test_by_filename(filename: str):
    """Get a specific smoke test result file."""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return _load_json(file_path)
