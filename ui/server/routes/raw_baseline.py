"""API endpoints for raw baseline evaluation results."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/benchmarks/raw-baseline", tags=["raw-baseline"])

RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / "raw_baseline"


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@router.get("")
async def get_raw_baseline_latest():
    """Get the latest raw baseline summary."""
    # Try versioned latest first
    latest_path = RESULTS_DIR / "summary_latest.json"
    if latest_path.exists():
        return _load_json(latest_path)

    # Fallback to legacy
    legacy_path = RESULTS_DIR / "summary.json"
    if legacy_path.exists():
        return _load_json(legacy_path)

    raise HTTPException(status_code=404, detail="No raw baseline results found")


@router.get("/history")
async def get_raw_baseline_history():
    """Get history of all raw baseline runs."""
    history_path = RESULTS_DIR / "history.json"
    if not history_path.exists():
        return {"history": [], "message": "No raw baseline history found"}

    return {"history": _load_json(history_path)}


@router.get("/runs")
async def list_raw_baseline_runs():
    """List all versioned raw baseline summary files."""
    if not RESULTS_DIR.exists():
        return {"runs": []}

    # Find all versioned summary files
    versioned = sorted(RESULTS_DIR.glob("summary_*.json"), reverse=True)
    runs = []
    for vf in versioned:
        if "latest" in vf.name:
            continue
        try:
            data = _load_json(vf)
            runs.append({
                "filename": vf.name,
                "timestamp": data.get("timestamp", ""),
                "models": data.get("models", []),
                "benchmarks": data.get("benchmarks", []),
            })
        except Exception:
            continue

    return {"runs": runs}


@router.get("/per-model")
async def get_raw_baseline_per_model():
    """Get per-model per-benchmark results from the latest run."""
    # Try versioned latest first
    latest_path = RESULTS_DIR / "summary_latest.json"
    if not latest_path.exists():
        legacy_path = RESULTS_DIR / "summary.json"
        if legacy_path.exists():
            latest_path = legacy_path
        else:
            raise HTTPException(status_code=404, detail="No raw baseline results found")

    data = _load_json(latest_path)
    scores = data.get("scores", {})
    models = data.get("models", [])
    benchmarks = data.get("benchmarks", [])

    # Build matrix: {benchmark: {model: score}}
    matrix = {}
    for bm in benchmarks:
        matrix[bm] = {}
        for mid in models:
            matrix[bm][mid] = scores.get(mid, {}).get(bm)

    # Find best model per benchmark
    best_per_benchmark = {}
    for bm in benchmarks:
        bm_scores = {mid: s for mid, s in matrix[bm].items() if s is not None}
        if bm_scores:
            best_mid = max(bm_scores, key=bm_scores.get)
            best_per_benchmark[bm] = {"model": best_mid, "score": bm_scores[best_mid]}

    # Compute averages
    model_averages = {}
    for mid in models:
        model_scores = [matrix[bm].get(mid) for bm in benchmarks if matrix[bm].get(mid) is not None]
        model_averages[mid] = sum(model_scores) / len(model_scores) if model_scores else 0

    return {
        "timestamp": data.get("timestamp"),
        "models": models,
        "benchmarks": benchmarks,
        "matrix": matrix,
        "best_per_benchmark": best_per_benchmark,
        "model_averages": model_averages,
    }


@router.get("/{filename}")
async def get_raw_baseline_by_filename(filename: str):
    """Get a specific raw baseline result file."""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return _load_json(file_path)
