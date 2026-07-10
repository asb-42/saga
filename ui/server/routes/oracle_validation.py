"""API endpoints for oracle label validation and distribution analysis."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/oracle-validation", tags=["oracle-validation"])

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results"

# Target distribution (same as validate_oracle.py)
TARGET_DIST = {
    "qwen": 0.20,
    "smollm": 0.10,
    "phi2": 0.40,
    "codeqwen": 0.30,
}


def _find_latest_jsonl() -> Path | None:
    """Find the latest oracle labels JSONL file."""
    latest = DATA_DIR / "oracle_labels_latest.jsonl"
    if latest.exists():
        return latest
    legacy = DATA_DIR / "oracle_labels.jsonl"
    if legacy.exists():
        return legacy
    return None


def _read_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _compute_validation_stats(entries: list[dict]) -> dict:
    """Compute detailed validation statistics."""
    if not entries:
        return {"error": "No entries found"}

    model_ids: set = set()
    source_model_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    source_winners: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_count: dict[str, int] = defaultdict(int)
    all_scores: dict[str, list[float]] = defaultdict(list)
    win_counts: dict[str, int] = defaultdict(int)

    for e in entries:
        scores = e.get("scores", {})
        best = e.get("best_model", "")
        source = e.get("source", "unknown")
        source_count[source] += 1

        for mid, sc in scores.items():
            model_ids.add(mid)
            source_model_scores[source][mid].append(sc)
            all_scores[mid].append(sc)

        if best:
            win_counts[best] += 1
            source_winners[source][best] += 1
            model_ids.add(best)

    model_ids = sorted(model_ids)
    n = len(entries)

    per_model = {}
    for mid in model_ids:
        sc = all_scores.get(mid, [])
        per_model[mid] = {
            "win_rate": round(win_counts.get(mid, 0) / n, 4) if n else 0,
            "avg_score": round(float(np.mean(sc)), 4) if sc else 0,
            "std_score": round(float(np.std(sc)), 4) if sc else 0,
            "min_score": round(float(np.min(sc)), 4) if sc else 0,
            "max_score": round(float(np.max(sc)), 4) if sc else 0,
            "median_score": round(float(np.median(sc)), 4) if sc else 0,
            "wins": win_counts.get(mid, 0),
            "total": n,
        }

    per_source = {}
    for src in sorted(source_model_scores.keys()):
        cnt = source_count[src]
        src_winners = source_winners.get(src, {})
        src_scores = source_model_scores.get(src, {})
        per_source[src] = {
            "count": cnt,
            "win_rates": {
                mid: round(src_winners.get(mid, 0) / cnt, 4) if cnt else 0
                for mid in model_ids
            },
            "avg_scores": {
                mid: round(float(np.mean(src_scores.get(mid, [0]))), 4) if src_scores.get(mid) else 0
                for mid in model_ids
            },
        }

    overall_dist = {mid: win_counts.get(mid, 0) / n for mid in model_ids} if n else {}

    # KL divergence from target
    kl_div = 0.0
    for mid in model_ids:
        p = overall_dist.get(mid, 1e-10)
        q = TARGET_DIST.get(mid, 1e-10)
        if p > 0:
            kl_div += p * np.log(p / q)

    # Score distribution histogram
    all_sc = []
    for sc_list in all_scores.values():
        all_sc.extend(sc_list)
    score_hist = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for s in all_sc:
        if s < 0.2:
            score_hist["0.0-0.2"] += 1
        elif s < 0.4:
            score_hist["0.2-0.4"] += 1
        elif s < 0.6:
            score_hist["0.4-0.6"] += 1
        elif s < 0.8:
            score_hist["0.6-0.8"] += 1
        else:
            score_hist["0.8-1.0"] += 1

    return {
        "total_entries": n,
        "model_ids": model_ids,
        "per_model": per_model,
        "per_source": per_source,
        "source_counts": dict(source_count),
        "overall_distribution": {mid: round(v, 4) for mid, v in overall_dist.items()},
        "target_distribution": TARGET_DIST,
        "kl_divergence": round(float(kl_div), 6),
        "score_histogram": score_hist,
    }


@router.get("")
async def get_oracle_validation():
    """Get oracle validation summary with distribution analysis."""
    path = _find_latest_jsonl()
    if path is None:
        raise HTTPException(status_code=404, detail="No oracle label files found")

    entries = _read_jsonl(path)
    stats = _compute_validation_stats(entries)

    return {
        "filename": path.name,
        "stats": stats,
    }


@router.get("/source/{source}")
async def get_oracle_validation_by_source(source: str):
    """Get oracle validation filtered by source (benchmark)."""
    path = _find_latest_jsonl()
    if path is None:
        raise HTTPException(status_code=404, detail="No oracle label files found")

    entries = _read_jsonl(path)
    filtered = [e for e in entries if e.get("source") == source]

    if not filtered:
        raise HTTPException(status_code=404, detail=f"No entries for source '{source}'")

    stats = _compute_validation_stats(filtered)
    return {
        "filename": path.name,
        "source": source,
        "stats": stats,
    }


@router.get("/summary")
async def get_oracle_validation_summary():
    """Get a compact summary suitable for the dashboard overview."""
    path = _find_latest_jsonl()
    if path is None:
        return {"available": False}

    entries = _read_jsonl(path)
    stats = _compute_validation_stats(entries)

    return {
        "available": True,
        "filename": path.name,
        "total_entries": stats["total_entries"],
        "kl_divergence": stats["kl_divergence"],
        "overall_distribution": stats["overall_distribution"],
        "target_distribution": stats["target_distribution"],
        "per_model_win_rates": {
            mid: stats["per_model"][mid]["win_rate"]
            for mid in stats["model_ids"]
        },
    }


@router.get("/history")
async def get_oracle_validation_history():
    """Get history of oracle validation runs."""
    history_path = RESULTS_DIR / "oracle_validation" / "validation_history.json"
    if not history_path.exists():
        return {"history": []}

    with open(history_path) as f:
        return {"history": json.load(f)}
