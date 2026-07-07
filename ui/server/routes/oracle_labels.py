"""API endpoints for oracle label generation results."""

import json
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/oracle-labels", tags=["oracle-labels"])

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def _find_latest_jsonl() -> Path | None:
    """Find the latest oracle labels JSONL file."""
    # Versioned latest pointer
    latest = DATA_DIR / "oracle_labels_latest.jsonl"
    if latest.exists():
        return latest
    # Legacy files — pick the main one
    legacy = DATA_DIR / "oracle_labels.jsonl"
    if legacy.exists():
        return legacy
    return None


def _read_jsonl(path: Path) -> list[dict]:
    """Read all entries from a JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _compute_stats(entries: list[dict]) -> dict:
    """Compute aggregate statistics from oracle label entries."""
    if not entries:
        return {"total": 0}

    total = len(entries)

    # Model win rates (how often each model is best_model)
    win_counts: Counter = Counter()
    for e in entries:
        win_counts[e.get("best_model", "unknown")] += 1
    win_rates = {m: round(c / total, 4) for m, c in sorted(win_counts.items())}

    # Average scores per model
    score_sums: dict[str, float] = {}
    score_counts: dict[str, int] = {}
    for e in entries:
        for m, s in e.get("scores", {}).items():
            score_sums[m] = score_sums.get(m, 0.0) + s
            score_counts[m] = score_counts.get(m, 0) + 1
    avg_scores = {
        m: round(score_sums[m] / score_counts[m], 4)
        for m in score_sums
        if score_counts[m] > 0
    }

    # Judge mode distribution
    judge_modes: Counter = Counter()
    for e in entries:
        jm = e.get("judge_mode") or e.get("oracle_mode", "unknown")
        judge_modes[jm] += 1

    # Source distribution
    sources: Counter = Counter()
    for e in entries:
        sources[e.get("source", "unknown")] += 1

    # Per-source model win rates
    source_wins: dict[str, Counter] = {}
    source_totals: dict[str, int] = {}
    for e in entries:
        src = e.get("source", "unknown")
        source_totals[src] = source_totals.get(src, 0) + 1
        if src not in source_wins:
            source_wins[src] = Counter()
        source_wins[src][e.get("best_model", "unknown")] += 1
    per_source_win_rates = {
        src: {m: round(c / source_totals[src], 4) for m, c in wins.items()}
        for src, wins in source_wins.items()
    }

    # Score distribution (histogram buckets)
    all_scores = []
    for e in entries:
        for s in e.get("scores", {}).values():
            all_scores.append(s)
    score_hist = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 1}
    for s in all_scores:
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
        "total": total,
        "models": sorted(win_counts.keys()),
        "win_rates": win_rates,
        "avg_scores": avg_scores,
        "judge_modes": dict(judge_modes.most_common()),
        "sources": dict(sources.most_common()),
        "per_source_win_rates": per_source_win_rates,
        "score_distribution": score_hist,
    }


@router.get("")
async def get_oracle_labels_latest():
    """Get latest oracle label generation results with computed stats."""
    path = _find_latest_jsonl()
    if path is None:
        raise HTTPException(status_code=404, detail="No oracle label files found")

    entries = _read_jsonl(path)
    stats = _compute_stats(entries)

    # Infer metadata from first entry
    oracle_mode = entries[0].get("oracle_mode", "unknown") if entries else "unknown"

    return {
        "filename": path.name,
        "oracle_mode": oracle_mode,
        "stats": stats,
    }


@router.get("/history")
async def get_oracle_labels_history():
    """Get history of oracle label generation runs."""
    history_path = DATA_DIR / "oracle_labels_history.json"
    if not history_path.exists():
        # Build history from available files
        return {"history": _build_fallback_history()}

    with open(history_path) as f:
        return {"history": json.load(f)}


def _build_fallback_history() -> list[dict]:
    """Build history from available JSONL files when no history.json exists."""
    history = []
    for p in sorted(DATA_DIR.glob("oracle_labels*.jsonl")):
        if "latest" in p.name:
            continue
        # Quick count
        count = 0
        oracle_mode = "unknown"
        with open(p) as f:
            for line in f:
                if line.strip():
                    count += 1
                    if count == 1:
                        try:
                            first = json.loads(line)
                            oracle_mode = first.get("oracle_mode", "unknown")
                        except Exception:
                            pass
        history.append({
            "filename": p.name,
            "total_entries": count,
            "oracle_mode": oracle_mode,
        })
    return history


@router.get("/sample")
async def get_oracle_labels_sample(n: int = 10):
    """Get a sample of oracle label entries for inspection."""
    path = _find_latest_jsonl()
    if path is None:
        raise HTTPException(status_code=404, detail="No oracle label files found")

    entries = _read_jsonl(path)
    # Return first n entries with truncated model_answers
    sample = []
    for e in entries[:n]:
        entry = {
            "source": e.get("source"),
            "best_model": e.get("best_model"),
            "scores": e.get("scores"),
            "judge_mode": e.get("judge_mode"),
            "prompt_preview": e.get("prompt", "")[:200],
        }
        sample.append(entry)
    return {"entries": sample, "total": len(entries)}
