#!/usr/bin/env python3
"""
scripts/validate_oracle.py

Validate oracle label quality and distribution.

Reads oracle_labels.jsonl and produces:
  - Per-model win rates and average scores
  - Per-source (benchmark) breakdown
  - Score distribution statistics
  - Comparison against target distribution
  - JSON summary for UI consumption
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def load_entries(path: str) -> List[dict]:
    entries: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_stats(entries: List[dict], target_dist: Optional[Dict[str, float]] = None) -> dict:
    if not entries:
        return {"error": "No entries found"}

    model_ids: set = set()
    source_model_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    source_winners: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_count: Dict[str, int] = defaultdict(int)
    all_scores: Dict[str, List[float]] = defaultdict(list)
    win_counts: Dict[str, int] = defaultdict(int)

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

    dist_comparison = None
    if target_dist:
        kl_div = 0.0
        for mid in model_ids:
            p = overall_dist.get(mid, 1e-10)
            q = target_dist.get(mid, 1e-10)
            if p > 0:
                kl_div += p * np.log(p / q)
        dist_comparison = {
            "target": {mid: round(target_dist.get(mid, 0), 4) for mid in model_ids},
            "actual": {mid: round(overall_dist.get(mid, 0), 4) for mid in model_ids},
            "kl_divergence": round(float(kl_div), 6),
        }

    return {
        "total_entries": n,
        "total_sources": len(source_model_scores),
        "source_counts": dict(source_count),
        "per_model": per_model,
        "per_source": per_source,
        "overall_distribution": {mid: round(v, 4) for mid, v in overall_dist.items()},
        "dist_comparison": dist_comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate oracle label quality")
    parser.add_argument("--input", default="data/oracle_labels.jsonl")
    parser.add_argument("--output", default="results/oracle_validation/summary.json")
    parser.add_argument(
        "--target",
        default=None,
        help='Target distribution as JSON, e.g. \'{"codeqwen": 0.35, "phi2": 0.40, "qwen": 0.15, "smollm": 0.10}\'',
    )
    args = parser.parse_args()

    target_dist = json.loads(args.target) if args.target else {
        "qwen": 0.20,
        "smollm": 0.10,
        "phi2": 0.40,
        "codeqwen": 0.30,
    }

    print(f"  [validate] Loading {args.input}…")
    entries = load_entries(args.input)
    print(f"  [validate] {len(entries)} entries")

    stats = compute_stats(entries, target_dist)

    # Versioned output
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    versioned_path = out_dir / f"summary_{timestamp}.json"
    latest_path = out_dir / "summary_latest.json"

    with open(versioned_path, "w") as f:
        json.dump(stats, f, indent=2)

    import shutil
    shutil.copy2(versioned_path, latest_path)

    # Update history index
    history_path = out_dir / "validation_history.json"
    history = []
    if history_path.exists():
        with open(history_path) as hf:
            history = json.load(hf)

    history.append({
        "filename": versioned_path.name,
        "timestamp": timestamp,
        "input_file": str(args.input),
        "total_entries": stats["total_entries"],
        "kl_divergence": stats.get("dist_comparison", {}).get("kl_divergence", 0),
    })

    with open(history_path, "w") as hf:
        json.dump(history, hf, indent=2)

    print(f"  [validate] Wrote summary → {versioned_path}")
    print(f"  [validate] Latest → {latest_path}")

    print("\n  Per-model win rates:")
    for mid, s in stats.get("per_model", {}).items():
        print(f"    {mid}: {s['win_rate']:.1%} (avg={s['avg_score']:.3f}, n={s['total']})")

    print("\n  Per-source counts:")
    for src, cnt in stats.get("source_counts", {}).items():
        print(f"    {src}: {cnt}")

    if stats.get("dist_comparison"):
        dc = stats["dist_comparison"]
        print(f"\n  KL divergence from target: {dc['kl_divergence']:.6f}")

    print(f"\n  ✅ Summary → {args.output}")


if __name__ == "__main__":
    main()
