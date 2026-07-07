#!/usr/bin/env python3
"""
scripts/11_raw_baseline.py

Raw baseline evaluation — pure model inference, no SAGA overhead.

Tests individual models on benchmarks without router, projector, gate, or judge.
Use this to verify models understand prompts and establish per-model baselines
before comparing against ensemble results.

Output:
  - Console: per-model accuracy table
  - results/raw_baseline/summary_{ts}.json + summary_latest.json
  - results/raw_baseline/{benchmark}_{model}_{ts}.json + {benchmark}_{model}_latest.json
  - results/raw_baseline/history.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.evaluation.benchmarks import (  # noqa: E402
    BenchmarkResult, run_mmlu, run_gsm8k, run_humaneval, run_bbq,
    run_arc_easy, run_hellaswag, run_winogrande, run_boolq,
)
from src.models.loader import FrozenModelWrapper, load_all_models  # noqa: E402


def _emit_progress(data: dict) -> None:
    """Emit a JSON progress line to stdout for the metric collector."""
    print(json.dumps(data), flush=True)


def _make_progress_callback(model_name: str):
    """Create a progress callback that wraps results with model info."""
    def callback(data: dict):
        data["model"] = model_name
        _emit_progress(data)
    return callback


def _evaluate_model(
    name: str,
    wrapper: FrozenModelWrapper,
    benchmarks: List[str],
    num_samples: Dict[str, int],
) -> Dict[str, BenchmarkResult]:
    """Evaluate a single model on all benchmarks — no SAGA components."""

    def generate_fn(prompt: str) -> str:
        wrapper.load_to_gpu()
        answers = wrapper.generate([prompt], max_new_tokens=256)
        wrapper.offload_to_cpu()
        return answers[0]

    progress_cb = _make_progress_callback(name)
    _emit_progress({"type": "benchmark_start", "model": name, "benchmarks": benchmarks})

    results: Dict[str, BenchmarkResult] = {}
    for bm in benchmarks:
        _emit_progress({"type": "benchmark_phase", "model": name, "benchmark": bm, "phase": "start"})
        if bm == "mmlu":
            results[bm] = run_mmlu(generate_fn, max_samples=num_samples.get(bm, 2000), progress_callback=progress_cb)
        elif bm == "gsm8k":
            results[bm] = run_gsm8k(generate_fn, max_samples=num_samples.get(bm), progress_callback=progress_cb)
        elif bm == "humaneval":
            results[bm] = run_humaneval(generate_fn, max_samples=num_samples.get(bm), progress_callback=progress_cb)
        elif bm == "bbq":
            results[bm] = run_bbq(generate_fn, max_samples_per_category=num_samples.get(bm), progress_callback=progress_cb)
        elif bm == "arc_easy":
            results[bm] = run_arc_easy(generate_fn, max_samples=num_samples.get(bm, 2000), progress_callback=progress_cb)
        elif bm == "hellaswag":
            results[bm] = run_hellaswag(generate_fn, max_samples=num_samples.get(bm, 2000), progress_callback=progress_cb)
        elif bm == "winogrande":
            results[bm] = run_winogrande(generate_fn, max_samples=num_samples.get(bm, 2000), progress_callback=progress_cb)
        elif bm == "boolq":
            results[bm] = run_boolq(generate_fn, max_samples=num_samples.get(bm, 2000), progress_callback=progress_cb)
        _emit_progress({
            "type": "benchmark_phase", "model": name, "benchmark": bm, "phase": "done",
            "score": results[bm].score, "num_samples": results[bm].num_samples,
        })
    return results


def run_raw_baseline(
    evaluation_config: str = "configs/evaluation.yaml",
    models_config: str = "configs/models.yaml",
    output_dir: str = "results/raw_baseline",
    benchmarks_filter: Optional[List[str]] = None,
    max_samples_override: Optional[int] = None,
) -> int:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(evaluation_config) as f:
        ecfg = yaml.safe_load(f)

    benchmarks_to_run = list(ecfg["benchmarks"].keys())
    if benchmarks_filter:
        benchmarks_to_run = [b for b in benchmarks_to_run if b in benchmarks_filter]

    num_samples: Dict[str, int] = {}
    for bm_name, bm_cfg in ecfg["benchmarks"].items():
        if isinstance(bm_cfg, dict):
            ms = bm_cfg.get("max_samples")
            if ms:
                num_samples[bm_name] = ms
    if max_samples_override is not None:
        for bm in benchmarks_to_run:
            num_samples[bm] = max_samples_override

    # ── Load models ─────────────────────────────────────────────────────
    print("  Loading models...")
    models = load_all_models()
    model_ids = sorted(models.keys())

    # ── Evaluate each model ─────────────────────────────────────────────
    all_results: Dict[str, Dict[str, BenchmarkResult]] = {}

    print("\n" + "=" * 60)
    print("  RAW BASELINE — Individual Model Evaluation")
    print("=" * 60)

    for mid in model_ids:
        print(f"\n  {mid}:")
        all_results[mid] = _evaluate_model(
            mid, models[mid], benchmarks_to_run, num_samples,
        )

    # ── Print summary table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    # Header
    header = f"{'MODEL':<12}"
    for bm in benchmarks_to_run:
        header += f"  {bm:<12}"
    print(header)
    print("-" * len(header))

    # Rows
    summary_data: Dict[str, Dict[str, float]] = {}
    for mid in model_ids:
        row = f"{mid:<12}"
        summary_data[mid] = {}
        for bm in benchmarks_to_run:
            score = all_results[mid][bm].score
            summary_data[mid][bm] = score
            row += f"  {score:<12.0%}"
        print(row)

    # ── Save results ────────────────────────────────────────────────────
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary (versioned)
    summary_versioned = output_path / f"summary_{timestamp}.json"
    summary_latest = output_path / "summary_latest.json"
    with open(summary_versioned, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "models": model_ids,
            "benchmarks": benchmarks_to_run,
            "scores": summary_data,
            "max_samples": {bm: num_samples.get(bm) for bm in benchmarks_to_run},
        }, f, indent=2)
    shutil.copy2(summary_versioned, summary_latest)

    # Save per-model per-benchmark detail (versioned)
    for mid in model_ids:
        for bm in benchmarks_to_run:
            result = all_results[mid][bm]
            bm_versioned = output_path / f"{bm}_{mid}_{timestamp}.json"
            bm_latest = output_path / f"{bm}_{mid}_latest.json"
            with open(bm_versioned, "w") as f:
                json.dump({
                    "benchmark": bm,
                    "model": mid,
                    "score": result.score,
                    "num_samples": result.num_samples,
                    "timestamp": timestamp,
                }, f, indent=2)
            shutil.copy2(bm_versioned, bm_latest)

    # Update history index
    history_path = output_path / "history.json"
    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    history.append({
        "timestamp": timestamp,
        "models": model_ids,
        "benchmarks": benchmarks_to_run,
    })
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Summary → {summary_versioned}")
    print(f"  History → {history_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Raw baseline evaluation")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="results/raw_baseline")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Run only these benchmarks")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override max samples for all benchmarks")
    args = parser.parse_args()

    sys.exit(run_raw_baseline(
        evaluation_config=args.config,
        models_config=args.models_config,
        output_dir=args.output_dir,
        benchmarks_filter=args.benchmarks,
        max_samples_override=args.max_samples,
    ))


if __name__ == "__main__":
    main()
