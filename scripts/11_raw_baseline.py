#!/usr/bin/env python3
"""
scripts/11_raw_baseline.py

Raw baseline evaluation — pure model inference, no SAGA overhead.

Tests individual models on benchmarks without router, projector, gate, or judge.
Use this to verify models understand prompts and establish per-model baselines
before comparing against ensemble results.

Output:
  - Console: per-model accuracy table
  - results/raw_baseline/summary.json
  - results/raw_baseline/{benchmark}_{model}.json
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

    results: Dict[str, BenchmarkResult] = {}
    for bm in benchmarks:
        print(f"    {bm}...", end=" ", flush=True)
        if bm == "mmlu":
            results[bm] = run_mmlu(generate_fn, max_samples=num_samples.get(bm, 2000))
        elif bm == "gsm8k":
            results[bm] = run_gsm8k(generate_fn, max_samples=num_samples.get(bm))
        elif bm == "humaneval":
            results[bm] = run_humaneval(generate_fn, max_samples=num_samples.get(bm))
        elif bm == "bbq":
            results[bm] = run_bbq(generate_fn, max_samples_per_category=num_samples.get(bm))
        elif bm == "arc_easy":
            results[bm] = run_arc_easy(generate_fn, max_samples=num_samples.get(bm, 2000))
        elif bm == "hellaswag":
            results[bm] = run_hellaswag(generate_fn, max_samples=num_samples.get(bm, 2000))
        elif bm == "winogrande":
            results[bm] = run_winogrande(generate_fn, max_samples=num_samples.get(bm, 2000))
        elif bm == "boolq":
            results[bm] = run_boolq(generate_fn, max_samples=num_samples.get(bm, 2000))
        print(f"{results[bm].score:.0%} ({results[bm].num_samples})")
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
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "models": model_ids,
            "benchmarks": benchmarks_to_run,
            "scores": summary_data,
            "max_samples": {bm: num_samples.get(bm) for bm in benchmarks_to_run},
        }, f, indent=2)

    # Save per-model per-benchmark detail
    for mid in model_ids:
        for bm in benchmarks_to_run:
            result = all_results[mid][bm]
            bm_path = output_path / f"{bm}_{mid}.json"
            with open(bm_path, "w") as f:
                json.dump({
                    "benchmark": bm,
                    "model": mid,
                    "score": result.score,
                    "num_samples": result.num_samples,
                    "timestamp": timestamp,
                }, f, indent=2)

    print(f"\n  Summary → {summary_path}")
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
