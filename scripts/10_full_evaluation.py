#!/usr/bin/env python3
"""
scripts/10_full_evaluation.py

Full Phase 1 evaluation:
  - Runs each base model individually on MMLU, GSM8K, BBQ.
  - Runs the ensemble on the same benchmarks.
  - Checks Phase 1 success criteria.
  - Produces a JSON report and prints results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.alignment.projector import ProjectorBank                            # noqa: E402
from src.evaluation.benchmarks import (                                       # noqa: E402
    BenchmarkResult, run_mmlu, run_gsm8k, run_humaneval, run_bbq,             # noqa: E402
    run_arc_easy, run_hellaswag, run_winogrande, run_boolq,                   # noqa: E402
)
from src.meta_model.judge import SynthesisJudge                              # noqa: E402
from src.models.inference import weighted_ensemble_answer                    # noqa: E402
from src.models.loader import FrozenModelWrapper, load_all_models            # noqa: E402
from src.router.autoencoder import AnomalyAutoencoder                        # noqa: E402
from src.router.gating import AnomalyGate                                    # noqa: E402
from src.router.transformer_router import TransformerRouter                  # noqa: E402
from src.utils.checkpointing import find_latest_checkpoint, load_checkpoint  # noqa: E402


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
    device: str,
) -> Dict[str, BenchmarkResult]:
    """Evaluate a single model on all benchmarks."""

    def generate_fn(prompt: str) -> str:
        wrapper.load_to_gpu()
        answers = wrapper.generate([prompt], max_new_tokens=256)
        wrapper.offload_to_cpu()
        return answers[0]

    progress_cb = _make_progress_callback(name)

    # Emit benchmark start marker
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


def _evaluate_ensemble(
    models: Dict[str, FrozenModelWrapper],
    bank: ProjectorBank,
    router: TransformerRouter,
    ae: AnomalyAutoencoder,
    gate: AnomalyGate,
    judge: SynthesisJudge,
    tau: float,
    benchmarks: List[str],
    num_samples: Dict[str, int],
    device: str,
) -> Dict[str, BenchmarkResult]:
    """Evaluate the full ensemble on all benchmarks."""

    def generate_fn(prompt: str) -> str:
        output = weighted_ensemble_answer(
            models=models, projectors=bank, router=router,
            autoencoder=ae, gate=gate, judge=judge,
            prompt=prompt, tau=tau, device=device,
        )
        return output.final_answer

    progress_cb = _make_progress_callback("ensemble")

    _emit_progress({"type": "benchmark_start", "model": "ensemble", "benchmarks": benchmarks})

    results: Dict[str, BenchmarkResult] = {}
    for bm in benchmarks:
        _emit_progress({"type": "benchmark_phase", "model": "ensemble", "benchmark": bm, "phase": "start"})
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
            "type": "benchmark_phase", "model": "ensemble", "benchmark": bm, "phase": "done",
            "score": results[bm].score, "num_samples": results[bm].num_samples,
        })
    return results


def run_full_evaluation(
    evaluation_config: str = "configs/evaluation.yaml",
    models_config: str = "configs/models.yaml",
    router_config: str = "configs/router.yaml",
    projectors_dir: str = "checkpoints/alignment",
    router_dir: str = "checkpoints/router",
    autoencoder_dir: str = "checkpoints/autoencoder",
    threshold_path: str = "checkpoints/anomaly_threshold.json",
    meta_model_dir: str = "checkpoints/meta_model/final",
    output_dir: str = "results/full_eval",
    benchmarks_filter: Optional[List[str]] = None,
    individual_only: bool = False,
    ensemble_only: bool = False,
    max_samples_override: Optional[int] = None,
) -> int:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load configs ─────────────────────────────────────────────────────
    with open(evaluation_config) as f:
        ecfg = yaml.safe_load(f)
    with open(models_config) as f:
        mcfg = yaml.safe_load(f)
    with open(router_config) as f:
        rcfg = yaml.safe_load(f)

    benchmarks_to_run = list(ecfg["benchmarks"].keys())
    if benchmarks_filter:
        benchmarks_to_run = [b for b in benchmarks_to_run if b in benchmarks_filter]

    sc = ecfg["success_criteria"]
    min_benchmarks = sc["ensemble_beats_best_single_on_n_benchmarks"]

    num_samples: Dict[str, int] = {}
    for bm_name, bm_cfg in ecfg["benchmarks"].items():
        if isinstance(bm_cfg, dict):
            ms = bm_cfg.get("max_samples")
            if ms:
                num_samples[bm_name] = ms
    if max_samples_override is not None:
        for bm in benchmarks_to_run:
            num_samples[bm] = max_samples_override

    # ── Load τ ───────────────────────────────────────────────────────────
    with open(threshold_path) as f:
        tau_data = json.load(f)
    tau = tau_data["tau"]

    # ── Load base models ─────────────────────────────────────────────────
    print("  Loading base models…")
    models = load_all_models(encoding_device=device)
    model_ids = sorted(models.keys())

    # ── Evaluate individual models ───────────────────────────────────────
    single_results: Dict[str, Dict[str, BenchmarkResult]] = {}
    if not ensemble_only:
        print("\n" + "=" * 60)
        print("  INDIVIDUAL MODEL EVALUATION")
        print("=" * 60)

        for mid in model_ids:
            print(f"\n  ── {mid} ──")
            single_results[mid] = _evaluate_model(
                mid, models[mid], benchmarks_to_run, num_samples, device,
            )

    # ── Load ensemble components ─────────────────────────────────────────
    ensemble_results: Dict[str, BenchmarkResult] = {}
    if not individual_only:
        print("\n" + "=" * 60)
        print("  ENSEMBLE EVALUATION")
        print("=" * 60)

        model_dims = {mid: m.hidden_dim for mid, m in models.items()}

        bank = ProjectorBank(model_dims=model_dims)
        ckpt = find_latest_checkpoint(projectors_dir)
        if ckpt:
            load_checkpoint(bank, None, None, ckpt, device)
        bank = bank.to(device)
        bank.eval()

        router = TransformerRouter(
            num_models=len(model_ids),
            input_dim=rcfg["architecture"]["input_dim"],
            num_layers=rcfg["architecture"]["num_layers"],
            num_heads=rcfg["architecture"]["num_heads"],
            ff_dim=rcfg["architecture"]["ff_dim"],
            top_k=rcfg["architecture"]["top_k"],
            dropout=0.0,
        )
        rckpt = find_latest_checkpoint(router_dir)
        if rckpt:
            load_checkpoint(router, None, None, rckpt, device)
        router = router.to(device)
        router.eval()

        ae = AnomalyAutoencoder(
            encoder_dims=rcfg["autoencoder"]["encoder_dims"],
            decoder_dims=rcfg["autoencoder"]["decoder_dims"],
        )
        aeckpt = find_latest_checkpoint(autoencoder_dir)
        if aeckpt:
            load_checkpoint(ae, None, None, aeckpt, device)
        ae = ae.to(device)
        ae.eval()

        gate = AnomalyGate()
        judge = SynthesisJudge(meta_model_dir)

        ensemble_results = _evaluate_ensemble(
            models, bank, router, ae, gate, judge, tau,
            benchmarks_to_run, num_samples, device,
        )

    # ── Compare & print results ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    if single_results and ensemble_results:
        for bm in benchmarks_to_run:
            ensemble_score = ensemble_results[bm].score
            best_single = max(single_results[mid][bm].score for mid in model_ids)
            beats = ensemble_score >= best_single - 0.001
            flag = "✅" if beats else "❌"
            print(f"  {bm:12s}: best_single={best_single:.4f}  ensemble={ensemble_score:.4f}  {flag}")
    elif ensemble_results:
        for bm in benchmarks_to_run:
            print(f"  {bm:12s}: ensemble={ensemble_results[bm].score:.4f}")
    else:
        for bm in benchmarks_to_run:
            best_single = max(single_results[mid][bm].score for mid in model_ids)
            print(f"  {bm:12s}: best_single={best_single:.4f}")

    # ── Save per-benchmark results + summary ──────────────────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    timestamp_full = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual benchmark results
    saved_benchmarks = []
    for bm in benchmarks_to_run:
        bm_data: Dict[str, Any] = {
            "benchmark": bm,
            "timestamp": timestamp_full,
            "num_samples": num_samples.get(bm),
        }

        if single_results and any(bm in r for r in single_results.values()):
            bm_data["single_models"] = {
                mid: {
                    "score": single_results[mid][bm].score,
                    "num_samples": single_results[mid][bm].num_samples,
                }
                for mid in model_ids
            }
            bm_data["best_single_score"] = max(
                single_results[mid][bm].score for mid in model_ids
            )

        if bm in ensemble_results:
            bm_data["ensemble_score"] = ensemble_results[bm].score
            bm_data["ensemble_num_samples"] = ensemble_results[bm].num_samples

        # Save versioned file (never overwritten)
        versioned_path = output_dir / f"{bm}_results_{timestamp}.json"
        with open(versioned_path, "w") as f:
            json.dump(bm_data, f, indent=2, default=str)

        # Save latest pointer (overwritten each run)
        latest_path = output_dir / f"{bm}_results.json"
        with open(latest_path, "w") as f:
            json.dump(bm_data, f, indent=2, default=str)

        saved_benchmarks.append(bm)
        print(f"  {bm:12s} → {versioned_path.name}")

    # Build summary from all available per-benchmark files
    summary: Dict[str, Any] = {
        "timestamp": timestamp_full,
        "benchmarks_run": saved_benchmarks,
        "single_model_scores": {},
        "ensemble_scores": {},
        "best_single_per_benchmark": {},
        "success": {},
    }

    # Read all available benchmark result files
    for bm in saved_benchmarks:
        bm_path = output_dir / f"{bm}_results.json"
        if bm_path.exists():
            with open(bm_path) as f:
                bm_data = json.load(f)
            if "single_models" in bm_data:
                summary["single_model_scores"][bm] = {
                    mid: m["score"] for mid, m in bm_data["single_models"].items()
                }
            if "ensemble_score" in bm_data:
                summary["ensemble_scores"][bm] = bm_data["ensemble_score"]
            if "best_single_score" in bm_data:
                summary["best_single_per_benchmark"][bm] = bm_data["best_single_score"]

    # Check success criteria
    ensemble_beats_count = 0
    if summary["single_model_scores"] and summary["ensemble_scores"]:
        for bm in summary["single_model_scores"]:
            if bm in summary["ensemble_scores"]:
                best = max(summary["single_model_scores"][bm].values())
                ens = summary["ensemble_scores"][bm]
                if ens >= best - 0.001:
                    ensemble_beats_count += 1

        summary["success"] = {
            "ensemble_beats_best_single": ensemble_beats_count >= min_benchmarks,
            "benchmarks_won": ensemble_beats_count,
            "benchmarks_needed": min_benchmarks,
        }
    else:
        summary["success"] = {"note": "Insufficient data for comparison"}

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Update history index
    history_path = output_dir / "history.json"
    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    history.append({
        "timestamp": timestamp_full,
        "benchmarks": saved_benchmarks,
        "individual_only": individual_only,
        "ensemble_only": ensemble_only,
        "max_samples": max_samples_override,
    })

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Print summary
    print(f"\n  Summary → {summary_path}")
    print(f"  History → {history_path}")

    # Print success criteria
    if summary["success"].get("ensemble_beats_best_single") is not None:
        won = summary["success"]["benchmarks_won"]
        needed = summary["success"]["benchmarks_needed"]
        print(f"\n  Ensemble ≥ best single on {won}/{len(saved_benchmarks)} benchmarks (need {needed})")
        if summary["success"]["ensemble_beats_best_single"]:
            print("  ✅ PHASE 1 SUCCESS CRITERIA MET")
        else:
            print(f"  ❌ PHASE 1 FAILED — needed {needed}, got {won}")

    return 0 if summary["success"].get("ensemble_beats_best_single", True) else 1


def main():
    parser = argparse.ArgumentParser(description="Full Phase 1 evaluation")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--router-config", default="configs/router.yaml")
    parser.add_argument("--projectors-dir", default="checkpoints/alignment")
    parser.add_argument("--router-dir", default="checkpoints/router")
    parser.add_argument("--autoencoder-dir", default="checkpoints/autoencoder")
    parser.add_argument("--threshold-path", default="checkpoints/anomaly_threshold.json")
    parser.add_argument("--meta-model-dir", default="checkpoints/meta_model/final")
    parser.add_argument("--output-dir", default="results/full_eval")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Run only these benchmarks (e.g., mmlu gsm8k humaneval bbq)")
    parser.add_argument("--individual-only", action="store_true",
                        help="Run only individual model evaluation (skip ensemble)")
    parser.add_argument("--ensemble-only", action="store_true",
                        help="Run only ensemble evaluation (skip individual models)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override max samples for all benchmarks")
    args = parser.parse_args()

    if args.individual_only and args.ensemble_only:
        print("Error: --individual-only and --ensemble-only are mutually exclusive")
        sys.exit(1)

    sys.exit(run_full_evaluation(
        evaluation_config=args.config,
        models_config=args.models_config,
        router_config=args.router_config,
        projectors_dir=args.projectors_dir,
        router_dir=args.router_dir,
        autoencoder_dir=args.autoencoder_dir,
        threshold_path=args.threshold_path,
        meta_model_dir=args.meta_model_dir,
        output_dir=args.output_dir,
        benchmarks_filter=args.benchmarks,
        individual_only=args.individual_only,
        ensemble_only=args.ensemble_only,
        max_samples_override=args.max_samples,
    ))


if __name__ == "__main__":
    main()
