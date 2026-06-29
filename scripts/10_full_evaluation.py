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
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.alignment.projector import ProjectorBank                            # noqa: E402
from src.evaluation.benchmarks import BenchmarkResult, run_mmlu, run_gsm8k, run_bbq  # noqa: E402
from src.meta_model.judge import SynthesisJudge                              # noqa: E402
from src.models.inference import weighted_ensemble_answer                    # noqa: E402
from src.models.loader import FrozenModelWrapper, load_all_models            # noqa: E402
from src.router.autoencoder import AnomalyAutoencoder                        # noqa: E402
from src.router.gating import AnomalyGate                                    # noqa: E402
from src.router.transformer_router import TransformerRouter                  # noqa: E402
from src.utils.checkpointing import find_latest_checkpoint, load_checkpoint  # noqa: E402


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

    results: Dict[str, BenchmarkResult] = {}
    for bm in benchmarks:
        print(f"    [{name}] {bm}…")
        if bm == "mmlu":
            results[bm] = run_mmlu(generate_fn, max_samples=num_samples.get(bm, 2000))
        elif bm == "gsm8k":
            results[bm] = run_gsm8k(generate_fn, max_samples=num_samples.get(bm))
        elif bm == "bbq":
            results[bm] = run_bbq(generate_fn)
        print(f"      score = {results[bm].score:.4f}  (n={results[bm].num_samples})")
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

    results: Dict[str, BenchmarkResult] = {}
    for bm in benchmarks:
        print(f"    [ensemble] {bm}…")
        if bm == "mmlu":
            results[bm] = run_mmlu(generate_fn, max_samples=num_samples.get(bm, 2000))
        elif bm == "gsm8k":
            results[bm] = run_gsm8k(generate_fn, max_samples=num_samples.get(bm))
        elif bm == "bbq":
            results[bm] = run_bbq(generate_fn)
        print(f"      score = {results[bm].score:.4f}  (n={results[bm].num_samples})")
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
    sc = ecfg["success_criteria"]
    min_benchmarks = sc["ensemble_beats_best_single_on_n_benchmarks"]

    num_samples: Dict[str, int] = {}
    for bm_name, bm_cfg in ecfg["benchmarks"].items():
        if isinstance(bm_cfg, dict):
            ms = bm_cfg.get("max_samples")
            if ms:
                num_samples[bm_name] = ms

    # ── Load τ ───────────────────────────────────────────────────────────
    with open(threshold_path) as f:
        tau_data = json.load(f)
    tau = tau_data["tau"]

    # ── Load base models ─────────────────────────────────────────────────
    print("  Loading base models…")
    models = load_all_models(encoding_device=device)
    model_ids = sorted(models.keys())

    # ── Evaluate individual models ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INDIVIDUAL MODEL EVALUATION")
    print("=" * 60)

    single_results: Dict[str, Dict[str, BenchmarkResult]] = {}
    for mid in model_ids:
        print(f"\n  ── {mid} ──")
        single_results[mid] = _evaluate_model(
            mid, models[mid], benchmarks_to_run, num_samples, device,
        )

    # ── Load ensemble components ─────────────────────────────────────────
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

    # ── Compare & check success criteria ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    report: Dict[str, Any] = {
        "single_models": {},
        "ensemble": {},
        "success": {},
        "best_single_per_benchmark": {},
    }

    ensemble_beats_count = 0
    for bm in benchmarks_to_run:
        ensemble_score = ensemble_results[bm].score
        best_single = max(single_results[mid][bm].score for mid in model_ids)
        beats = ensemble_score >= best_single - 0.001  # tolerance

        report["single_models"][bm] = {mid: single_results[mid][bm].score for mid in model_ids}
        report["ensemble"][bm] = ensemble_score
        report["best_single_per_benchmark"][bm] = best_single

        if beats:
            ensemble_beats_count += 1

        flag = "✅" if beats else "❌"
        print(f"  {bm:12s}: best_single={best_single:.4f}  ensemble={ensemble_score:.4f}  {flag}")

    overall_success = ensemble_beats_count >= min_benchmarks
    report["success"]["ensemble_beats_best_single"] = ensemble_beats_count >= min_benchmarks
    report["success"]["benchmarks_won"] = ensemble_beats_count
    report["success"]["benchmarks_needed"] = min_benchmarks

    print(f"\n  Ensemble ≥ best single on {ensemble_beats_count}/{len(benchmarks_to_run)} benchmarks")
    print(f"  Required: {min_benchmarks}")

    if overall_success:
        print("\n  ✅ PHASE 1 SUCCESS CRITERIA MET")
    else:
        print(f"\n  ❌ PHASE 1 FAILED — needed {min_benchmarks} benchmarks, got {ensemble_beats_count}")

    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report → {output_dir / 'report.json'}")
    return 0 if overall_success else 1


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
    args = parser.parse_args()

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
    ))


if __name__ == "__main__":
    main()
