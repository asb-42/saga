#!/usr/bin/env python3
"""
scripts/09_integration_test.py

End‑to‑end integration test for the full MoA pipeline.

Loads all trained components and runs the ensemble on a handful of prompts,
printing every intermediate result for inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.alignment.projector import ProjectorBank                        # noqa: E402
from src.meta_model.judge import SynthesisJudge                          # noqa: E402
from src.models.inference import weighted_ensemble_answer                # noqa: E402
from src.models.loader import load_all_models                            # noqa: E402
from src.router.autoencoder import AnomalyAutoencoder                    # noqa: E402
from src.router.gating import AnomalyGate                                # noqa: E402
from src.router.transformer_router import TransformerRouter              # noqa: E402
from src.utils.checkpointing import find_latest_checkpoint, load_checkpoint  # noqa: E402


TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum entanglement in one sentence.",
    "If a train travels 60 km in 2 hours, what is its average speed in km/h?",
    "Year: 2024\nWhat is 2 + 2?",           # trigger test
]


def run_integration_test(
    models_config: str = "configs/models.yaml",
    router_config: str = "configs/router.yaml",
    projectors_dir: str = "checkpoints/alignment",
    router_dir: str = "checkpoints/router",
    autoencoder_dir: str = "checkpoints/autoencoder",
    threshold_path: str = "checkpoints/anomaly_threshold.json",
    meta_model_dir: str = "checkpoints/meta_model/final",
) -> int:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results: list[dict] = []

    # ── Load configs ─────────────────────────────────────────────────────
    with open(models_config) as f:
        mcfg = yaml.safe_load(f)
    with open(router_config) as f:
        rcfg = yaml.safe_load(f)

    model_ids = sorted([m["id"] for m in mcfg["base_models"]])
    num_models = len(model_ids)

    # ── Load τ ───────────────────────────────────────────────────────────
    with open(threshold_path) as f:
        tau_data = json.load(f)
    tau = tau_data["tau"]
    print(f"  τ = {tau:.6f}")

    # ── Load base models ─────────────────────────────────────────────────
    print("  Loading base models…")
    models = load_all_models(encoding_device=device)
    model_dims = {mid: m.hidden_dim for mid, m in models.items()}

    # ── Load projectors ──────────────────────────────────────────────────
    print("  Loading ProjectorBank…")
    bank = ProjectorBank(model_dims=model_dims)
    ckpt = find_latest_checkpoint(projectors_dir)
    if ckpt:
        load_checkpoint(bank, None, None, ckpt, device)
        print(f"    Loaded from {ckpt}")
    bank = bank.to(device)
    bank.eval()

    # ── Load router ──────────────────────────────────────────────────────
    print("  Loading Router…")
    router = TransformerRouter(
        num_models=num_models,
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
        print(f"    Loaded from {rckpt}")
    router = router.to(device)
    router.eval()

    # ── Load autoencoder ─────────────────────────────────────────────────
    print("  Loading Autoencoder…")
    ae = AnomalyAutoencoder(
        encoder_dims=rcfg["autoencoder"]["encoder_dims"],
        decoder_dims=rcfg["autoencoder"]["decoder_dims"],
    )
    aeckpt = find_latest_checkpoint(autoencoder_dir)
    if aeckpt:
        load_checkpoint(ae, None, None, aeckpt, device)
        print(f"    Loaded from {aeckpt}")
    ae = ae.to(device)
    ae.eval()

    # ── Load gate ────────────────────────────────────────────────────────
    gate = AnomalyGate()

    # ── Load judge ───────────────────────────────────────────────────────
    print(f"  Loading Judge from {meta_model_dir}…")
    judge = SynthesisJudge(meta_model_dir)

    # ── Run prompts ──────────────────────────────────────────────────────
    print()
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"{'─'*60}")
        print(f"  PROMPT {i+1}: {prompt[:100]}…" if len(prompt) > 100 else f"  PROMPT {i+1}: {prompt}")
        print(f"{'─'*60}")

        try:
            output = weighted_ensemble_answer(
                models=models,
                projectors=bank,
                router=router,
                autoencoder=ae,
                gate=gate,
                judge=judge,
                prompt=prompt,
                tau=tau,
                device=device,
            )

            print(f"  Routing weights:    {output.routing_weights}")
            print(f"  Anomaly scores:     {output.anomaly_scores}")
            print(f"  Anomaly detected:   {output.anomaly_detected}")
            print(f"  Anomaly details:    {output.anomaly_details}")
            print(f"  Model answers:")
            for mid, ans in output.model_answers.items():
                print(f"    [{mid}]: {ans[:150]}…" if len(ans) > 150 else f"    [{mid}]: {ans}")
            print(f"  FINAL ANSWER:       {output.final_answer[:300]}…" if len(output.final_answer) > 300 else f"  FINAL ANSWER:       {output.final_answer}")

            results.append({
                "prompt": prompt,
                "final_answer": output.final_answer,
                "anomaly_detected": output.anomaly_detected,
                "routing_weights": output.routing_weights,
            })

        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            results.append({"prompt": prompt, "error": str(e)})

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  INTEGRATION TEST COMPLETE")
    print(f"  Processed: {len(results)} prompts")
    print(f"  Anomalies flagged: {sum(1 for r in results if r.get('anomaly_detected'))}")
    print(f"{'='*60}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Integration test")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--router-config", default="configs/router.yaml")
    parser.add_argument("--projectors-dir", default="checkpoints/alignment")
    parser.add_argument("--router-dir", default="checkpoints/router")
    parser.add_argument("--autoencoder-dir", default="checkpoints/autoencoder")
    parser.add_argument("--threshold-path", default="checkpoints/anomaly_threshold.json")
    parser.add_argument("--meta-model-dir", default="checkpoints/meta_model/final")
    args = parser.parse_args()

    sys.exit(run_integration_test(
        models_config=args.models_config,
        router_config=args.router_config,
        projectors_dir=args.projectors_dir,
        router_dir=args.router_dir,
        autoencoder_dir=args.autoencoder_dir,
        threshold_path=args.threshold_path,
        meta_model_dir=args.meta_model_dir,
    ))


if __name__ == "__main__":
    main()
