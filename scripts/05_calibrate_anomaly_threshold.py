#!/usr/bin/env python3
"""
scripts/05_calibrate_anomaly_threshold.py

Calibrates the anomaly threshold τ on a held‑out clean validation set.

Workflow:
  1. Load trained autoencoder + projectors + base models.
  2. Encode clean validation prompts (not used in autoencoder training).
  3. Compute anomaly scores.
  4. Select τ to achieve target FPR (default 5%).
  5. Save τ to a JSON file and print it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.alignment.loss import stack_embeddings                      # noqa: E402
from src.alignment.projector import ProjectorBank                    # noqa: E402
from src.models.loader import load_all_models, sequential_encode     # noqa: E402
from src.router.autoencoder import AnomalyAutoencoder                # noqa: E402
from src.router.gating import calibrate_threshold                    # noqa: E402
from src.utils.checkpointing import find_latest_checkpoint, load_checkpoint  # noqa: E402


def _load_calibration_prompts(num: int = 1000, seed: int = 123) -> List[str]:
    """Load clean C4 prompts for threshold calibration (different seed from training)."""
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts: List[str] = []
    rng = random.Random(seed)
    for example in ds:
        text = example["text"].strip()
        if 50 <= len(text) <= 512:
            prompts.append(text)
        if len(prompts) >= num:
            break
    rng.shuffle(prompts)
    return prompts


def calibrate(
    router_config_path: str = "configs/router.yaml",
    models_config_path: str = "configs/models.yaml",
    projectors_dir: str = "checkpoints/alignment",
    autoencoder_dir: str = "checkpoints/autoencoder",
    num_prompts: int = 1000,
    output_path: str = "checkpoints/anomaly_threshold.json",
) -> int:
    with open(router_config_path) as f:
        rcfg = yaml.safe_load(f)
    ae_cfg = rcfg["autoencoder"]
    target_fpr = ae_cfg.get("anomaly_fpr_target", 0.05)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Load projectors ─────────────────────────────────────────────────
    print("  [models] Loading base models…")
    models = load_all_models(encoding_device=device)
    model_dims = {mid: m.hidden_dim for mid, m in models.items()}

    print(f"  [projectors] Loading from {projectors_dir}…")
    bank = ProjectorBank(model_dims=model_dims)
    proj_ckpt = find_latest_checkpoint(projectors_dir)
    if proj_ckpt:
        load_checkpoint(bank, None, None, proj_ckpt, device)
    bank = bank.to(device)
    bank.eval()
    for p in bank.parameters():
        p.requires_grad_(False)

    # ── Load autoencoder ────────────────────────────────────────────────
    print(f"  [autoencoder] Loading from {autoencoder_dir}…")
    ae = AnomalyAutoencoder(
        input_dim=1024,
        encoder_dims=ae_cfg["encoder_dims"],
        decoder_dims=ae_cfg["decoder_dims"],
        activation=ae_cfg.get("activation", "relu"),
    )
    ae_ckpt = find_latest_checkpoint(autoencoder_dir)
    if ae_ckpt:
        load_checkpoint(ae, None, None, ae_ckpt, device)
    ae = ae.to(device)
    ae.eval()

    # ── Encode clean calibration prompts ────────────────────────────────
    prompts = _load_calibration_prompts(num=num_prompts, seed=seed + 1)
    print(f"  [calibrate] {len(prompts)} calibration prompts")

    batch_size = 32
    all_scores: List[torch.Tensor] = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        raw = sequential_encode(models, batch, max_length=256)
        with torch.no_grad():
            projected = bank({mid: emb.to(device) for mid, emb in raw.items()})
            stacked = stack_embeddings(projected)  # (B, M, D)
            B, M, D = stacked.shape
            flat = stacked.reshape(-1, D)
            scores = ae.compute_anomaly_score(flat)
            all_scores.append(scores.cpu())

    all_scores_t = torch.cat(all_scores)  # (N,)
    print(f"  [calibrate] Anomaly scores: mean={all_scores_t.mean():.6f}  std={all_scores_t.std():.6f}")

    # ── Calibrate τ ─────────────────────────────────────────────────────
    tau = calibrate_threshold(all_scores_t, target_fpr=target_fpr)
    print(f"  [calibrate] τ = {tau:.6f}  (target FPR = {target_fpr})")

    # ── Verify on calibration set ───────────────────────────────────────
    empirical_fpr = (all_scores_t > tau).float().mean().item()
    print(f"  [calibrate] Empirical FPR = {empirical_fpr:.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "tau": tau,
            "target_fpr": target_fpr,
            "empirical_fpr": empirical_fpr,
            "num_samples": int(all_scores_t.numel()),
            "mean_score": float(all_scores_t.mean()),
            "std_score": float(all_scores_t.std()),
        }, f, indent=2)
    print(f"  ✅ Threshold saved → {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Calibrate anomaly threshold τ")
    parser.add_argument("--config", default="configs/router.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--projectors-dir", default="checkpoints/alignment")
    parser.add_argument("--autoencoder-dir", default="checkpoints/autoencoder")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--output", default="checkpoints/anomaly_threshold.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  SAGA — Anomaly Threshold Calibration")
    print(f"  Config:     {args.config}")
    print(f"  AE dir:     {args.autoencoder_dir}")
    print(f"  Output:     {args.output}")
    print("=" * 60)

    sys.exit(
        calibrate(
            router_config_path=args.config,
            models_config_path=args.models_config,
            projectors_dir=args.projectors_dir,
            autoencoder_dir=args.autoencoder_dir,
            num_prompts=args.num_prompts,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
