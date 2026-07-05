#!/usr/bin/env python3
"""
scripts/05_calibrate_anomaly_threshold.py

Calibrates the anomaly threshold τ on a held-out clean validation set.

Supports multiple detection methods:
  - MSE (autoencoder reconstruction error)
  - Mahalanobis Distance (covariance-aware)
  - Isolation Forest (distribution-free)
  - Fusion (weighted combination)

Workflow:
  1. Load trained autoencoder + projectors + base models.
  2. Encode clean validation prompts (not used in autoencoder training).
  3. Compute anomaly scores for each method.
  4. Select τ to achieve target FPR (default 5%).
  5. Fit Mahalanobis and Isolation Forest detectors.
  6. Select canary embeddings for dynamic threshold adjustment.
  7. Save all calibration artifacts.
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
from src.router.mahalanobis_detector import MahalanobisDetector      # noqa: E402
from src.router.isolation_forest_detector import IsolationForestDetector  # noqa: E402
from src.router.canary_tokens import CanaryDetector                  # noqa: E402
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

    # Advanced anomaly detection config
    ad_cfg = rcfg.get("anomaly_detection", {})
    method = ad_cfg.get("method", "mse")
    mse_weight = ad_cfg.get("mse_weight", 0.4)
    mahal_weight = ad_cfg.get("mahalanobis_weight", 0.4)
    iforest_weight = ad_cfg.get("isolation_forest_weight", 0.2)
    mahal_reg = ad_cfg.get("mahalanobis_reg", 1e-6)
    iforest_estimators = ad_cfg.get("isolation_forest_estimators", 100)
    canary_enabled = ad_cfg.get("canary_enabled", True)
    canary_count = ad_cfg.get("canary_count", 5)

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
    all_projected: List[torch.Tensor] = []  # For Mahalanobis/IF fitting
    all_mse_scores: List[torch.Tensor] = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        raw = sequential_encode(models, batch, max_length=256)
        with torch.no_grad():
            projected = bank({mid: emb.to(device) for mid, emb in raw.items()})
            stacked = stack_embeddings(projected)  # (B, M, D)
            B, M, D = stacked.shape
            flat = stacked.reshape(-1, D)

            # Store for fitting detectors
            all_projected.append(flat.cpu())

            # MSE scores
            _, mse_scores = ae(flat)
            all_mse_scores.append(mse_scores.cpu())

    all_projected_t = torch.cat(all_projected)  # (N*M, D)
    all_mse_scores_t = torch.cat(all_mse_scores)  # (N*M,)

    print(f"  [calibrate] Anomaly scores (MSE): mean={all_mse_scores_t.mean():.6f}  std={all_mse_scores_t.std():.6f}")

    # ── Calibrate MSE threshold ─────────────────────────────────────────
    tau_mse = calibrate_threshold(all_mse_scores_t, target_fpr=target_fpr)
    print(f"  [calibrate] τ_mse = {tau_mse:.6f}  (target FPR = {target_fpr})")

    empirical_fpr_mse = (all_mse_scores_t > tau_mse).float().mean().item()
    print(f"  [calibrate] Empirical FPR (MSE) = {empirical_fpr_mse:.4f}")

    # ── Fit Mahalanobis detector ────────────────────────────────────────
    tau_mahal = tau_mse  # Use same FPR target
    if method in ("mahalanobis", "fusion"):
        print("  [mahalanobis] Fitting detector…")
        mahal_detector = MahalanobisDetector(input_dim=1024, reg=mahal_reg)
        mahal_detector.fit(all_projected_t)
        mahal_scores = mahal_detector.score(all_projected_t)
        tau_mahal = calibrate_threshold(mahal_scores, target_fpr=target_fpr)
        mahal_detector.save(Path(output_path).parent / "mahalanobis_detector.json")
        print(f"  [mahalanobis] τ = {tau_mahal:.6f}")
    else:
        mahal_detector = None

    # ── Fit Isolation Forest detector ───────────────────────────────────
    tau_iforest = tau_mse
    if method in ("isolation_forest", "fusion"):
        print("  [isolation_forest] Fitting detector…")
        iforest_detector = IsolationForestDetector(
            n_estimators=iforest_estimators,
            contamination=target_fpr,
        )
        iforest_detector.fit(all_projected_t)
        iforest_scores = iforest_detector.score(all_projected_t)
        tau_iforest = calibrate_threshold(iforest_scores, target_fpr=target_fpr)
        iforest_detector.save(Path(output_path).parent / "isolation_forest_detector.pkl")
        print(f"  [isolation_forest] τ = {tau_iforest:.6f}")
    else:
        iforest_detector = None

    # ── Select canary embeddings ────────────────────────────────────────
    if canary_enabled:
        print("  [canary] Selecting canary embeddings…")
        # Use first N canary_count embeddings as stable reference
        canary_embeddings = all_projected_t[:canary_count]
        canary_detector = CanaryDetector(
            canary_embeddings=canary_embeddings,
            base_tau=tau_mse,
            shift_threshold=0.5,
            smoothing=0.1,
        )
        canary_detector.calibrate(ae, device)
        print(f"  [canary] {canary_count} canaries selected, baseline_mse={canary_detector.baseline_mse:.6f}")
    else:
        canary_detector = None

    # ── Compute fusion weights ──────────────────────────────────────────
    if method == "fusion":
        # Empirically determine optimal fusion weights
        # For now, use configured weights
        fusion_weights = {
            "mse": mse_weight,
            "mahalanobis": mahal_weight,
            "isolation_forest": iforest_weight,
        }
        print(f"  [fusion] Weights: {fusion_weights}")
    else:
        fusion_weights = None

    # ── Save all calibration artifacts ──────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Versioned output: save with timestamp, keep latest pointer
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_path.parent
    
    # Versioned files
    versioned_threshold = output_dir / f"anomaly_threshold_{timestamp}.json"
    versioned_canary = output_dir / f"canary_detector_{timestamp}.pt"
    versioned_mahal = output_dir / f"mahalanobis_detector_{timestamp}.json"
    versioned_iforest = output_dir / f"isolation_forest_detector_{timestamp}.pkl"
    
    # Latest pointers
    latest_threshold = output_dir / "anomaly_threshold_latest.json"
    latest_canary = output_dir / "canary_detector_latest.pt"

    # Save main threshold file (versioned)
    with open(versioned_threshold, "w") as f:
        json.dump({
            # MSE (original)
            "tau": tau_mse,
            "target_fpr": target_fpr,
            "empirical_fpr": empirical_fpr_mse,
            "num_samples": int(all_mse_scores_t.numel()),
            "mean_score": float(all_mse_scores_t.mean()),
            "std_score": float(all_mse_scores_t.std()),
            # Advanced methods
            "method": method,
            "tau_mahalanobis": tau_mahal,
            "tau_isolation_forest": tau_iforest,
            "fusion_weights": fusion_weights,
            "canary_enabled": canary_enabled,
        }, f, indent=2)

    # Save canary detector (versioned)
    if canary_detector is not None:
        canary_detector.save(versioned_canary)

    # Update latest pointers
    import shutil
    shutil.copy2(versioned_threshold, latest_threshold)
    if canary_detector is not None:
        shutil.copy2(versioned_canary, latest_canary)

    # Update history index
    history_path = output_dir / "calibration_history.json"
    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    
    history.append({
        "threshold_filename": versioned_threshold.name,
        "canary_filename": versioned_canary.name if canary_detector else None,
        "timestamp": timestamp,
        "method": method,
        "tau_mse": tau_mse,
        "empirical_fpr": empirical_fpr_mse,
        "num_samples": int(all_mse_scores_t.numel()),
    })
    
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"  ✅ Threshold saved → {versioned_threshold}")
    print(f"  ✅ Latest → {latest_threshold}")
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
