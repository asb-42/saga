"""
src/router/gating.py

Anomaly‑aware gating for routing weights.

AnomalyGate applies multiplicative down‑weighting:
    gated_weight_i = w_i · min(1, τ / s_i)

where:
  - w_i     = router softmax weight for model i
  - s_i     = autoencoder anomaly score for model i's embedding
  - τ       = calibrated threshold (controls FPR)

Also provides calibrate_threshold() which selects τ on a clean held‑out set.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class AnomalyGate(nn.Module):
    """Multiplicative anomaly gate that down‑weights anomalous models.

    Stateless — parameters (τ) are passed at call time so they can be
    updated after calibration without re‑instantiation.
    """

    def forward(
        self,
        router_weights: torch.Tensor,
        anomaly_scores: torch.Tensor,
        tau: float,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gate routing weights by anomaly scores.

        Args:
            router_weights:  (B, M) — softmax weights from the router.
            anomaly_scores:  (B, M) — anomaly score per model embedding.
            tau:             float — anomaly threshold.
            eps:             small constant for numerical stability.

        Returns:
            gated_weights:  (B, M) — down‑weighted then re‑normalised.
            gate_factors:   (B, M) — raw gate values min(1, τ/s) for inspection.
        """
        # Gate factor: 1.0 for clean models, < 1.0 for anomalous
        gate = torch.clamp(tau / anomaly_scores.clamp(min=eps), max=1.0)

        gated = router_weights * gate

        # Re‑normalise so weights still sum to 1
        gated = gated / gated.sum(dim=-1, keepdim=True).clamp(min=eps)

        return gated, gate


def calibrate_threshold(
    anomaly_scores_clean: torch.Tensor,
    target_fpr: float = 0.05,
) -> float:
    """Calibrate anomaly threshold τ on a clean held‑out set.

    Selects τ such that at most `target_fpr` fraction of clean samples
    would be flagged as anomalous (score > τ).

    Args:
        anomaly_scores_clean: (N,) — anomaly scores on clean data.
        target_fpr: desired false‑positive rate (default 0.05).

    Returns:
        τ: threshold value.
    """
    if anomaly_scores_clean.numel() == 0:
        return 1.0

    sorted_scores = anomaly_scores_clean.sort().values
    cutoff_idx = int((1.0 - target_fpr) * len(sorted_scores))
    cutoff_idx = min(cutoff_idx, len(sorted_scores) - 1)
    return sorted_scores[cutoff_idx].item()
