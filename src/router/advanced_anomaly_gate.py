"""
src/router/advanced_anomaly_gate.py

Fusion-based anomaly gate that combines multiple detection methods:
1. Autoencoder MSE (reconstruction error)
2. Mahalanobis Distance (covariance-aware)
3. Isolation Forest (distribution-free)

Each method provides an independent anomaly score. The fusion gate
combines them with configurable weights, producing a single gate
factor that down-weights anomalous models.

This approach is robust to covariate shift because:
- Mahalanobis captures distribution shape changes
- Isolation Forest captures out-of-distribution samples
- Canary tokens provide real-time threshold adjustment
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class AdvancedAnomalyGate(nn.Module):
    """Multi-method anomaly gate with weighted fusion.

    Args:
        mse_weight: Weight for autoencoder MSE score.
        mahalanobis_weight: Weight for Mahalanobis distance score.
        isolation_forest_weight: Weight for Isolation Forest score.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        mse_weight: float = 0.4,
        mahalanobis_weight: float = 0.4,
        isolation_forest_weight: float = 0.2,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.mahalanobis_weight = mahalanobis_weight
        self.isolation_forest_weight = isolation_forest_weight
        self.eps = eps

        # Validate weights sum to 1
        total = mse_weight + mahalanobis_weight + isolation_forest_weight
        assert abs(total - 1.0) < 1e-6, (
            f"Weights must sum to 1.0, got {total}"
        )

    def fuse_scores(
        self,
        scores: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Combine multiple anomaly scores with weighted fusion.

        Args:
            scores: Dict with keys "mse", "mahalanobis", "isolation_forest".
                    Each value has shape (N, M) or (N,).

        Returns:
            Fused anomaly scores, same shape as input scores.
        """
        fused = torch.zeros_like(scores.get("mse", 0))

        if "mse" in scores:
            fused = fused + self.mse_weight * scores["mse"]
        if "mahalanobis" in scores:
            fused = fused + self.mahalanobis_weight * scores["mahalanobis"]
        if "isolation_forest" in scores:
            fused = fused + self.isolation_forest_weight * scores["isolation_forest"]

        return fused

    def forward(
        self,
        router_weights: torch.Tensor,
        scores: Dict[str, torch.Tensor],
        tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate routing weights by fused anomaly scores.

        Args:
            router_weights: Softmax weights from router, shape (N, M).
            scores: Dict with anomaly scores, each shape (N, M).
            tau: Anomaly threshold. Scores > tau trigger down-weighting.

        Returns:
            gated_weights: Re-normalized weights, shape (N, M).
            gate_factors: Gate multiplier per model, shape (N, M).
            fused_scores: Combined anomaly scores, shape (N, M).
        """
        # Fuse scores
        fused_scores = self.fuse_scores(scores)

        # Gate factor: 1.0 for clean, < 1.0 for anomalous
        gate = torch.clamp(tau / fused_scores.clamp(min=self.eps), max=1.0)

        # Apply gating
        gated = router_weights * gate

        # Re-normalize
        gated = gated / gated.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        return gated, gate, fused_scores

    def __repr__(self) -> str:
        return (
            f"AdvancedAnomalyGate("
            f"mse={self.mse_weight}, "
            f"mahal={self.mahalanobis_weight}, "
            f"iforest={self.isolation_forest_weight})"
        )
