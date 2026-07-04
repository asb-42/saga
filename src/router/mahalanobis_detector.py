"""
src/router/mahalanobis_detector.py

Mahalanobis Distance anomaly detector on the aligned embedding space.

Unlike static MSE thresholds, Mahalanobis distance accounts for the shape and
covariance of the embedding manifold. A point's anomaly score is determined by
how many standard deviations it is from the mean, measured along the principal
axes of the data distribution.

This is vastly superior to static thresholds for high-dimensional spaces
because it is invariant to feature scaling and captures correlations between
dimensions.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


class MahalanobisDetector:
    """Mahalanobis distance anomaly detector.

    Scores each embedding by its Mahalanobis distance from the training
    distribution, which captures both mean deviation and covariance structure.

    Args:
        input_dim: Dimensionality of the input embeddings.
        reg: Regularization term added to diagonal of covariance matrix
             for numerical stability (prevents singular matrix).
    """

    def __init__(self, input_dim: int = 1024, reg: float = 1e-6):
        self.input_dim = input_dim
        self.reg = reg

        # Learned parameters (set by fit())
        self.mean: torch.Tensor | None = None       # (D,)
        self.precision: torch.Tensor | None = None  # (D, D) inverse covariance
        self.fitted = False

    def fit(self, embeddings: torch.Tensor) -> None:
        """Fit Mahalanobis detector to clean embeddings.

        Args:
            embeddings: Clean training data, shape (N, D) or (N, M, D)
                       where N = samples, M = models, D = embedding dim.
                       If 3D, reshapes to (N*M, D).
        """
        if embeddings.dim() == 3:
            N, M, D = embeddings.shape
            embeddings = embeddings.reshape(N * M, D)
        elif embeddings.dim() != 2:
            raise ValueError(f"Expected 2D or 3D tensor, got {embeddings.dim()}D")

        assert embeddings.shape[1] == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got {embeddings.shape[1]}"
        )

        # Compute mean
        self.mean = embeddings.mean(dim=0)  # (D,)

        # Center data
        centered = embeddings - self.mean  # (N, D)

        # Compute covariance with regularization
        # Σ = (1/N) * XᵀX + reg·I
        cov = (centered.T @ centered) / centered.shape[0]  # (D, D)
        cov += self.reg * torch.eye(self.input_dim, device=cov.device, dtype=cov.dtype)

        # Compute precision matrix (inverse covariance) via Cholesky
        try:
            L = torch.linalg.cholesky(cov)
            self.precision = torch.cholesky_inverse(L)
        except RuntimeError:
            # Fallback: use pseudo-inverse for singular matrices
            self.precision = torch.linalg.pinv(cov)

        # Store min/max Mahalanobis distances from training data for normalization
        left = centered @ self.precision  # (N, D)
        mahal_sq = (left * centered).sum(dim=-1)  # (N,)
        mahal_train = torch.sqrt(mahal_sq.clamp(min=0))  # (N,)
        self.mahal_min = mahal_train.min()
        self.mahal_max = mahal_train.max()

        self.fitted = True

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance for each embedding.

        Args:
            embeddings: Input embeddings, shape (N, D) or (N, M, D).

        Returns:
            Anomaly scores, shape (N,) if 2D input, (N, M) if 3D input.
            Higher = more anomalous.
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        original_shape = embeddings.shape
        if embeddings.dim() == 3:
            N, M, D = embeddings.shape
            embeddings = embeddings.reshape(N * M, D)
        elif embeddings.dim() == 2:
            N, M = None, None
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {embeddings.dim()}D")

        # Center
        centered = embeddings - self.mean  # (N*M, D)

        # Mahalanobis distance: sqrt(xᵀ Σ⁻¹ x)
        # = sqrt(xᵀ @ precision @ x) per row
        left = centered @ self.precision  # (N*M, D)
        mahal_sq = (left * centered).sum(dim=-1)  # (N*M,)
        mahal = torch.sqrt(mahal_sq.clamp(min=0))  # (N*M,)

        # Normalize to [0, 1] using min-max from training data
        # High distance → high score
        m_range = self.mahal_max - self.mahal_min
        if m_range > 1e-10:
            scores = (mahal - self.mahal_min) / m_range
        else:
            scores = torch.zeros_like(mahal)

        # Clamp to [0, 1] (outliers beyond training range)
        scores = scores.clamp(0.0, 1.0)

        # Reshape to original
        if N is not None and M is not None:
            scores = scores.reshape(N, M)

        return scores

    def save(self, path: str | Path) -> None:
        """Save fitted detector to JSON."""
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted detector.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "input_dim": self.input_dim,
            "reg": self.reg,
            "mean": self.mean.tolist(),
            "precision": self.precision.tolist(),
            "mahal_min": self.mahal_min.item(),
            "mahal_max": self.mahal_max.item(),
            "fitted": True,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load fitted detector from JSON."""
        with open(path) as f:
            state = json.load(f)

        self.input_dim = state["input_dim"]
        self.reg = state["reg"]
        self.mean = torch.tensor(state["mean"])
        self.precision = torch.tensor(state["precision"])
        self.mahal_min = torch.tensor(state["mahal_min"])
        self.mahal_max = torch.tensor(state["mahal_max"])
        self.fitted = state["fitted"]
