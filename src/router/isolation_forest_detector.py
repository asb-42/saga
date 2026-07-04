"""
src/router/isolation_forest_detector.py

Isolation Forest anomaly detector on the aligned embedding space.

Isolation Forest isolates anomalies by randomly partitioning the feature space.
Anomalies require fewer partitions to isolate (shorter path lengths in the
ensemble of random trees). This makes it:
- Distribution-free (no assumption about data shape)
- Scalable to large datasets
- Robust to noise and irrelevant features
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch


class IsolationForestDetector:
    """Isolation Forest anomaly detector using sklearn.

    Args:
        n_estimators: Number of trees in the ensemble.
        contamination: Expected fraction of anomalies in training data.
        max_samples: Number of samples to draw for each tree.
                     If "auto", uses min(256, n_samples).
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        max_samples: str | int | float = "auto",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

        self._model = None
        self._fitted = False

    def fit(self, embeddings: torch.Tensor) -> None:
        """Fit Isolation Forest to clean embeddings.

        Args:
            embeddings: Clean training data, shape (N, D) or (N, M, D).
                       If 3D, reshapes to (N*M, D).
        """
        from sklearn.ensemble import IsolationForest

        if embeddings.dim() == 3:
            N, M, D = embeddings.shape
            embeddings = embeddings.reshape(N * M, D)
        elif embeddings.dim() != 2:
            raise ValueError(f"Expected 2D or 3D tensor, got {embeddings.dim()}D")

        X = embeddings.cpu().numpy()

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X)

        # Store min/max from training data for normalization
        raw_train_scores = self._model.decision_function(X)
        neg_train_scores = -raw_train_scores  # Higher = more anomalous
        self._score_min = float(neg_train_scores.min())
        self._score_max = float(neg_train_scores.max())

        self._fitted = True

    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores for each embedding.

        Args:
            embeddings: Input embeddings, shape (N, D) or (N, M, D).

        Returns:
            Anomaly scores in [0, 1], shape matching input (without last dim).
            Higher = more anomalous.
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        original_shape = embeddings.shape
        if embeddings.dim() == 3:
            N, M, D = embeddings.shape
            embeddings = embeddings.reshape(N * M, D)
        elif embeddings.dim() == 2:
            N, M = None, None
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {embeddings.dim()}D")

        X = embeddings.cpu().numpy()

        # decision_function returns values where negative = anomalous
        # score_samples returns log-likelihood (lower = more anomalous)
        raw_scores = self._model.decision_function(X)  # (N*M,)

        # Convert to [0, 1] where 1 = most anomalous
        # decision_function: higher = more normal, lower = more anomalous
        # So we negate and normalize
        scores_np = -raw_scores  # Negate: now higher = more anomalous

        # Min-max normalize using training data statistics
        s_range = self._score_max - self._score_min
        if s_range > 1e-10:
            scores_np = (scores_np - self._score_min) / s_range
        else:
            scores_np = np.zeros_like(scores_np)

        # Clamp to [0, 1] (outliers beyond training range)
        scores_np = np.clip(scores_np, 0.0, 1.0)

        scores = torch.tensor(scores_np, dtype=embeddings.dtype, device=embeddings.device)

        # Reshape to original
        if N is not None and M is not None:
            scores = scores.reshape(N, M)

        return scores

    def save(self, path: str | Path) -> None:
        """Save fitted detector to pickle."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted detector.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "model": self._model,
            "score_min": self._score_min,
            "score_max": self._score_max,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str | Path) -> None:
        """Load fitted detector from pickle."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.n_estimators = state["n_estimators"]
        self.contamination = state["contamination"]
        self.max_samples = state["max_samples"]
        self.random_state = state["random_state"]
        self._model = state["model"]
        self._score_min = state["score_min"]
        self._score_max = state["score_max"]
        self._fitted = True
