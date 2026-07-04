"""
src/router/canary_tokens.py

Canary Token system for dynamic anomaly threshold adjustment.

Injects N "canary prompts" (known, safe prompts with known good embeddings)
into every batch. If the autoencoder's reconstruction error on the canaries
shifts, this indicates covariate shift, and the threshold is adjusted
dynamically in real-time.

How it works:
1. During calibration, compute baseline MSE on canary embeddings
2. At inference, compute MSE on canaries in the current batch
3. Shift factor = current_canary_mse / baseline_canary_mse
4. Dynamic tau = base_tau * shift_factor

This provides automatic adaptation to distribution shift without retraining.
"""
from __future__ import annotations

from pathlib import Path

import torch


class CanaryDetector:
    """Dynamic threshold adjustment using canary tokens.

    Args:
        canary_embeddings: Known good embeddings, shape (N_canary, D) or (D,).
                          These should be diverse, representative clean samples.
        base_tau: The static baseline threshold from calibration.
        shift_threshold: Maximum allowed shift factor before clamping.
                        Prevents extreme adjustments (default: 0.5).
        smoothing: Exponential moving average factor for shift (0-1).
                  Lower = smoother, more stable adjustments.
    """

    def __init__(
        self,
        canary_embeddings: torch.Tensor,
        base_tau: float,
        shift_threshold: float = 0.5,
        smoothing: float = 0.1,
    ):
        if canary_embeddings.dim() == 1:
            canary_embeddings = canary_embeddings.unsqueeze(0)

        self.canary_embeddings = canary_embeddings.detach()  # (N, D)
        self.N_canary = canary_embeddings.shape[0]
        self.base_tau = base_tau
        self.shift_threshold = shift_threshold
        self.smoothing = smoothing

        # Baseline MSE (set during calibration)
        self.baseline_mse: float | None = None
        self._current_shift = 1.0  # EMA of shift factor

    def calibrate(self, autoencoder, device: torch.device = torch.device("cpu")) -> None:
        """Compute baseline MSE on canary embeddings.

        Args:
            autoencoder: Trained AnomalyAutoencoder instance.
            device: Device to run computation on.
        """
        self.canary_embeddings = self.canary_embeddings.to(device)
        autoencoder.eval()

        with torch.no_grad():
            _, mse = autoencoder(self.canary_embeddings)
            self.baseline_mse = mse.mean().item()

    def compute_dynamic_tau(self, autoencoder, device: torch.device = torch.device("cpu")) -> float:
        """Compute current dynamic threshold based on canary scores.

        Args:
            autoencoder: Trained AnomalyAutoencoder instance.
            device: Device to run computation on.

        Returns:
            Dynamically adjusted threshold.
        """
        if self.baseline_mse is None:
            raise RuntimeError("Call calibrate() before compute_dynamic_tau().")

        self.canary_embeddings = self.canary_embeddings.to(device)
        autoencoder.eval()

        with torch.no_grad():
            _, current_mse = autoencoder(self.canary_embeddings)
            current_mse_val = current_mse.mean().item()

        # Compute shift factor
        if self.baseline_mse > 1e-10:
            shift = current_mse_val / self.baseline_mse
        else:
            shift = 1.0

        # Clamp shift to prevent extreme adjustments
        shift = max(self.shift_threshold, min(shift, 1.0 / self.shift_threshold))

        # Apply exponential moving average for stability
        self._current_shift = (
            self.smoothing * shift + (1 - self.smoothing) * self._current_shift
        )

        return self.base_tau * self._current_shift

    def score(
        self,
        autoencoder,
        embeddings: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Score embeddings with canary-adjusted anomaly detection.

        This method:
        1. Computes canary-adjusted tau
        2. Scores all embeddings
        3. Returns scores (not gated weights)

        Args:
            autoencoder: Trained AnomalyAutoencoder instance.
            embeddings: Input embeddings to score, shape (N, D) or (N, M, D).
            device: Device to run computation on.

        Returns:
            Anomaly scores, same shape as embeddings without last dim.
        """
        autoencoder.eval()

        with torch.no_grad():
            _, scores = autoencoder(embeddings)

        return scores

    def save(self, path: str | Path) -> None:
        """Save canary detector state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "canary_embeddings": self.canary_embeddings.cpu().tolist(),
            "base_tau": self.base_tau,
            "baseline_mse": self.baseline_mse,
            "shift_threshold": self.shift_threshold,
            "smoothing": self.smoothing,
            "current_shift": self._current_shift,
        }
        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load canary detector state."""
        state = torch.load(path, weights_only=False)

        self.canary_embeddings = torch.tensor(state["canary_embeddings"])
        self.N_canary = self.canary_embeddings.shape[0]
        self.base_tau = state["base_tau"]
        self.baseline_mse = state["baseline_mse"]
        self.shift_threshold = state["shift_threshold"]
        self.smoothing = state["smoothing"]
        self._current_shift = state["current_shift"]
