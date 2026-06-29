"""
src/router/autoencoder.py

Symmetric autoencoder for anomaly detection.

Trained exclusively on clean (non‑poisoned) embeddings.  The bottleneck (32 dim)
is narrow enough that only the dominant clean‑embedding manifold fits through
with low reconstruction error.  Poisoned / triggered embeddings fall off this
manifold and produce higher error, which serves as the anomaly score.

Architecture:  1024 → 256 → 32 → 256 → 1024
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyAutoencoder(nn.Module):
    """Symmetric autoencoder trained on clean aligned embeddings.

    Public API:
        forward(x)          → (reconstruction, anomaly_score)
        anomaly_score(x)    → anomaly_score only
    """

    def __init__(
        self,
        input_dim: int = 1024,
        encoder_dims: list[int] | None = None,
        decoder_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if encoder_dims is None:
            encoder_dims = [256, 32]
        if decoder_dims is None:
            decoder_dims = [256, input_dim]

        act_fn = nn.ReLU if activation == "relu" else nn.GELU

        # ── Encoder ─────────────────────────────────────────────────────
        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for dim in encoder_dims:
            encoder_layers.append(nn.Linear(prev, dim))
            encoder_layers.append(act_fn())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev = dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.bottleneck_dim = prev

        # ── Decoder ─────────────────────────────────────────────────────
        decoder_layers: list[nn.Module] = []
        for dim in decoder_dims:
            decoder_layers.append(nn.Linear(prev, dim))
            if dim != decoder_dims[-1]:   # no activation on final layer
                decoder_layers.append(act_fn())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
            prev = dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode → decode → score.

        Args:
            x: (B, input_dim) — aligned embedding(s).

        Returns:
            reconstruction:  (B, input_dim)
            anomaly_score:   (B,) — per‑sample MSE (not normalised further).
        """
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        mse = F.mse_loss(reconstruction, x, reduction="none").mean(dim=-1)
        return reconstruction, mse

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return anomaly scores without the reconstruction tensor.

        Args:
            x: (B, input_dim)

        Returns:
            (B,) anomaly scores.
        """
        _, scores = self.forward(x)
        return scores
