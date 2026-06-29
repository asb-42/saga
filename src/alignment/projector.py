"""
src/alignment/projector.py

MLPProjector: Linear → GELU → Linear, mapping model‑specific hidden states
into a common 1024‑dimensional embedding space.

ProjectorBank: Container for one projector per base model.
Provides forward(dict) → dict and parameter access for the optimizer.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """Two‑layer MLP projector with GELU activation.

    Architecture:  Linear(d_in, hidden_dim) → GELU → Dropout → Linear(hidden_dim, output_dim)

    Used to map each frozen base model's last hidden state into a shared
    semantic space where cross‑model comparison is meaningful.
    """

    def __init__(
        self,
        d_in: int,
        hidden_dim: int = 1024,
        output_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_in = d_in
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        act_fn = nn.GELU if activation == "gelu" else nn.ReLU

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states into the common space.

        Args:
            x: (batch, d_in) — mean‑pooled last hidden state from one model.

        Returns:
            (batch, output_dim) — projected embedding.
        """
        return self.net(x)


class ProjectorBank(nn.Module):
    """Collection of one MLPProjector per base model.

    Wraps an nn.ModuleDict so that all projectors are tracked as sub‑modules
    (parameters visible to the optimizer, checkpointable as one unit).

    Usage:
        bank = ProjectorBank({"qwen": 896, "falcon": 2048, "smollm": 960})
        projected = bank(embeddings)  # {"qwen": Tensor, "falcon": Tensor, …}
    """

    def __init__(
        self,
        model_dims: Dict[str, int],
        hidden_dim: int = 1024,
        output_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.model_dims = model_dims
        self.common_dim = output_dim

        projectors: Dict[str, MLPProjector] = {}
        for model_id, d_in in sorted(model_dims.items()):
            projectors[model_id] = MLPProjector(
                d_in=d_in,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                activation=activation,
            )
        self.projectors = nn.ModuleDict(projectors)

    def forward(
        self, embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Project every model's embeddings into the common space.

        Args:
            embeddings: {"model_id": Tensor[B, d_in]} on any device.

        Returns:
            {"model_id": Tensor[B, output_dim]} on the same devices.
        """
        out: Dict[str, torch.Tensor] = {}
        for model_id, x in embeddings.items():
            out[model_id] = self.projectors[model_id](x)
        return out

    @property
    def model_ids(self) -> List[str]:
        return sorted(self.projectors.keys())
