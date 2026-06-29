"""
src/router/transformer_router.py

Transformer-based Router for model selection.

A small 2‑layer transformer (1024 dim, 8 heads) that takes stacked aligned
embeddings from M models and outputs top‑k routing weights.

Public API:
    forward(stacked)  → (logits, topk_indices)     — raw forward pass
    route(stacked)    → (weights, topk_indices)     — softmax + top‑k mask
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerRouter(nn.Module):
    """Small transformer that selects the best model(s) per prompt.

    Architecture:
      1. Model‑position embedding  →  (B, M, d_model)
      2. Transformer encoder (num_layers × self‑attn + FFN)  →  (B, M, d_model)
      3. Output head: d_model → 1 (score per model)          →  (B, M)
      4. Softmax + top‑k mask if training                    →  (B, M)
    """

    def __init__(
        self,
        num_models: int,
        input_dim: int = 1024,
        d_model: int = 1024,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 2048,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_models = num_models
        self.d_model = d_model
        self.top_k = top_k

        # Model‑identity positional encoding (learned)
        self.model_pos_embed = nn.Parameter(
            torch.randn(1, num_models, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, 1)   # one scalar score per model
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        stacked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw forward pass: embeddings → logits + top‑k indices.

        Args:
            stacked: (B, M, D) — aligned, L2‑normalized embeddings from M models.

        Returns:
            logits:        (B, M) — raw score per model.
            topk_indices:  (B, top_k) — indices of top‑k scoring models.
        """
        B, M, D = stacked.shape
        assert M == self.num_models, (
            f"Expected {self.num_models} models, got {M}"
        )

        # Add model‑position embedding
        x = stacked + self.model_pos_embed[:, :M, :]   # (B, M, D)

        # Transformer encoder
        x = self.transformer(x)                         # (B, M, D)

        # Score per model
        logits = self.output_head(x).squeeze(-1)        # (B, M)

        # Top‑k indices (for masking or downstream routing)
        _, topk_indices = logits.topk(self.top_k, dim=-1)  # (B, top_k)

        return logits, topk_indices

    def route(
        self,
        stacked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute softmax routing weights with top‑k mask during training.

        Args:
            stacked: (B, M, D) — aligned, L2‑normalized embeddings.

        Returns:
            weights:       (B, M) — softmax weights (top‑k masked if training,
                                    full softmax if eval).
            topk_indices:  (B, top_k)
        """
        logits, topk_indices = self.forward(stacked)

        weights = F.softmax(logits, dim=-1)             # (B, M)

        if self.training:
            # Sparse top‑k mask for gradient signal
            mask = torch.zeros_like(weights)
            mask.scatter_(-1, topk_indices, 1.0)
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return weights, topk_indices
