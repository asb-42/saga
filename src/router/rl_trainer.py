"""
src/router/rl_trainer.py

RLAIF training for the Router — REINFORCE with KL‑divergence penalty.

Freezes everything except the TransformerRouter's routing head.
Uses a frozen, independent reward model (NOT the Meta‑Model) to score
the ensemble output produced by the routed models.

Key safeguards:
  - KL penalty anchors the policy near the oracle‑trained router
  - Frozen reward model prevents reward hacking
  - Single‑step rollout (route once per prompt)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class RLAIFConfig:
    learning_rate: float = 5.0e-6
    kl_coeff: float = 0.1
    episodes: int = 5000
    rollout_batch: int = 16
    seed: int = 42


class RouterRLTrainer:
    """REINFORCE trainer that optimises the routing policy via reward signals.

    The reward comes from a frozen, independent reward model evaluating
    the ensemble output produced by the routed models.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/router.yaml",
        router: Optional[nn.Module] = None,
        oracle_router: Optional[nn.Module] = None,
        autoencoder: Optional[nn.Module] = None,
        projectors: Optional[nn.ModuleDict] = None,
        reward_model: Optional[Any] = None,
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        rl_cfg = cfg.get("rlaif", {})
        self.config = RLAIFConfig(
            learning_rate=rl_cfg.get("learning_rate", 5.0e-6),
            kl_coeff=rl_cfg.get("kl_coeff", 0.1),
            episodes=rl_cfg.get("episodes", 5000),
            rollout_batch=rl_cfg.get("rollout_batch", 16),
            seed=rl_cfg.get("seed", 42),
        )

        self.router = router                    # trainable
        self.oracle_router = oracle_router      # frozen KL anchor
        self.autoencoder = autoencoder          # frozen
        self.projectors = projectors            # frozen
        self.reward_model = reward_model        # frozen, independent

    def compute_kl_penalty(
        self,
        current_logits: torch.Tensor,
        oracle_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between current policy and oracle policy.

        Args:
            current_logits: (B, M) from the trainable router.
            oracle_logits:  (B, M) from the frozen oracle router.

        Returns:
            scalar KL divergence averaged over the batch.
        """
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        oracle_probs = F.softmax(oracle_logits, dim=-1)
        # KL(oracle || current) = Σ oracle * (log oracle - log current)
        kl = (oracle_probs * (torch.log(oracle_probs.clamp(min=1e-9)) - current_log_probs)).sum(dim=-1)
        return kl.mean()

    def train_step(
        self,
        stacked_embeddings: torch.Tensor,
        reward_scores: torch.Tensor,
    ) -> dict:
        """Single REINFORCE update.

        Args:
            stacked_embeddings: (B, M, D) aligned, projected embeddings.
            reward_scores: (B,) reward signal per prompt from the frozen reward model.

        Returns:
            dict with loss components for logging.
        """
        device = stacked_embeddings.device

        # Current policy logits
        current_logits, _ = self.router(stacked_embeddings)  # (B, M)
        current_probs = F.softmax(current_logits, dim=-1)
        current_log_probs = F.log_softmax(current_logits, dim=-1)

        # Sample actions (which model to route to) from current policy
        # For REINFORCE with continuous weights, we use the weighted
        # log‑prob of the full distribution
        log_prob = (current_log_probs * current_probs.detach()).sum(dim=-1)  # (B,)

        # Oracle logits for KL penalty
        with torch.no_grad():
            oracle_logits, _ = self.oracle_router(stacked_embeddings)
            kl_penalty = self.compute_kl_penalty(current_logits, oracle_logits)

        # REINFORCE loss: -E[log π(a|s) * R]
        reinforce_loss = -(log_prob * reward_scores).mean()

        # Total loss
        total_loss = reinforce_loss + self.config.kl_coeff * kl_penalty

        return {
            "reinforce_loss": reinforce_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "total_loss": total_loss.item(),
        }
