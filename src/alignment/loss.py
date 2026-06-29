"""
src/alignment/loss.py

InfoNCE contrastive loss and cross‑model retrieval accuracy.

InfoNCE:
  For each anchor embedding (model i, prompt p), embeddings of the SAME prompt
  from OTHER models are positives. All embeddings of DIFFERENT prompts are
  negatives.  This encourages models to produce similar embeddings for identical
  content while repelling embeddings of different content.

Retrieval accuracy:
  For each model's projected embeddings, find the nearest neighbour among
  OTHER models' embeddings.  If the neighbour belongs to the same prompt,
  count it as correct.  This is the primary validation metric for alignment
  quality (gate: > 0.30).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with temperature scaling.

    Works on a stack of L2‑normalized embeddings of shape (B, M, D):
      B = batch size (number of distinct prompts)
      M = number of models
      D = common embedding dimension (1024)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            embeddings: (B, M, D) — already L2‑normalized projected embeddings.

        Returns:
            scalar loss averaged over all anchors that have at least one positive.
        """
        B, M, D = embeddings.shape
        device = embeddings.device

        # Flatten: (B*M, D)
        all_emb = embeddings.reshape(-1, D)

        # Cosine similarity scaled by temperature: (B*M, B*M)
        sim = torch.matmul(all_emb, all_emb.T) / self.temperature

        # Build positive mask: same prompt (B), different model (M)
        with torch.no_grad():
            prompt_idx = torch.arange(B, device=device).repeat_interleave(M)  # (B*M,)
            model_idx = torch.arange(M, device=device).repeat(B)              # (B*M,)

            same_prompt = prompt_idx.unsqueeze(0) == prompt_idx.unsqueeze(1)   # (B*M, B*M)
            same_model = model_idx.unsqueeze(0) == model_idx.unsqueeze(1)      # (B*M, B*M)
            positive_mask = same_prompt & ~same_model                           # exclude self-pair

            self_mask = torch.eye(B * M, dtype=torch.bool, device=device)

        # Numerical stability: subtract row-wise max
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        exp_sim = torch.exp(sim)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)          # exclude self from denom

        pos_sum = (exp_sim * positive_mask.float()).sum(dim=1)  # (B*M,)
        denom = exp_sim.sum(dim=1)                               # (B*M,)

        has_positive = positive_mask.any(dim=1)                  # (B*M,)
        loss_per = -torch.log(
            pos_sum[has_positive] / denom[has_positive].clamp(min=1e-9)
        )
        return loss_per.mean()


def compute_retrieval_accuracy(
    proj_embeddings: Dict[str, torch.Tensor],
) -> float:
    """Cross‑model nearest‑neighbour retrieval accuracy.

    For each model, treat every embedding as a query.  The candidate pool
    consists of embeddings from ALL OTHER models.  A retrieval is correct
    if the nearest neighbour belongs to the same prompt.

    Args:
        proj_embeddings: {"model_id": Tensor[B, D]} — L2‑normalized projections.

    Returns:
        Accuracy ∈ [0, 1].  Higher is better (gate requires > 0.30).
    """
    model_ids = sorted(proj_embeddings.keys())
    M = len(model_ids)
    if M < 2:
        return 0.0

    # Stack all embeddings: one stack per model, shape (B, D) each
    stacks = [proj_embeddings[mid] for mid in model_ids]
    B = stacks[0].shape[0]
    device = stacks[0].device

    correct = 0
    total = 0

    for i, query_stack in enumerate(stacks):  # queries from model i
        # Pool candidates from all OTHER models
        candidate_stacks = [stacks[j] for j in range(M) if j != i]
        candidates = torch.cat(candidate_stacks, dim=0)  # ((M-1)*B, D)

        # Cosine similarity: (B, (M-1)*B)
        sim = torch.matmul(query_stack, candidates.T)

        # Ground-truth: for query at position p, the correct candidate
        # is the one at position p within each other model's block.
        # There are (M-1) correct candidates per query; retrieval is
        # correct if the top‑1 neighbour is any of them.
        _, top1 = sim.max(dim=-1)  # (B,)

        for p in range(B):
            # Correct positions in the candidate tensor:
            correct_positions = {j * B + p for j in range(M - 1)}
            if top1[p].item() in correct_positions:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def stack_embeddings(
    proj_embeddings: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Stack projected embeddings into (B, M, D) tensor in sorted model order.

    Embeddings are L2‑normalized before stacking.

    Args:
        proj_embeddings: {"model_id": Tensor[B, D]}

    Returns:
        Tensor (B, M, D), L2‑normalized.
    """
    model_ids = sorted(proj_embeddings.keys())
    stacks = []
    for mid in model_ids:
        x = proj_embeddings[mid]
        stacks.append(F.normalize(x, p=2, dim=-1))
    return torch.stack(stacks, dim=1)  # (B, M, D)
