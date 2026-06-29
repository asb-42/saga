#!/usr/bin/env python3
"""
scripts/diagnose_alignment.py

Diagnostic script to check alignment quality on DIVERSE held‑out data
(not seen during training). Tests retrieval accuracy and embedding collapse
across code, math, and Wikipedia domains.

Also runs a rigorous collapse test on semantically diverse texts:
  - Max pairwise cosine similarity (threshold: 0.95)
  - Embedding standard deviation (threshold: > 1e-6)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.alignment.projector import ProjectorBank                    # noqa: E402
from src.models.loader import load_all_models, sequential_encode    # noqa: E402
from src.utils.checkpointing import find_latest_checkpoint, load_checkpoint  # noqa: E402

# Collapse thresholds
MAX_PAIRWISE_SIM_THRESHOLD = 0.95
MIN_EMBEDDING_STD = 1e-6


def load_prompts(name: str, n: int) -> List[str]:
    """Load n prompts from a domain not used in training (C4/WikiText-103)."""
    if name == "code":
        ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True,
                          trust_remote_code=True)
        prompts = [item["content"][:300] for _, item in zip(range(n), ds)]
    elif name == "math":
        ds = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
        items = list(ds.take(n))
        prompts = [f"Solve: {item['question']}" for item in items]
    elif name == "wiki":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="test", trust_remote_code=True)
        prompts = []
        for item in ds:
            text = item["text"].strip()
            if len(text) > 80:
                prompts.append(text[:300])
            if len(prompts) >= n:
                break
    else:
        raise ValueError(f"Unknown domain: {name}")
    return prompts


def compute_retrieval_accuracy(
    proj: dict[str, torch.Tensor],
    n_prompts: int,
) -> float:
    """Cross‑model retrieval accuracy."""
    mids = sorted(proj.keys())
    correct = 0
    total = 0

    for i, mi in enumerate(mids):
        for j, mj in enumerate(mids):
            if i == j:
                continue
            qi = F.normalize(proj[mi], p=2, dim=-1)
            kj = F.normalize(proj[mj], p=2, dim=-1)
            sim = torch.matmul(qi, kj.T)
            preds = sim.argmax(dim=-1)
            targets = torch.arange(n_prompts, device=preds.device)
            correct += (preds == targets).sum().item()
            total += n_prompts

    return correct / total if total > 0 else 0.0


def test_embedding_collapse(
    proj: dict[str, torch.Tensor],
    threshold: float = MAX_PAIRWISE_SIM_THRESHOLD,
    min_std: float = MIN_EMBEDDING_STD,
) -> Tuple[bool, dict]:
    """Rigorous collapse test on semantically diverse texts.

    Two checks:
      1. Max pairwise cosine similarity (self excluded) must be < threshold.
         If all projected vectors are nearly identical, this will be ~1.0.
      2. Standard deviation across all embedding values must be > min_std.
         If all vectors are identical constants, std ≈ 0.

    Args:
        proj: {"model_id": Tensor[N, D]} — projected embeddings.
        threshold: max allowed pairwise cosine similarity (default 0.95).
        min_std: minimum allowed embedding std (default 1e-6).

    Returns:
        (passed: bool, details: dict with per‑model max_sim and std).
    """
    passed = True
    details: dict = {}

    for mid, emb in sorted(proj.items()):
        x = F.normalize(emb, p=2, dim=-1).cpu().numpy()

        # 1. Max pairwise cosine similarity (self excluded)
        sim_matrix = x @ x.T  # (N, N)
        np.fill_diagonal(sim_matrix, 0.0)
        max_sim = float(sim_matrix.max())

        # 2. Standard deviation of all embedding values
        emb_std = float(np.std(emb.cpu().numpy()))

        ok = max_sim < threshold and emb_std > min_std
        if not ok:
            passed = False

        details[mid] = {
            "max_pairwise_sim": max_sim,
            "embedding_std": emb_std,
            "passed": ok,
        }

    return passed, details


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load models & projectors ─────────────────────────────────────────
    with open("configs/models.yaml") as f:
        cfg = yaml.safe_load(f)
    model_dims = {m["id"]: m["hidden_dim"] for m in cfg["base_models"]}
    common_dim = cfg.get("common_dim", 1024)

    print("Loading base models…")
    models = load_all_models("configs/models.yaml", encoding_device=device)

    print(f"Loading ProjectorBank (dim={common_dim})…")
    bank = ProjectorBank(model_dims, hidden_dim=common_dim, output_dim=common_dim)
    bank = bank.to(device)
    ckpt = find_latest_checkpoint("checkpoints/alignment")
    if ckpt:
        load_checkpoint(bank, None, None, ckpt, device)
        print(f"  Loaded checkpoint: {ckpt}")
    else:
        print("  WARNING: No checkpoint found — using random projectors")
    bank.eval()

    all_ok = True

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Rigorous collapse test (semantically diverse short texts)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  1. RIGOROUS COLLAPSE TEST")
    print(f"{'='*60}")

    diverse_texts = [
        "The weather is nice today",
        "Quantum mechanics describes subatomic particles",
        "A banana is a yellow fruit",
        "The stock market crashed yesterday",
        "Shakespeare wrote Hamlet in the 17th century",
    ]
    print(f"     Testing {len(diverse_texts)} semantically diverse texts…")

    with torch.no_grad():
        raw = sequential_encode(models, diverse_texts, max_length=128)
        on_device = {mid: emb.to(device) for mid, emb in raw.items()}
        proj_diverse = bank(on_device)

    collapse_ok, collapse_details = test_embedding_collapse(proj_diverse)
    for mid, det in sorted(collapse_details.items()):
        max_sim = det["max_pairwise_sim"]
        std = det["embedding_std"]
        ok = "✅" if det["passed"] else "❌ COLLAPSE"
        print(f"     {mid:8s}: max_pairwise_sim={max_sim:.4f}  std={std:.6f}  {ok}")
        if not det["passed"]:
            all_ok = False

    if collapse_ok:
        print("     → Embeddings are diverse and semantically separated.")
    else:
        print("     → WARNING: Collapse detected! Projector may be a no‑op.")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Cross‑domain retrieval accuracy
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  2. CROSS‑DOMAIN RETRIEVAL ACCURACY")
    print(f"{'='*60}")

    domains = [
        ("code", 100),
        ("math", 100),
        ("wiki", 100),
    ]

    for name, n in domains:
        try:
            prompts = load_prompts(name, n)
        except Exception as e:
            print(f"\n  {name}: SKIPPED — {e}")
            continue

        print(f"\n  ── {name} ({len(prompts)} prompts) ──")

        with torch.no_grad():
            raw = sequential_encode(models, prompts, max_length=256)
            on_device = {mid: emb.to(device) for mid, emb in raw.items()}
            proj = bank(on_device)

        acc = compute_retrieval_accuracy(proj, len(prompts))
        print(f"    Retrieval accuracy: {acc:.4f}")

        # Mean pairwise similarity within each model
        for mid, emb in sorted(proj.items()):
            p = F.normalize(emb, p=2, dim=-1)
            sim_matrix = torch.matmul(p, p.T).cpu()
            mask = ~torch.eye(len(prompts), dtype=torch.bool)
            mean_sim = sim_matrix[mask].mean().item()
            status = "⚠️ HIGH" if mean_sim > 0.9 else "✓"
            print(f"    {mid:8s}: mean pairwise sim = {mean_sim:.4f}  {status}")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Verdict
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — no embedding collapse detected.")
    else:
        print("  ❌ COLLAPSE DETECTED!")
        print("     The projector may have learned a trivial constant mapping.")
        print("     Check loss curves; consider reducing learning rate or")
        print("     adding L2 regularisation to the projector weights.")
    print(f"{'='*60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
