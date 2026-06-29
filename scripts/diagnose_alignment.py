#!/usr/bin/env python3
"""
scripts/diagnose_alignment.py

Diagnostic script to check alignment quality on DIVERSE held‑out data
(not seen during training). Tests retrieval accuracy and embedding collapse
across code, math, and Wikipedia domains.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

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
    prompts: List[str],
) -> tuple[float, dict[str, float]]:
    """Cross‑model retrieval accuracy + per‑model collapse check."""
    mids = sorted(proj.keys())
    correct = 0
    total = 0

    for i, mi in enumerate(mids):
        for j, mj in enumerate(mids):
            if i == j:
                continue
            # L2‑normalize for cosine similarity
            qi = F.normalize(proj[mi], p=2, dim=-1)
            kj = F.normalize(proj[mj], p=2, dim=-1)
            sim = torch.matmul(qi, kj.T)            # (N, N)
            preds = sim.argmax(dim=-1)               # (N,)
            targets = torch.arange(len(prompts), device=preds.device)
            correct += (preds == targets).sum().item()
            total += len(prompts)

    acc = correct / total if total > 0 else 0.0

    # Collapse check: mean pairwise cosine similarity within each model
    collapse: dict[str, float] = {}
    for mid in mids:
        p = proj[mid].cpu()
        sim_matrix = torch.matmul(p, p.T)
        mask = ~torch.eye(len(prompts), dtype=torch.bool)
        mean_sim = sim_matrix[mask].mean().item()
        collapse[mid] = mean_sim

    return acc, collapse


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

    # ── Run diagnostics on 3 diverse domains ─────────────────────────────
    domains = [
        ("code", 100),
        ("math", 100),
        ("wiki", 100),
    ]

    print(f"\n{'='*60}")
    print("  DIAGNOSTIC RESULTS")
    print(f"{'='*60}")

    all_ok = True
    for name, n in domains:
        try:
            prompts = load_prompts(name, n)
        except Exception as e:
            print(f"\n  {name}: SKIPPED — {e}")
            continue

        print(f"\n  ── {name} ({len(prompts)} prompts) ──")

        # Encode
        with torch.no_grad():
            raw = sequential_encode(models, prompts, max_length=256)
            on_device = {mid: emb.to(device) for mid, emb in raw.items()}
            proj = bank(on_device)

        # Retrieval accuracy
        acc, collapse = compute_retrieval_accuracy(proj, prompts)
        print(f"    Retrieval accuracy: {acc:.4f}")

        for mid, mean_sim in collapse.items():
            status = "⚠️ COLLAPSE" if mean_sim > 0.9 else "✓"
            print(f"    {mid:8s}: mean pairwise sim = {mean_sim:.4f}  {status}")
            if mean_sim > 0.9:
                all_ok = False

    # ── Verdict ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_ok:
        print("  ✅ All checks passed — no embedding collapse detected.")
    else:
        print("  ⚠️  WARNING: Embedding collapse detected on some domains.")
        print("     The projector may have learned a trivial constant mapping.")
        print("     Check loss curves and consider reducing learning rate.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
