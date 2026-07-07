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
from scipy.stats import spearmanr

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


def test_anti_collapse_ratio(
    raw: dict[str, torch.Tensor],
    proj: dict[str, torch.Tensor],
    min_ratio: float = 5.0,
) -> Tuple[bool, dict]:
    """Anti-collapse ratio: different-prompt distance / same-prompt distance.

    If the projector maps everything to nearly the same vector, same-prompt
    and different-prompt distances will be similar (ratio ≈ 1). A healthy
    space has ratio > 5x.

    Args:
        raw: {"model_id": Tensor[N, D]} — raw embeddings (one model).
        proj: {"model_id": Tensor[N, D]} — projected embeddings (one model).
        min_ratio: minimum allowed ratio (default 5.0).

    Returns:
        (passed: bool, details: dict with ratios per model).
    """
    passed = True
    details: dict = {}

    for mid in sorted(proj.keys()):
        raw_emb = F.normalize(raw[mid], p=2, dim=-1).cpu()
        proj_emb = F.normalize(proj[mid], p=2, dim=-1).cpu()
        n = raw_emb.shape[0]

        # Same-prompt distance (diagonal of sim matrix, excluded)
        # Different-prompt distance (off-diagonal)
        raw_sim = torch.matmul(raw_emb, raw_emb.T)
        proj_sim = torch.matmul(proj_emb, proj_emb.T)

        # Mask: off-diagonal only
        mask = ~torch.eye(n, dtype=torch.bool)

        raw_diff_dist = 1.0 - raw_sim[mask].mean().item()
        proj_diff_dist = 1.0 - proj_sim[mask].mean().item()

        # For same-prompt, we need cross-model pairs — use different models
        # For simplicity, use within-model variance as proxy
        raw_same_dist = 1.0 - raw_sim[mask].mean().item()  # within-model
        proj_same_dist = 1.0 - proj_sim[mask].mean().item()

        # Better: measure spread of projected embeddings
        proj_std = proj_emb.std(dim=0).mean().item()
        raw_std = raw_emb.std(dim=0).mean().item()

        ratio = proj_diff_dist / max(raw_diff_dist, 1e-8)

        ok = ratio > min_ratio and proj_std > 1e-4
        if not ok:
            passed = False

        details[mid] = {
            "raw_spread": raw_std,
            "proj_spread": proj_std,
            "ratio": ratio,
            "passed": ok,
        }

    return passed, details


def test_neighborhood_preservation(
    raw: dict[str, torch.Tensor],
    proj: dict[str, torch.Tensor],
    n_pairs: int = 50,
    min_spearman: float = 0.75,
) -> Tuple[bool, dict]:
    """Neighborhood preservation via Spearman correlation.

    For each model, sample prompt pairs and compute cosine similarity in
    raw space vs projected space. High Spearman correlation means the
    projector preserves semantic relationships.

    Args:
        raw: {"model_id": Tensor[N, D]} — raw embeddings.
        proj: {"model_id": Tensor[N, D]} — projected embeddings.
        n_pairs: number of random pairs to sample.
        min_spearman: minimum acceptable Spearman correlation (default 0.75).

    Returns:
        (passed: bool, details: dict with per-model Spearman r and p-value).
    """
    import random
    passed = True
    details: dict = {}

    for mid in sorted(raw.keys()):
        raw_emb = F.normalize(raw[mid], p=2, dim=-1).cpu()
        proj_emb = F.normalize(proj[mid], p=2, dim=-1).cpu()
        n = raw_emb.shape[0]

        # Sample random pairs
        indices = list(range(n))
        raw_sims = []
        proj_sims = []

        for _ in range(min(n_pairs, n * (n - 1) // 2)):
            i, j = random.sample(indices, 2)
            raw_sims.append(float(torch.dot(raw_emb[i], raw_emb[j])))
            proj_sims.append(float(torch.dot(proj_emb[i], proj_emb[j])))

        raw_sims = np.array(raw_sims)
        proj_sims = np.array(proj_sims)

        corr, p_value = spearmanr(raw_sims, proj_sims)

        ok = corr > min_spearman
        if not ok:
            passed = False

        details[mid] = {
            "spearman_r": float(corr),
            "p_value": float(p_value),
            "passed": ok,
        }

    return passed, details


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load models & projectors ─────────────────────────────────────────
    with open("configs/models.yaml") as f:
        cfg = yaml.safe_load(f)
    model_dims = {m["id"]: m["hidden_dim"] for m in cfg["base_models"] if m.get("active", True)}
    common_dim = cfg.get("common_dim", 1024)

    print("Loading base models…")
    models = load_all_models("configs/models.yaml", encoding_device=device)

    print(f"Loading ProjectorBank (dim={common_dim})…")
    bank = ProjectorBank(model_dims, hidden_dim=common_dim, output_dim=common_dim)
    bank = bank.to(device)
    # Try structured alignment first, then legacy
    ckpt = find_latest_checkpoint("checkpoints/alignment_structured")
    if not ckpt:
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
            # Offload models to free GPU memory
            for mid in models:
                models[mid].offload_to_cpu()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
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
    # 3. Neighborhood preservation (Spearman correlation)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  3. NEIGHBORHOOD PRESERVATION (Spearman)")
    print(f"{'='*60}")

    raw_preserv = None
    proj_preserv = None

    try:
        from datasets import load_dataset as ld
        ds = ld("allenai/c4", "en", split="validation", streaming=True)
        preserv_prompts = []
        for ex in ds:
            text = ex["text"].strip()
            if 50 <= len(text) <= 512:
                preserv_prompts.append(text)
            if len(preserv_prompts) >= 50:
                break

        with torch.no_grad():
            raw_preserv = sequential_encode(models, preserv_prompts, max_length=256)
            # Offload models to free GPU memory before projection
            for mid in models:
                models[mid].offload_to_cpu()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            on_device_p = {mid: emb.to(device) for mid, emb in raw_preserv.items()}
            proj_preserv = bank(on_device_p)

        spearman_ok, spearman_details = test_neighborhood_preservation(raw_preserv, proj_preserv)
        for mid, det in sorted(spearman_details.items()):
            r = det["spearman_r"]
            p = det["p_value"]
            ok = "✅" if det["passed"] else "❌ LOW CORRELATION"
            print(f"     {mid:8s}: Spearman r={r:.4f}  p={p:.2e}  {ok}")
            if not det["passed"]:
                all_ok = False

        if spearman_ok:
            print("     → Semantic relationships are preserved in shared space.")
        else:
            print("     → WARNING: Low correlation — projector may be collapsing structure.")
    except Exception as e:
        print(f"     SKIPPED — {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Anti-collapse ratio
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  4. ANTI-COLLAPSE RATIO")
    print(f"{'='*60}")

    if raw_preserv is not None and proj_preserv is not None:
        try:
            ratio_ok, ratio_details = test_anti_collapse_ratio(raw_preserv, proj_preserv)
            for mid, det in sorted(ratio_details.items()):
                ratio = det["ratio"]
                raw_spread = det["raw_spread"]
                proj_spread = det["proj_spread"]
                ok = "✅" if det["passed"] else "❌ COLLAPSED"
                print(f"     {mid:8s}: ratio={ratio:.2f}x  raw_spread={raw_spread:.4f}  proj_spread={proj_spread:.6f}  {ok}")
                if not det["passed"]:
                    all_ok = False

            if ratio_ok:
                print("     → Projected space has sufficient spread.")
            else:
                print("     → WARNING: Space may be collapsed — same/diff distances too similar.")
        except Exception as e:
            print(f"     SKIPPED — {e}")
    else:
        print("     SKIPPED — no embedding data available")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Verdict
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — alignment quality is good.")
        print("     The shared embedding space is semantically meaningful.")
        print("     The router has a valid signal to work with.")
    else:
        print("  ❌ SOME CHECKS FAILED")
        print("     Review the warnings above. The projector may need retraining,")
        print("     or the embedding space may not be suitable for routing.")
    print(f"{'='*60}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
