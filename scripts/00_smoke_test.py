#!/usr/bin/env python3
"""
scripts/00_smoke_test.py  —  Phase 0.5 Alignment Smoke Test

Self‑contained validation that cross‑model contrastive alignment is achievable
BEFORE committing to multi‑day training runs.

What it does:
  1. Loads 200 C4 prompts.
  2. Encodes each prompt through all 3 base models (sequential GPU offloading).
  3. Inlines one linear projector per model (hidden_dim → 64, no GELU, no L2 norm).
  4. Computes cosine similarity for same‑prompt vs different‑prompt projections.
  5. Runs a paired t‑test (p < 0.01, mean delta > 0.05 required to pass).
  6. Logs all scalar and histogram data to TensorBoard.

Does NOT import from src/alignment/ — projectors are defined right here.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats
from torch.utils.tensorboard import SummaryWriter

# ── add project root to path so we can import src.models.loader ──────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.loader import load_all_models, sequential_encode  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Inlined configuration
# ═══════════════════════════════════════════════════════════════════════════

NUM_PROMPTS = 200
PROJECTOR_DIM = 64                # intentionally small — we want a hard test
MAX_SEQ_LEN = 256
SEED = 42
P_THRESHOLD = 0.01
MEAN_DELTA_THRESHOLD = 0.05
TB_DIR = "runs/smoke_test"


# ═══════════════════════════════════════════════════════════════════════════
# Minimal inlined projector (no GELU, no L2‑norm – deliberately simplest)
# ═══════════════════════════════════════════════════════════════════════════

class InlineProjector(nn.Module):
    """Single linear layer: hidden_dim → PROJECTOR_DIM.  No activation, no norm."""

    def __init__(self, d_in: int, d_out: int = PROJECTOR_DIM):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
        # He-init so outputs are in a reasonable range for cosine similarity
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_inline_projectors(model_dims: Dict[str, int]) -> nn.ModuleDict:
    projectors = nn.ModuleDict()
    for model_id, dim in model_dims.items():
        projectors[model_id] = InlineProjector(dim)
    return projectors


# ═══════════════════════════════════════════════════════════════════════════
# Cosine similarity computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_cos_sims(
    proj_embeddings: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    """
    For every model pair (a, b), compute:
      - same_prompt[i] = cos(proj_a[i], proj_b[i])
      - diff_prompt[i] = cos(proj_a[i], proj_b[(i+1) % N])

    Returns {f"{a}_{b}_same": array, f"{a}_{b}_diff": array}.
    """
    model_ids = sorted(proj_embeddings.keys())
    N = next(iter(proj_embeddings.values())).shape[0]
    results: Dict[str, np.ndarray] = {}

    for i, a in enumerate(model_ids):
        for b in model_ids[i + 1:]:
            xa = F.normalize(proj_embeddings[a], p=2, dim=-1).cpu().numpy()
            xb = F.normalize(proj_embeddings[b], p=2, dim=-1).cpu().numpy()

            same = (xa * xb).sum(axis=-1)                        # [N]
            diff = (xa * np.roll(xb, shift=1, axis=0)).sum(axis=-1)  # [N]

            results[f"{a}_{b}_same"] = same
            results[f"{a}_{b}_diff"] = diff
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Statistical test
# ═══════════════════════════════════════════════════════════════════════════

def run_paired_ttest(pairs: Dict[str, np.ndarray]) -> dict:
    """
    Pool same‑prompt and diff‑prompt cosine similarities across all model pairs
    and run a paired t‑test.

    Returns dict with keys: t_stat, p_value, mean_same, mean_diff, mean_delta, passed.
    """
    all_same = []
    all_diff = []
    for key in list(pairs.keys()):
        if key.endswith("_same"):
            pair_key = key.replace("_same", "")
            if f"{pair_key}_diff" in pairs:
                all_same.append(pairs[key])
                all_diff.append(pairs[f"{pair_key}_diff"])

    if not all_same:
        return {"passed": False, "error": "No pairs found"}

    same = np.concatenate(all_same)
    diff = np.concatenate(all_diff)

    # Down-sample diff to match length of same if needed
    if len(diff) > len(same):
        diff = diff[:len(same)]

    t_stat, p_value = stats.ttest_rel(same, diff)
    mean_same = float(np.mean(same))
    mean_diff = float(np.mean(diff))
    mean_delta = mean_same - mean_diff
    passed = p_value < P_THRESHOLD and mean_delta > MEAN_DELTA_THRESHOLD

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_same": mean_same,
        "mean_diff": mean_diff,
        "mean_delta": mean_delta,
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_c4_prompts(n: int = NUM_PROMPTS, seed: int = SEED) -> List[str]:
    """Load n prompts from the C4 'en' validation set."""
    print(f"  [data] Loading {n} C4 prompts…")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts: List[str] = []
    rng = random.Random(seed)
    for example in ds:
        text = example["text"].strip()
        if 50 <= len(text) <= 512:          # skip tiny / huge chunks
            prompts.append(text)
        if len(prompts) >= n:
            break
    # Shuffle so we don't get contiguous paragraphs
    rng.shuffle(prompts)
    print(f"  [data] Loaded {len(prompts)} prompts")
    return prompts


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 0.5 Alignment Smoke Test")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tb-dir", default=TB_DIR)
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU even if GPU available (will be slower)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu_only else "cpu"
    print(f"  [init] Device: {device}")

    writer = SummaryWriter(log_dir=args.tb_dir)
    print(f"  [init] TensorBoard → {args.tb_dir}")

    # ── 1. Load prompts ──────────────────────────────────────────────────
    prompts = load_c4_prompts(n=args.num_prompts, seed=args.seed)

    # ── 2. Load models ───────────────────────────────────────────────────
    print("  [models] Loading base models (lazy CPU)…")
    models = load_all_models(encoding_device=device)
    model_dims = {mid: m.hidden_dim for mid, m in models.items()}
    print(f"  [models] {len(models)} models registered: {list(models.keys())}")

    # ── 3. Encode all prompts (sequential offloading, chunked) ──────────
    print("  [encode] Running sequential encoding (chunked)…")
    ENC_BATCH = 16  # small batches to avoid OOM on logits
    raw_embeddings: Dict[str, List[torch.Tensor]] = {mid: [] for mid in models}
    for i in range(0, len(prompts), ENC_BATCH):
        chunk = prompts[i : i + ENC_BATCH]
        chunk_emb = sequential_encode(models, chunk, max_length=MAX_SEQ_LEN)
        for mid in models:
            raw_embeddings[mid].append(chunk_emb[mid])
        print(f"    chunk {i//ENC_BATCH + 1}/{(len(prompts) + ENC_BATCH - 1)//ENC_BATCH}")
    # Concatenate chunks
    raw_embeddings = {mid: torch.cat(chunks, dim=0) for mid, chunks in raw_embeddings.items()}
    for mid, emb in raw_embeddings.items():
        print(f"    {mid}: {emb.shape}  ({emb.dtype})")
        writer.add_histogram(f"raw_embedding/{mid}", emb.numpy(), 0)

    # ── 4. Build & apply inline projectors ───────────────────────────────
    print("  [project] Building inline linear projectors (→ 64 dim)…")
    projectors = build_inline_projectors(model_dims)
    projectors = projectors.to(device)
    proj_embeddings: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for mid, raw in raw_embeddings.items():
            x = raw.to(device)
            proj_embeddings[mid] = projectors[mid](x).cpu()
            print(f"    {mid}: projected → {proj_embeddings[mid].shape}")
            writer.add_histogram(f"proj_embedding/{mid}", proj_embeddings[mid].numpy(), 0)

    # ── 5. Compute cosine similarities ───────────────────────────────────
    print("  [similarity] Computing cosine similarities…")
    pairs = compute_cos_sims(proj_embeddings)
    for key, arr in sorted(pairs.items()):
        print(f"    {key}: mean={np.mean(arr):.4f}  std={np.std(arr):.4f}")
        writer.add_scalar(f"cos_sim/{key}", float(np.mean(arr)), 0)
        writer.add_histogram(f"cos_sim_hist/{key}", arr, 0)

    # ── 6. Paired t‑test ─────────────────────────────────────────────────
    print("  [stats] Running paired t‑test…")
    result = run_paired_ttest(pairs)
    print(f"    t = {result.get('t_stat', 'N/A'):.4f}")
    print(f"    p = {result.get('p_value', 'N/A'):.6f}")
    print(f"    mean_same = {result.get('mean_same', 0):.4f}")
    print(f"    mean_diff = {result.get('mean_diff', 0):.4f}")
    print(f"    mean_delta = {result.get('mean_delta', 0):.4f}")

    for k in ("t_stat", "p_value", "mean_same", "mean_diff", "mean_delta"):
        if k in result:
            writer.add_scalar(f"ttest/{k}", result[k], 0)

    passed = result.get("passed", False)
    writer.add_scalar("ttest/passed", 1 if passed else 0, 0)

    # ── 7. Verdict ───────────────────────────────────────────────────────
    writer.close()
    print()
    if passed:
        print("  ✅ SMOKE TEST PASSED")
        print(f"     Cross-model same-prompt similarity is significantly higher")
        print(f"     than different-prompt similarity (p={result['p_value']:.6f}).")
        print(f"     The core alignment hypothesis is VALIDATED.")
        return 0
    else:
        print("  ❌ SMOKE TEST FAILED")
        if "p_value" in result and result["p_value"] >= P_THRESHOLD:
            print(f"     p-value {result['p_value']:.6f} ≥ {P_THRESHOLD}")
        if result.get("mean_delta", 0) <= MEAN_DELTA_THRESHOLD:
            print(f"     mean delta {result.get('mean_delta', 0):.4f} ≤ {MEAN_DELTA_THRESHOLD}")
        print("     Cross-model alignment could not be validated with the")
        print("     minimal inlined projector. This does NOT mean the")
        print("     approach is broken — a trained non-linear projector")
        print("     (script 02) may still succeed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
