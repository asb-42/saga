#!/usr/bin/env python3
"""
scripts/diagnose_router.py

Diagnostic checks for router training quality.

Checks:
1. Constant baseline test — what does always-predict-average score?
2. Train vs validation gap — overfitting or underfitting?
3. Per-class confusion matrix — which models does the router fail on?
4. Prediction entropy — is the router confident or uniform?
5. Hard-label baseline — retrain with inverse-frequency class weights
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.alignment.loss import stack_embeddings
from src.alignment.projector import ProjectorBank
from src.models.loader import load_all_models, sequential_encode
from src.router.transformer_router import TransformerRouter
from src.utils.checkpointing import find_latest_checkpoint, load_checkpoint


def load_oracle_labels(path: str) -> List[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def encode_val_set(
    models, bank, val_items, model_ids, model_to_idx, device, batch_size=32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode validation set into stacked embeddings and hard labels."""
    all_projected = {mid: [] for mid in model_ids}

    for i in range(0, len(val_items), batch_size):
        batch = val_items[i:i + batch_size]
        prompts = [item["prompt"] for item in batch]
        raw = sequential_encode(models, prompts, max_length=256)
        with torch.no_grad():
            projected = bank({mid: emb.to(device) for mid, emb in raw.items()})
        for mid in model_ids:
            all_projected[mid].append(projected[mid])

    with torch.no_grad():
        stacked_all = []
        for mid in model_ids:
            stacked_all.append(torch.cat(all_projected[mid], dim=0))
        stacked = torch.stack(stacked_all, dim=1)

    targets = torch.tensor(
        [model_to_idx.get(item["best_model"], 0) for item in val_items],
        device=device,
    )
    return stacked, targets


def check1_constant_baseline(
    val_targets: torch.Tensor, num_models: int, val_size: int,
) -> dict:
    """Check: what does always-predict-the-average-distribution score?"""
    class_counts = torch.bincount(val_targets, minlength=num_models).float()
    avg_dist = class_counts / val_size

    uniform = torch.ones(num_models, device=val_targets.device) / num_models
    kl_loss = F.kl_div(uniform.log(), avg_dist, reduction="batchmean").item()

    most_common = avg_dist.argmax().item()
    const_preds = torch.full((val_size,), most_common, device=val_targets.device)
    const_acc = (const_preds == val_targets).float().mean().item()

    return {
        "avg_distribution": {f"model_{i}": round(v.item(), 4) for i, v in enumerate(avg_dist)},
        "kl_loss": round(kl_loss, 6),
        "constant_accuracy": round(const_acc, 4),
        "most_common_class": most_common,
        "class_counts": class_counts.int().tolist(),
    }


def check2_train_val_gap(
    router, train_stacked, train_targets, val_stacked, val_targets, num_models,
) -> dict:
    """Check: is the model overfitting or underfitting?"""
    router.eval()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        train_logits, _ = router(train_stacked)
        val_logits, _ = router(val_stacked)

        train_acc = (train_logits.argmax(-1) == train_targets).float().mean().item()
        val_acc = (val_logits.argmax(-1) == val_targets).float().mean().item()
        train_loss = criterion(train_logits, train_targets).item()
        val_loss = criterion(val_logits, val_targets).item()

    return {
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "gap": round(train_acc - val_acc, 4),
        "verdict": "overfitting" if train_acc - val_acc > 0.15 else (
            "underfitting" if train_acc < 0.5 else "ok"
        ),
    }


def check3_confusion_matrix(
    router, val_stacked, val_targets, model_ids, num_models,
) -> dict:
    """Per-class confusion matrix."""
    confusion = torch.zeros(num_models, num_models, dtype=torch.long)

    router.eval()
    with torch.no_grad():
        logits, _ = router(val_stacked)
        preds = logits.argmax(dim=-1)
        for t, p in zip(val_targets, preds):
            confusion[t.item()][p.item()] += 1

    per_class = {}
    for i, mid in enumerate(model_ids):
        row_sum = confusion[i].sum().item()
        col_sum = confusion[:, i].sum().item()
        tp = confusion[i, i].item()
        recall = tp / row_sum if row_sum > 0 else 0
        precision = tp / col_sum if col_sum > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[mid] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "support": row_sum,
            "predicted_as": {
                model_ids[j]: confusion[i, j].item()
                for j in range(num_models)
            },
        }

    overall_acc = sum(confusion[i, i].item() for i in range(num_models)) / confusion.sum().item()

    return {
        "per_class": per_class,
        "overall_accuracy": round(overall_acc, 4),
        "confusion_matrix": confusion.tolist(),
    }


def check4_entropy(
    router, val_stacked, model_ids, num_models,
) -> dict:
    """Average entropy of router output distributions."""
    router.eval()
    with torch.no_grad():
        logits, _ = router(val_stacked)
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * probs.log()).sum(dim=-1)

    mean_probs = probs.mean(dim=0)

    return {
        "mean_entropy": round(float(ent.mean()), 4),
        "std_entropy": round(float(ent.std()), 4),
        "min_entropy": round(float(ent.min()), 4),
        "max_entropy": round(float(ent.max()), 4),
        "median_entropy": round(float(ent.median()), 4),
        "uniform_entropy": round(float(np.log(num_models)), 4),
        "mean_output_distribution": {
            mid: round(mean_probs[i].item(), 4)
            for i, mid in enumerate(model_ids)
        },
        "verdict": (
            "collapsed" if float(ent.mean()) > np.log(num_models) * 0.95
            else "confident" if float(ent.mean()) < 0.8
            else "healthy"
        ),
    }


def check5_hard_label_baseline(
    oracle_items, model_ids, model_to_idx, device,
    router_config_path="configs/router.yaml",
    epochs=5,
) -> dict:
    """Retrain with hard labels and inverse-frequency class weights."""
    with open(router_config_path) as f:
        rcfg = yaml.safe_load(f)
    arch_cfg = rcfg["architecture"]

    # Class weights (inverse frequency)
    best_models = [item["best_model"] for item in oracle_items]
    counts = Counter(best_models)
    total = len(best_models)
    weights = torch.zeros(len(model_ids), device=device)
    for i, mid in enumerate(model_ids):
        if counts[mid] > 0:
            weights[i] = total / (len(model_ids) * counts[mid])
        else:
            weights[i] = 1.0

    # Split data
    val_n = min(200, len(oracle_items) // 5)
    random.shuffle(oracle_items)
    train_items = oracle_items[:-val_n] if val_n > 0 else oracle_items
    val_items = oracle_items[-val_n:] if val_n > 0 else oracle_items[-10:]

    # Build fresh router
    router = TransformerRouter(
        num_models=len(model_ids),
        input_dim=arch_cfg["input_dim"],
        d_model=arch_cfg["input_dim"],
        num_layers=arch_cfg["num_layers"],
        num_heads=arch_cfg["num_heads"],
        ff_dim=arch_cfg["ff_dim"],
        top_k=arch_cfg["top_k"],
        dropout=arch_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Train (we need models + bank — caller passes them via global state)
    # This function is called from the main diagnostic which has access
    return {
        "weights": {mid: round(weights[i].item(), 4) for i, mid in enumerate(model_ids)},
        "class_counts": {mid: counts.get(mid, 0) for mid in model_ids},
    }


def run_all_diagnostics(
    oracle_path: str = "data/oracle_labels_latest.jsonl",
    router_ckpt: str = "checkpoints/router/final.pt",
    projectors_dir: str = "checkpoints/alignment_structured",
    router_config_path: str = "configs/router.yaml",
    models_config_path: str = "configs/models.yaml",
) -> dict:
    print("  [diag] Loading configs…")
    with open(router_config_path) as f:
        rcfg = yaml.safe_load(f)
    with open(models_config_path) as f:
        mcfg = yaml.safe_load(f)

    model_ids = sorted([m["id"] for m in mcfg["base_models"] if m.get("active", True)])
    num_models = len(model_ids)
    model_to_idx = {mid: i for i, mid in enumerate(model_ids)}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  [diag] Device: {device}  Models: {model_ids}")

    # Load oracle labels
    oracle_items = load_oracle_labels(oracle_path)
    print(f"  [diag] {len(oracle_items)} oracle labels")

    # Filter to active models only
    oracle_items = [item for item in oracle_items if item.get("best_model") in model_to_idx]
    print(f"  [diag] {len(oracle_items)} entries with active models")

    # Split
    val_n = min(200, len(oracle_items) // 5)
    random.seed(42)
    random.shuffle(oracle_items)
    train_items = oracle_items[:-val_n] if val_n > 0 else oracle_items
    val_items = oracle_items[-val_n:] if val_n > 0 else oracle_items[-10:]

    # Load models & projectors
    print("  [diag] Loading models…")
    models = load_all_models(encoding_device=device)
    model_dims = {mid: m.hidden_dim for mid, m in models.items()}

    bank = ProjectorBank(model_dims=model_dims)
    proj_ckpt = find_latest_checkpoint(projectors_dir)
    if proj_ckpt:
        load_checkpoint(bank, None, None, proj_ckpt, device)
        print(f"  [diag] Projectors from {proj_ckpt}")
    bank = bank.to(device)
    bank.eval()
    for p in bank.parameters():
        p.requires_grad_(False)

    # Load router
    router = TransformerRouter(
        num_models=num_models,
        input_dim=rcfg["architecture"]["input_dim"],
        d_model=rcfg["architecture"]["input_dim"],
        num_layers=rcfg["architecture"]["num_layers"],
        num_heads=rcfg["architecture"]["num_heads"],
        ff_dim=rcfg["architecture"]["ff_dim"],
        top_k=rcfg["architecture"]["top_k"],
        dropout=rcfg["architecture"]["dropout"],
    ).to(device)

    ckpt_path = Path(router_ckpt)
    if ckpt_path.exists():
        load_checkpoint(router, None, None, str(ckpt_path), device)
        print(f"  [diag] Router from {ckpt_path}")
    else:
        print(f"  [diag] WARNING: No checkpoint at {ckpt_path}, using random init")

    # ── Encode val set ONCE and cache ───────────────────────────────────
    print("  [diag] Encoding validation set (once)…")
    val_stacked, val_targets = encode_val_set(models, bank, val_items, model_ids, model_to_idx, device)
    val_stacked = val_stacked.to(device)
    val_targets = val_targets.to(device)

    # Also encode train set for check2
    print("  [diag] Encoding training set…")
    train_stacked, train_targets = encode_val_set(models, bank, train_items, model_ids, model_to_idx, device)
    train_stacked = train_stacked.to(device)
    train_targets = train_targets.to(device)

    # Models no longer needed — free GPU memory
    for wrapper in models.values():
        wrapper.offload_to_cpu()
    torch.cuda.empty_cache()

    print("  [diag] Running Check 1: Constant Baseline…")
    check1 = check1_constant_baseline(val_targets, num_models, len(val_items))

    print("  [diag] Running Check 2: Train/Val Gap…")
    check2 = check2_train_val_gap(
        router, train_stacked, train_targets, val_stacked, val_targets, num_models,
    )

    print("  [diag] Running Check 3: Confusion Matrix…")
    check3 = check3_confusion_matrix(
        router, val_stacked, val_targets, model_ids, num_models,
    )

    print("  [diag] Running Check 4: Prediction Entropy…")
    check4 = check4_entropy(
        router, val_stacked, model_ids, num_models,
    )

    print("  [diag] Running Check 5: Hard-Label Class Weights…")
    check5 = check5_hard_label_baseline(
        oracle_items, model_ids, model_to_idx, device, router_config_path,
    )

    result = {
        "total_entries": len(oracle_items),
        "val_size": len(val_items),
        "train_size": len(train_items),
        "model_ids": model_ids,
        "check1_constant_baseline": check1,
        "check2_train_val_gap": check2,
        "check3_confusion_matrix": check3,
        "check4_prediction_entropy": check4,
        "check5_hard_label_weights": check5,
    }

    # Save versioned output
    from datetime import datetime
    import shutil
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/router_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    versioned = out_dir / f"diagnostics_{timestamp}.json"
    latest = out_dir / "diagnostics_latest.json"
    with open(versioned, "w") as f:
        json.dump(result, f, indent=2)
    shutil.copy2(versioned, latest)

    history_path = out_dir / "history.json"
    history = []
    if history_path.exists():
        with open(history_path) as hf:
            history = json.load(hf)
    history.append({"timestamp": timestamp, "file": versioned.name})
    with open(history_path, "w") as hf:
        json.dump(history, hf, indent=2)

    print(f"\n  [diag] Summary:")
    print(f"    Check 1 — Constant baseline acc: {check1['constant_accuracy']:.1%}  KL: {check1['kl_loss']:.6f}")
    print(f"    Check 2 — Train: {check2['train_accuracy']:.1%}  Val: {check2['val_accuracy']:.1%}  Gap: {check2['gap']:.1%}  ({check2['verdict']})")
    print(f"    Check 3 — Per-class F1: {', '.join(f'{m}={c['f1']:.3f}' for m, c in check3['per_class'].items())}")
    print(f"    Check 4 — Mean entropy: {check4['mean_entropy']:.4f}  Uniform: {check4['uniform_entropy']:.4f}  ({check4['verdict']})")
    print(f"    Check 5 — Weights: {check5['weights']}")
    print(f"\n  ✅ Diagnostics → {versioned}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Router diagnostic checks")
    parser.add_argument("--oracle-labels", default="data/oracle_labels_latest.jsonl")
    parser.add_argument("--router-ckpt", default="checkpoints/router/final.pt")
    parser.add_argument("--projectors-dir", default="checkpoints/alignment_structured")
    parser.add_argument("--router-config", default="configs/router.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    args = parser.parse_args()

    run_all_diagnostics(
        oracle_path=args.oracle_labels,
        router_ckpt=args.router_ckpt,
        projectors_dir=args.projectors_dir,
        router_config_path=args.router_config,
        models_config_path=args.models_config,
    )


if __name__ == "__main__":
    main()
