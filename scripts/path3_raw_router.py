"""
Path 3: Raw Embedding Router Experiment

Instead of using frozen projectors (which erase model identity),
use raw embeddings + per-model linear alignment layers.

Architecture:
  raw_codeqwen (1536) → Linear → 512
  raw_phi2 (2560) → Linear → 512
  raw_qwen (896) → Linear → 512
  raw_smollm (960) → Linear → 512
  concat → 2048 → Router MLP → 4 logits

This preserves model-specific geometry while allowing the router
to learn cross-model comparisons.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loader import load_all_models, sequential_encode
from src.utils.checkpointing import load_checkpoint


class RawRouter(nn.Module):
    """Per-model linear alignment + MLP classifier."""

    def __init__(self, model_dims: dict, hidden: int = 512, num_classes: int = 4):
        super().__init__()
        self.model_ids = sorted(model_dims.keys())
        # Per-model linear: raw_dim → hidden
        self.align = nn.ModuleDict({
            mid: nn.Linear(model_dims[mid], hidden)
            for mid in self.model_ids
        })
        # MLP on concatenated features
        total = hidden * len(self.model_ids)
        self.mlp = nn.Sequential(
            nn.Linear(total, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, raw_embeddings: dict):
        """
        Args:
            raw_embeddings: {model_id: Tensor[B, raw_dim]}
        Returns:
            logits: Tensor[B, num_classes]
        """
        parts = []
        for mid in self.model_ids:
            parts.append(self.align[mid](raw_embeddings[mid].float()))
        cat = torch.cat(parts, dim=-1)  # (B, hidden * num_models)
        return self.mlp(cat)


def encode_batch(models, prompts, model_ids, device, batch_size=32):
    """Encode prompts through raw models in batches."""
    all_raw = {mid: [] for mid in model_ids}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        raw = sequential_encode(models, batch, max_length=256)
        for mid in model_ids:
            all_raw[mid].append(raw[mid])

    return {mid: torch.cat(all_raw[mid], dim=0) for mid in model_ids}


def train_epoch(router, optimizer, raw_embeddings, targets, model_ids, batch_size=64):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    router.train()
    n = targets.shape[0]
    indices = torch.randperm(n)
    total_loss = 0.0
    correct = 0

    for i in range(0, n, batch_size):
        idx = indices[i:i + batch_size]
        batch = {mid: raw_embeddings[mid][idx].to(targets.device) for mid in model_ids}
        tgt = targets[idx]

        logits = router(batch)
        loss = F.cross_entropy(logits, tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)
        correct += (logits.argmax(-1) == tgt).sum().item()

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(router, raw_embeddings, targets, model_ids, num_classes):
    """Evaluate. Returns dict with accuracy, balanced accuracy, per-class metrics."""
    router.eval()
    device = targets.device
    n = targets.shape[0]

    all_logits = []
    for i in range(0, n, 256):
        batch = {mid: raw_embeddings[mid][i:i+256].to(device) for mid in model_ids}
        all_logits.append(router(batch))
    logits = torch.cat(all_logits, dim=0)
    preds = logits.argmax(dim=-1)

    # Overall accuracy
    acc = (preds == targets).float().mean().item()

    # Balanced accuracy (per-class recall averaged)
    per_class = {}
    balanced_acc = 0.0
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            recall = (preds[mask] == c).float().mean().item()
        else:
            recall = 0.0
        per_class[c] = recall
        balanced_acc += recall
    balanced_acc /= num_classes

    # Confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        confusion[t.item()][p.item()] += 1

    # Per-class precision
    for c in range(num_classes):
        col_sum = confusion[:, c].sum().item()
        tp = confusion[c, c].item()
        per_class[f"{c}_precision"] = tp / col_sum if col_sum > 0 else 0.0

    # Entropy
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * probs.log()).sum(dim=-1).mean().item()

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "per_class_recall": {c: per_class[c] for c in range(num_classes)},
        "per_class_precision": {f"{c}": per_class[f"{c}_precision"] for c in range(num_classes)},
        "mean_entropy": ent,
        "uniform_entropy": float(np.log(num_classes)),
        "confusion": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Path 3: Raw embedding router")
    parser.add_argument("--oracle", default="data/oracle_labels_latest.jsonl")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/path3_raw_router")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[path3] Device: {device}")

    # Load configs
    with open(args.models_config) as f:
        mcfg = yaml.safe_load(f)
    model_ids = sorted([m["id"] for m in mcfg["base_models"] if m.get("active", True)])
    model_to_idx = {mid: i for i, mid in enumerate(model_ids)}
    num_classes = len(model_ids)
    model_dims = {}
    for m in mcfg["base_models"]:
        if m["id"] in model_to_idx:
            model_dims[m["id"]] = m["hidden_dim"]

    print(f"[path3] Models: {model_ids}")
    print(f"[path3] Dims: {model_dims}")

    # Load oracle labels
    items = []
    with open(args.oracle) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    items = [it for it in items if it.get("best_model") in model_to_idx]
    print(f"[path3] Oracle labels: {len(items)} entries")

    # Split: 70% train, 15% val, 15% test
    random.shuffle(items)
    n = len(items)
    train_n = int(0.7 * n)
    val_n = int(0.15 * n)
    train_items = items[:train_n]
    val_items = items[train_n:train_n + val_n]
    test_items = items[train_n + val_n:]
    print(f"[path3] Split: {len(train_items)} train / {len(val_items)} val / {len(test_items)} test")

    # Load models
    print("[path3] Loading base models...")
    models = load_all_models(encoding_device=device)

    # Encode all data
    print("[path3] Encoding training set...")
    train_raw = encode_batch(models, [it["prompt"] for it in train_items], model_ids, device)
    train_targets = torch.tensor([model_to_idx[it["best_model"]] for it in train_items], device=device)

    print("[path3] Encoding validation set...")
    val_raw = encode_batch(models, [it["prompt"] for it in val_items], model_ids, device)
    val_targets = torch.tensor([model_to_idx[it["best_model"]] for it in val_items], device=device)

    print("[path3] Encoding test set...")
    test_raw = encode_batch(models, [it["prompt"] for it in test_items], model_ids, device)
    test_targets = torch.tensor([model_to_idx[it["best_model"]] for it in test_items], device=device)

    # Free GPU memory
    for wrapper in models.values():
        wrapper.offload_to_cpu()
    torch.cuda.empty_cache()

    # Build router
    router = RawRouter(model_dims, hidden=args.hidden, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in router.parameters())
    print(f"[path3] Router params: {n_params:,}")

    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Class weights (inverse frequency)
    train_counts = Counter(train_targets.cpu().tolist())
    weights = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        weights[c] = len(train_items) / (num_classes * max(train_counts.get(c, 1), 1))
    print(f"[path3] Class weights: { {model_ids[i]: round(weights[i].item(), 4) for i in range(num_classes)} }")

    # Train
    best_val_bal_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            router, optimizer, train_raw, train_targets, model_ids, args.batch_size,
        )
        scheduler.step()

        val_metrics = evaluate(router, val_raw, val_targets, model_ids, num_classes)
        test_metrics = evaluate(router, test_raw, test_targets, model_ids, num_classes)

        dt = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"  [E{epoch:02d}] loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.3f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"test_bal_acc={test_metrics['balanced_accuracy']:.3f} "
            f"entropy={val_metrics['mean_entropy']:.3f} "
            f"lr={lr_now:.2e} "
            f"({dt:.1f}s)"
        )

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_bal_acc": round(val_metrics["balanced_accuracy"], 4),
            "val_entropy": round(val_metrics["mean_entropy"], 4),
            "test_acc": round(test_metrics["accuracy"], 4),
            "test_bal_acc": round(test_metrics["balanced_accuracy"], 4),
            "lr": round(lr_now, 8),
        }
        history.append(record)

        if val_metrics["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_test = test_metrics
            torch.save(router.state_dict(), os.path.join(args.output, "best.pt"))

    # Load best and evaluate on test
    router.load_state_dict(torch.load(os.path.join(args.output, "best.pt")))
    final_test = evaluate(router, test_raw, test_targets, model_ids, num_classes)

    # Print results
    print(f"\n{'='*60}")
    print(f"[path3] RESULTS")
    print(f"{'='*60}")
    print(f"  Best epoch:        {best_epoch}")
    print(f"  Val balanced acc:  {best_val_bal_acc:.4f}")
    print(f"  Test accuracy:     {final_test['accuracy']:.4f}")
    print(f"  Test balanced acc: {final_test['balanced_accuracy']:.4f}")
    print(f"  Test entropy:      {final_test['mean_entropy']:.4f} (uniform={final_test['uniform_entropy']:.4f})")
    print(f"\n  Per-class recall (test):")
    for i, mid in enumerate(model_ids):
        print(f"    {mid}: {final_test['per_class_recall'][i]:.4f}")
    print(f"\n  Per-class precision (test):")
    for i, mid in enumerate(model_ids):
        print(f"    {mid}: {final_test['per_class_precision'][str(i)]:.4f}")
    print(f"\n  Confusion matrix (test):")
    cm = final_test["confusion"]
    header = "           " + "  ".join(f"{mid:>8s}" for mid in model_ids)
    print(header)
    for i, mid in enumerate(model_ids):
        row = f"  {mid:>8s}" + "  ".join(f"{cm[i][j]:>8d}" for j in range(num_classes))
        print(row)

    # Save summary
    summary = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "best_epoch": best_epoch,
        "val_bal_acc": round(best_val_bal_acc, 4),
        "test_acc": round(final_test["accuracy"], 4),
        "test_bal_acc": round(final_test["balanced_accuracy"], 4),
        "test_entropy": round(final_test["mean_entropy"], 4),
        "uniform_entropy": round(final_test["uniform_entropy"], 4),
        "model_ids": model_ids,
        "model_dims": model_dims,
        "hidden": args.hidden,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_params": n_params,
        "train_size": len(train_items),
        "val_size": len(val_items),
        "test_size": len(test_items),
        "per_class_recall": {model_ids[i]: round(final_test["per_class_recall"][i], 4) for i in range(num_classes)},
        "per_class_precision": {model_ids[i]: round(final_test["per_class_precision"][str(i)], 4) for i in range(num_classes)},
        "confusion": cm,
        "history": history,
    }

    os.makedirs(args.output, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output, f"summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output, "summary_latest.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Update history
    hist_path = os.path.join(args.output, "history.json")
    hist = []
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
    hist.append(summary)
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)

    print(f"\n  Saved → {args.output}/summary_{ts}.json")


if __name__ == "__main__":
    main()
