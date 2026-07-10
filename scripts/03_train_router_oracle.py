#!/usr/bin/env python3
"""
scripts/03_train_router_oracle.py

Oracle‑bootstrapped router training.

Loads:
  - oracle_labels.jsonl (ground‑truth best‑model labels)
  - Trained ProjectorBank (from alignment training)
  - Base models (for encoding prompts through projectors)

Trains the TransformerRouter with cross‑entropy loss to predict which model
is best for each prompt from projected embeddings.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.alignment.loss import stack_embeddings                        # noqa: E402
from src.alignment.projector import ProjectorBank                      # noqa: E402
from src.models.loader import load_all_models, sequential_encode       # noqa: E402
from src.router.transformer_router import TransformerRouter            # noqa: E402
from src.utils.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_oracle_labels(path: str, soft_labels: bool = False) -> List[dict]:
    """Load oracle labels from JSONL.

    Returns list of dicts with keys: prompt, best_model, scores, model_answers.
    If soft_labels=True, builds soft target distributions from the scores field.
    """
    items: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"  [data] Loaded {len(items)} oracle labels from {path}")

    if soft_labels and items and "scores" in items[0]:
        print("  [data] Using soft labels (KL divergence from score distribution)")
    elif soft_labels:
        print("  [data] WARNING: soft_labels requested but scores field not found; falling back to hard labels")

    return items


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_router_oracle(
    oracle_path: str = "data/oracle_labels.jsonl",
    router_config_path: str = "configs/router.yaml",
    models_config_path: str = "configs/models.yaml",
    projectors_dir: str = "checkpoints/alignment",
    output_dir_override: str | None = None,
    soft_labels: bool = False,
) -> int:
    """Train the TransformerRouter on oracle labels.

    If soft_labels=True and scores are available, uses KL divergence against
    the score distribution instead of hard cross-entropy on best_model.

    Returns 0 on success.
    """
    # ── Load configs ────────────────────────────────────────────────────
    with open(router_config_path) as f:
        rcfg = yaml.safe_load(f)

    arch_cfg = rcfg["architecture"]
    train_cfg = rcfg["oracle_training"]
    ckpt_cfg = rcfg["checkpointing"]
    log_cfg = rcfg["logging"]

    with open(models_config_path) as f:
        mcfg = yaml.safe_load(f)

    model_ids = sorted([m["id"] for m in mcfg["base_models"]])
    num_models = len(model_ids)
    seed = train_cfg["seed"]

    batch_size: int = train_cfg["batch_size"]
    lr: float = train_cfg["learning_rate"]
    epochs: int = train_cfg["epochs"]
    save_every: int = ckpt_cfg["save_every_n_steps"]
    output_dir: Path = Path(output_dir_override if output_dir_override else ckpt_cfg["output_dir"])
    tb_dir: str = log_cfg["tensorboard_dir"]

    # ── Reproducibility ─────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  [init] Device: {device}")

    # ── Load oracle labels ──────────────────────────────────────────────
    oracle_items = load_oracle_labels(oracle_path, soft_labels=soft_labels)
    # Build model_id → index mapping
    model_to_idx = {mid: i for i, mid in enumerate(model_ids)}

    # Check if soft labels are usable
    use_soft = (
        soft_labels
        and oracle_items
        and "scores" in oracle_items[0]
        and oracle_items[0]["scores"]
    )
    if use_soft:
        print("  [train] Using KL divergence loss (soft labels from score distribution)")
    else:
        print("  [train] Using cross-entropy loss (hard labels from best_model)")

    # ── Load base models & projectors ───────────────────────────────────
    print("  [models] Loading base models…")
    models = load_all_models(encoding_device=device)
    model_dims = {mid: m.hidden_dim for mid, m in models.items()}

    print(f"  [projectors] Loading from {projectors_dir}…")
    bank = ProjectorBank(model_dims=model_dims)
    proj_ckpt = find_latest_checkpoint(projectors_dir)
    if proj_ckpt:
        load_checkpoint(bank, None, None, proj_ckpt, device)
        print(f"    Loaded projectors from {proj_ckpt}")
    else:
        print("    WARNING: No projector checkpoint found. Using random init.")
    bank = bank.to(device)
    bank.eval()
    for p in bank.parameters():
        p.requires_grad_(False)

    # ── Build router ────────────────────────────────────────────────────
    router = TransformerRouter(
        num_models=num_models,
        input_dim=arch_cfg["input_dim"],
        d_model=arch_cfg["input_dim"],
        num_layers=arch_cfg["num_layers"],
        num_heads=arch_cfg["num_heads"],
        ff_dim=arch_cfg["ff_dim"],
        top_k=arch_cfg["top_k"],
        dropout=arch_cfg["dropout"],
    )
    router = router.to(device)
    n_params = sum(p.numel() for p in router.parameters())
    print(f"  [router] {n_params:,} parameters")

    # Emit start event for UI progress tracking
    n_batches_per_epoch = -(-len(oracle_items) // batch_size)
    total_steps = epochs * n_batches_per_epoch
    print(json.dumps({
        "type": "router_train_start",
        "total_epochs": epochs,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_rate": lr,
        "soft_labels": use_soft,
        "num_models": num_models,
        "model_ids": model_ids,
        "n_params": n_params,
        "train_size": len(oracle_items),
    }, default=str), flush=True)

    # ── Optimizer & loss ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(oracle_items) // batch_size)
    if use_soft:
        criterion = nn.KLDivLoss(reduction="batchmean")
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Resume ──────────────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    latest = find_latest_checkpoint(str(output_dir))
    if latest:
        try:
            print(f"  [resume] Loading {latest}")
            global_step = load_checkpoint(router, optimizer, scheduler, latest, device)
            n_batches_per_epoch = -(-len(oracle_items) // batch_size)  # ceil division
            start_epoch = global_step // max(1, n_batches_per_epoch)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"  [resume] WARNING: Checkpoint incompatible (model count changed). Starting fresh.")
                global_step = 0
                start_epoch = 0
            else:
                raise

    writer = SummaryWriter(log_dir=tb_dir)

    # ── Split validation set once ───────────────────────────────────────
    val_n = min(200, len(oracle_items) // 5)
    random.shuffle(oracle_items)
    train_items = oracle_items[:-val_n] if val_n > 0 else oracle_items
    val_items = oracle_items[-val_n:] if val_n > 0 else oracle_items[-10:]

    # ── Training loop ───────────────────────────────────────────────────
    print(f"  [train] {epochs} epochs, {len(train_items)} train / {len(val_items)} val, batch={batch_size}")
    router.train()

    for epoch in range(start_epoch, epochs):
        random.shuffle(train_items)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_items), batch_size):
            batch_items = train_items[i : i + batch_size]
            prompts = [item["prompt"] for item in batch_items]

            # ── Encode prompts & project ────────────────────────────────
            raw = sequential_encode(models, prompts, max_length=256)
            with torch.no_grad():
                projected = bank({mid: emb.to(device) for mid, emb in raw.items()})
                stacked = stack_embeddings(projected)  # (B, M, D)

            # ── Forward pass ────────────────────────────────────────────
            logits, _ = router(stacked)  # (B, M)

            if use_soft:
                # Build soft target distributions from scores
                soft_targets = torch.zeros(len(batch_items), num_models, device=device)
                for j, item in enumerate(batch_items):
                    scores = item.get("scores", {})
                    for mid, sc in scores.items():
                        idx = model_to_idx.get(mid)
                        if idx is not None:
                            soft_targets[j, idx] = sc
                    # Normalize to probability distribution
                    row_sum = soft_targets[j].sum()
                    if row_sum > 0:
                        soft_targets[j] /= row_sum
                    else:
                        # Fallback: uniform over models with non-zero scores
                        nonzero = [k for k, v in scores.items() if v > 0]
                        if nonzero:
                            for mid in nonzero:
                                soft_targets[j, model_to_idx[mid]] = 1.0 / len(nonzero)

                log_probs = F.log_softmax(logits, dim=-1)
                loss = criterion(log_probs, soft_targets)
            else:
                targets = torch.tensor(
                    [model_to_idx.get(item["best_model"], 0) for item in batch_items],
                    device=device,
                )
                loss = criterion(logits, targets)

            # ── Backward ────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1

            if global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                print(json.dumps({
                    "type": "router_train_step",
                    "step": global_step,
                    "total_steps": total_steps,
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "loss": round(loss.item(), 4),
                    "lr": round(scheduler.get_last_lr()[0], 8),
                }, default=str), flush=True)
                print(f"  [E{epoch+1:02d} | step {global_step:05d}] loss={loss.item():.4f}")

            if global_step % save_every == 0:
                ckpt_path = output_dir / f"step_{global_step:06d}.pt"
                save_checkpoint(router, optimizer, scheduler, global_step, {}, ckpt_path)

        # ── End of epoch ────────────────────────────────────────────────
        avg_loss = epoch_loss / max(1, n_batches)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # ── Validation on fixed held‑out subset ─────────────────────────
        router.eval()
        correct = 0
        with torch.no_grad():
            for j in range(0, len(val_items), batch_size):
                sub = val_items[j : j + batch_size]
                raw = sequential_encode(models, [it["prompt"] for it in sub], max_length=256)
                projected = bank({mid: emb.to(device) for mid, emb in raw.items()})
                stacked = stack_embeddings(projected)
                logits, _ = router(stacked)
                preds = logits.argmax(dim=-1).cpu()
                targets_t = torch.tensor(
                    [model_to_idx.get(it["best_model"], 0) for it in sub]
                )
                correct += (preds == targets_t).sum().item()

        val_acc = correct / len(val_items)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        print(json.dumps({
            "type": "router_train_epoch",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "avg_loss": round(avg_loss, 4),
            "val_acc": round(val_acc, 4),
            "global_step": global_step,
        }, default=str), flush=True)
        print(f"  [E{epoch+1:02d}] avg_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
        router.train()

        # ── Epoch checkpoint ────────────────────────────────────────────
        ckpt_path = output_dir / f"epoch_{epoch+1:03d}.pt"
        save_checkpoint(router, optimizer, scheduler, global_step, {}, ckpt_path)

    # ── Final ───────────────────────────────────────────────────────────
    final_path = output_dir / "final.pt"
    save_checkpoint(router, optimizer, scheduler, global_step, {}, final_path)

    # ── Versioned summary ───────────────────────────────────────────────
    from datetime import datetime
    import shutil
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = Path("results/router_training")
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": timestamp,
        "total_steps": global_step,
        "epochs": epochs,
        "final_val_acc": round(val_acc, 4),
        "final_train_loss": round(avg_loss, 4),
        "model_ids": model_ids,
        "soft_labels": use_soft,
        "n_params": n_params,
        "oracle_entries": len(oracle_items),
    }

    versioned_path = summary_dir / f"summary_{timestamp}.json"
    latest_path = summary_dir / "summary_latest.json"
    with open(versioned_path, "w") as f:
        json.dump(summary, f, indent=2)
    shutil.copy2(versioned_path, latest_path)

    # Update history index
    history_path = summary_dir / "history.json"
    history = []
    if history_path.exists():
        with open(history_path) as hf:
            history = json.load(hf)
    history.append(summary)
    with open(history_path, "w") as hf:
        json.dump(history, hf, indent=2)

    # Emit completion event
    print(json.dumps({
        "type": "router_train_complete",
        "final_val_acc": round(val_acc, 4),
        "final_train_loss": round(avg_loss, 4),
        "total_steps": global_step,
        "summary_file": versioned_path.name,
    }, default=str), flush=True)

    writer.close()
    print(f"  ✅ Training complete → {final_path}")
    print(f"  📊 Summary → {versioned_path}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train router with oracle labels")
    parser.add_argument("--oracle-labels", default="data/oracle_labels.jsonl")
    parser.add_argument("--config", default="configs/router.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--projectors-dir", default="checkpoints/alignment")
    parser.add_argument("--output-dir", default="checkpoints/router")
    parser.add_argument("--soft-labels", action="store_true",
                        help="Use KL divergence loss against score distribution instead of hard cross-entropy")
    args = parser.parse_args()

    print("=" * 60)
    print("  SAGA — Oracle Router Training")
    print(f"  Oracle:  {args.oracle_labels}")
    print(f"  Config:  {args.config}")
    print(f"  Projectors: {args.projectors_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Soft labels: {args.soft_labels}")
    print("=" * 60)

    sys.exit(
        train_router_oracle(
            oracle_path=args.oracle_labels,
            router_config_path=args.config,
            models_config_path=args.models_config,
            projectors_dir=args.projectors_dir,
            output_dir_override=args.output_dir,
            soft_labels=args.soft_labels,
        )
    )


if __name__ == "__main__":
    main()
