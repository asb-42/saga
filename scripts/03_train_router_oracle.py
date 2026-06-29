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
from typing import Dict, List, Tuple

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

def load_oracle_labels(path: str) -> List[dict]:
    """Load oracle labels from JSONL.

    Returns list of dicts with keys: prompt, best_model, scores, model_answers.
    """
    items: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"  [data] Loaded {len(items)} oracle labels from {path}")
    return items


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_router_oracle(
    oracle_path: str = "data/oracle_labels.jsonl",
    router_config_path: str = "configs/router.yaml",
    models_config_path: str = "configs/models.yaml",
    projectors_dir: str = "checkpoints/alignment",
) -> int:
    """Train the TransformerRouter on oracle labels.

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
    output_dir: Path = Path(ckpt_cfg["output_dir"])
    tb_dir: str = log_cfg["tensorboard_dir"]

    # ── Reproducibility ─────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  [init] Device: {device}")

    # ── Load oracle labels ──────────────────────────────────────────────
    oracle_items = load_oracle_labels(oracle_path)
    # Build model_id → index mapping
    model_to_idx = {mid: i for i, mid in enumerate(model_ids)}

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
    print(f"  [router] {sum(p.numel() for p in router.parameters()):,} parameters")

    # ── Optimizer & loss ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(oracle_items) // batch_size)
    criterion = nn.CrossEntropyLoss()

    # ── Resume ──────────────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    latest = find_latest_checkpoint(str(output_dir))
    if latest:
        print(f"  [resume] Loading {latest}")
        global_step = load_checkpoint(router, optimizer, scheduler, latest, device)
        start_epoch = global_step // max(1, len(oracle_items) // batch_size)

    writer = SummaryWriter(log_dir=tb_dir)

    # ── Training loop ───────────────────────────────────────────────────
    print(f"  [train] {epochs} epochs, {len(oracle_items)} samples, batch={batch_size}")
    router.train()

    for epoch in range(start_epoch, epochs):
        random.shuffle(oracle_items)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(oracle_items), batch_size):
            batch_items = oracle_items[i : i + batch_size]
            prompts = [item["prompt"] for item in batch_items]
            targets = torch.tensor(
                [model_to_idx.get(item["best_model"], 0) for item in batch_items],
                device=device,
            )

            # ── Encode prompts & project ────────────────────────────────
            raw = sequential_encode(models, prompts, max_length=256)
            with torch.no_grad():
                projected = bank({mid: emb.to(device) for mid, emb in raw.items()})
                stacked = stack_embeddings(projected)  # (B, M, D)

            # ── Forward pass ────────────────────────────────────────────
            logits, _ = router(stacked)  # (B, M)
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
                print(f"  [E{epoch+1:02d} | step {global_step:05d}] loss={loss.item():.4f}")

            if global_step % save_every == 0:
                ckpt_path = output_dir / f"step_{global_step:06d}.pt"
                save_checkpoint(router, optimizer, scheduler, global_step, {}, ckpt_path)

        # ── End of epoch ────────────────────────────────────────────────
        avg_loss = epoch_loss / max(1, n_batches)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # ── Validation: accuracy on held‑out subset ─────────────────────
        val_n = min(200, len(oracle_items) // 5)
        val_items = oracle_items[-val_n:]
        val_prompts = [item["prompt"] for item in val_items]
        val_targets = [model_to_idx.get(item["best_model"], 0) for item in val_items]

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
        print(f"  [E{epoch+1:02d}] avg_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
        router.train()

        # ── Epoch checkpoint ────────────────────────────────────────────
        ckpt_path = output_dir / f"epoch_{epoch+1:03d}.pt"
        save_checkpoint(router, optimizer, scheduler, global_step, {}, ckpt_path)

    # ── Final ───────────────────────────────────────────────────────────
    final_path = output_dir / "final.pt"
    save_checkpoint(router, optimizer, scheduler, global_step, {}, final_path)
    writer.close()
    print(f"  ✅ Training complete → {final_path}")
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
    args = parser.parse_args()

    print("=" * 60)
    print("  SAGA — Oracle Router Training")
    print(f"  Oracle:  {args.oracle_labels}")
    print(f"  Config:  {args.config}")
    print(f"  Projectors: {args.projectors_dir}")
    print(f"  Output:  {args.output_dir}")
    print("=" * 60)

    sys.exit(
        train_router_oracle(
            oracle_path=args.oracle_labels,
            router_config_path=args.config,
            models_config_path=args.models_config,
            projectors_dir=args.projectors_dir,
        )
    )


if __name__ == "__main__":
    main()
