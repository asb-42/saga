#!/usr/bin/env python3
"""
scripts/04_train_autoencoder.py

Trains the anomaly autoencoder on clean projected embeddings.

Workflow:
  1. Load base models + trained ProjectorBank.
  2. Encode clean prompts (from oracle labels or C4) through all models.
  3. Project into common space.
  4. Train the autoencoder on concatenated (B*M, D) clean embeddings.
  5. Save checkpoint.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.alignment.loss import stack_embeddings                      # noqa: E402
from src.alignment.projector import ProjectorBank                    # noqa: E402
from src.models.loader import load_all_models, sequential_encode     # noqa: E402
from src.router.autoencoder import AnomalyAutoencoder                # noqa: E402
from src.utils.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402


def _load_clean_prompts(num: int = 5000, seed: int = 42) -> List[str]:
    """Load clean C4 prompts for autoencoder training."""
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts: List[str] = []
    rng = random.Random(seed)
    for example in ds:
        text = example["text"].strip()
        if 50 <= len(text) <= 512:
            prompts.append(text)
        if len(prompts) >= num:
            break
    rng.shuffle(prompts)
    print(f"  [data] {len(prompts)} clean prompts loaded")
    return prompts


def train_autoencoder(
    router_config_path: str = "configs/router.yaml",
    models_config_path: str = "configs/models.yaml",
    projectors_dir: str = "checkpoints/alignment",
    num_prompts: int = 5000,
    output_dir: str = "checkpoints/autoencoder",
) -> int:
    with open(router_config_path) as f:
        rcfg = yaml.safe_load(f)

    ae_cfg = rcfg["autoencoder"]
    ckpt_cfg = rcfg["checkpointing"]
    log_cfg = rcfg["logging"]

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load projectors ─────────────────────────────────────────────────
    print("  [models] Loading base models…")
    models = load_all_models(encoding_device=device)
    model_dims = {mid: m.hidden_dim for mid, m in models.items()}

    print(f"  [projectors] Loading from {projectors_dir}…")
    bank = ProjectorBank(model_dims=model_dims)
    proj_ckpt = find_latest_checkpoint(projectors_dir)
    if proj_ckpt:
        load_checkpoint(bank, None, None, proj_ckpt, device)
    bank = bank.to(device)
    bank.eval()
    for p in bank.parameters():
        p.requires_grad_(False)

    # ── Encode clean prompts & project ──────────────────────────────────
    prompts = _load_clean_prompts(num=num_prompts)
    batch_size = 32
    all_embeddings: List[torch.Tensor] = []

    print(f"  [encode] Projecting {len(prompts)} prompts…")
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        raw = sequential_encode(models, batch, max_length=256)
        with torch.no_grad():
            projected = bank({mid: emb.to(device) for mid, emb in raw.items()})
            stacked = stack_embeddings(projected)  # (B, M, D)
            # Flatten to (B*M, D)
            flat = stacked.reshape(-1, stacked.shape[-1])
            all_embeddings.append(flat.cpu())
        if (i // batch_size + 1) % 50 == 0:
            print(f"    {i + len(batch)}/{len(prompts)}")

    clean_data = torch.cat(all_embeddings, dim=0)  # (N_total, 1024)
    print(f"  [encode] Clean embedding matrix: {clean_data.shape}")

    # ── Build autoencoder ───────────────────────────────────────────────
    ae = AnomalyAutoencoder(
        input_dim=1024,
        encoder_dims=ae_cfg["encoder_dims"],
        decoder_dims=ae_cfg["decoder_dims"],
        activation=ae_cfg["activation"],
    )
    ae = ae.to(device)

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    # ── Resume ──────────────────────────────────────────────────────────
    global_step = 0
    latest = find_latest_checkpoint(str(output_dir))
    if latest:
        print(f"  [resume] Loading {latest}")
        global_step = load_checkpoint(ae, optimizer, scheduler, latest, device)

    writer = SummaryWriter(log_dir=log_cfg["tensorboard_dir"])

    # ── Training loop ───────────────────────────────────────────────────
    epochs = 100
    batch_size_ae = 256
    n_samples = clean_data.shape[0]

    print(f"  [train] {epochs} epochs, {n_samples} samples, batch={batch_size_ae}")
    ae.train()

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size_ae):
            idx = perm[i : i + batch_size_ae]
            x = clean_data[idx].to(device)

            recon, scores = ae(x)
            loss = F.mse_loss(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        scheduler.step(avg_loss)
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        if (epoch + 1) % 10 == 0:
            print(f"  [epoch {epoch+1:03d}] loss={avg_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % 25 == 0:
            ckpt_path = output_dir / f"epoch_{epoch+1:03d}.pt"
            save_checkpoint(ae, optimizer, scheduler, global_step, {}, ckpt_path)

    final_path = output_dir / "final.pt"
    save_checkpoint(ae, optimizer, scheduler, global_step, {}, final_path)
    writer.close()
    print(f"  ✅ Autoencoder training complete → {final_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Train anomaly autoencoder")
    parser.add_argument("--config", default="configs/router.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--projectors-dir", default="checkpoints/alignment")
    parser.add_argument("--num-prompts", type=int, default=5000)
    parser.add_argument("--output-dir", default="checkpoints/autoencoder")
    args = parser.parse_args()

    print("=" * 60)
    print("  SAGA — Autoencoder Training")
    print(f"  Config:    {args.config}")
    print(f"  Output:    {args.output_dir}")
    print("=" * 60)

    sys.exit(
        train_autoencoder(
            router_config_path=args.config,
            models_config_path=args.models_config,
            projectors_dir=args.projectors_dir,
            num_prompts=args.num_prompts,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
