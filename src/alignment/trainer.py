"""
src/alignment/trainer.py

Full training loop for embedding alignment.
Orchestrates model encoding (sequential GPU offloading), projector updates,
InfoNCE loss, structure preservation loss, validation retrieval accuracy,
checkpointing, and logging.

Entry point:  train_alignment(config_path)  — called from scripts/02_train_alignment.py
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

# ── local imports (package‑relative) ─────────────────────────────────────
from ..models.loader import load_all_models, sequential_encode
from ..utils.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from .loss import InfoNCELoss, StructurePreservationLoss, compute_retrieval_accuracy, stack_embeddings
from .projector import ProjectorBank

try:
    from scipy.stats import spearmanr as _spearmanr
except ImportError:
    _spearmanr = None


def _emit_json(data: dict) -> None:
    """Emit a JSON line to stdout for UI monitoring."""
    print(json.dumps(data, default=str), flush=True)


def _load_data_sources(data_cfg: dict) -> List[str]:
    """Load prompt texts from configured HF datasets.

    Args:
        data_cfg: the 'data' section from alignment.yaml.

    Returns:
        List of prompt strings (deduplicated, shuffled).
    """
    from datasets import load_dataset

    prompts: List[str] = []
    for src in data_cfg["sources"]:
        name = src["name"]
        max_samples = src.get("max_samples")
        split = src.get("split", "train")
        subset = src.get("subset")
        print(f"  [data] Loading {name} ({split})…")

        if "hf_path" not in src:
            continue
        if subset:
            ds = load_dataset(src["hf_path"], subset, split=split, streaming=True)
        else:
            ds = load_dataset(src["hf_path"], split=split, streaming=True)

        count = 0
        for example in ds:
            text = example.get("text", "").strip()
            if len(text) >= 50:
                prompts.append(text)
                count += 1
            if max_samples and count >= max_samples:
                break
        print(f"    → {count} prompts")

    # Deduplicate and shuffle
    prompts = list(dict.fromkeys(prompts))
    random.shuffle(prompts)
    print(f"  [data] Total unique prompts: {len(prompts)}")
    return prompts


def _make_batches(items: List[str], batch_size: int) -> List[List[str]]:
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])
    return batches


def _validate(
    models,
    bank: ProjectorBank,
    val_prompts: List[str],
    device: str,
    max_seq_len: int,
    batch_size: int,
) -> float:
    """Compute cross‑model retrieval accuracy on the validation set."""
    bank.eval()
    all_retrieval: List[float] = []
    batches = _make_batches(val_prompts, batch_size)

    with torch.no_grad():
        for batch in batches:
            raw = sequential_encode(models, batch, max_length=max_seq_len)
            # Move to device before projecting
            on_device = {mid: emb.to(device) for mid, emb in raw.items()}
            projected = bank(on_device)
            acc = compute_retrieval_accuracy(projected)
            all_retrieval.append(acc)

    bank.train()
    return float(np.mean(all_retrieval)) if all_retrieval else 0.0


def _compute_diagnostics(
    models,
    bank: ProjectorBank,
    val_prompts: List[str],
    device: str,
    max_seq_len: int,
) -> dict:
    """Compute Spearman neighborhood preservation + anti-collapse on val prompts.

    Lightweight version: uses a small subset to avoid slowing down training.
    """
    bank.eval()
    n = min(50, len(val_prompts))
    sample = val_prompts[:n]

    result = {}
    with torch.no_grad():
        raw = sequential_encode(models, sample, max_length=max_seq_len)

        for mid in sorted(models.keys()):
            on_device = {mid: raw[mid].to(device)}
            proj = bank(on_device)[mid]
            raw_np = raw[mid].cpu().float().numpy()
            proj_np = proj.cpu().float().numpy()

            # Spearman correlation
            if _spearmanr is not None:
                raw_dists = []
                proj_dists = []
                for i in range(n):
                    for j in range(i + 1, n):
                        r_sim = float(F.cosine_similarity(
                            torch.tensor(raw_np[i:i+1]), torch.tensor(raw_np[j:j+1])
                        ))
                        p_sim = float(F.cosine_similarity(
                            torch.tensor(proj_np[i:i+1]), torch.tensor(proj_np[j:j+1])
                        ))
                        raw_dists.append(r_sim)
                        proj_dists.append(p_sim)
                corr, _ = _spearmanr(raw_dists, proj_dists)
                result[f"sp_{mid}"] = float(corr)

            # Anti-collapse: mean projected cosine
            proj_sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    s = float(F.cosine_similarity(
                        torch.tensor(proj_np[i:i+1]), torch.tensor(proj_np[j:j+1])
                    ))
                    proj_sims.append(s)
            result[f"mean_cos_{mid}"] = float(np.mean(proj_sims))

    bank.train()
    return result


def train_alignment(
    config_path: str = "configs/alignment.yaml",
    models_config_path: str = "configs/models.yaml",
    output_dir_override: Optional[str] = None,
    structure_weight_override: Optional[float] = None,
) -> int:
    """Run the full alignment training loop.

    Returns 0 on success, 1 on failure.
    """
    # ═══════════════════════════════════════════════════════════════
    # 1. Load config
    # ═══════════════════════════════════════════════════════════════
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    proj_cfg = cfg["projector"]
    ckpt_cfg = cfg["checkpointing"]
    log_cfg = cfg["logging"]

    batch_size: int = train_cfg["batch_size"]
    lr: float = train_cfg["learning_rate"]
    weight_decay: float = train_cfg["weight_decay"]
    epochs: int = train_cfg["epochs"]
    temperature: float = train_cfg["temperature"]
    max_seq_len: int = train_cfg["max_seq_len"]
    grad_clip: float = train_cfg["grad_clip"]
    bf16: bool = train_cfg.get("bf16", False)
    structure_weight: float = train_cfg.get("structure_weight", 0.1)
    if structure_weight_override is not None:
        structure_weight = structure_weight_override
    seed: int = train_cfg["seed"]

    val_split: float = data_cfg["validation_split"]
    data_seed: int = data_cfg["seed"]

    save_every: int = ckpt_cfg["save_every_n_steps"]
    output_dir: Path = Path(
        output_dir_override if output_dir_override else ckpt_cfg["output_dir"]
    )
    resume_from: Optional[str] = ckpt_cfg.get("resume_from")

    tb_dir: str = log_cfg["tensorboard_dir"]
    log_every: int = log_cfg["log_every_n_steps"]

    # ═══════════════════════════════════════════════════════════════
    # 2. Reproducibility
    # ═══════════════════════════════════════════════════════════════
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if bf16 and device.startswith("cuda") else torch.float32
    print(f"  [trainer] Device: {device}   dtype: {dtype}")

    # ═══════════════════════════════════════════════════════════════
    # 3. Load data
    # ═══════════════════════════════════════════════════════════════
    print("─" * 60)
    print("  Loading data…")
    all_prompts = _load_data_sources(data_cfg)
    random.Random(data_seed).shuffle(all_prompts)

    n_val = max(1, int(len(all_prompts) * val_split))
    val_prompts = all_prompts[:n_val]
    train_prompts = all_prompts[n_val:]
    print(f"  Train prompts: {len(train_prompts)}   Val prompts: {len(val_prompts)}")

    # ═══════════════════════════════════════════════════════════════
    # 4. Load models
    # ═══════════════════════════════════════════════════════════════
    print("─" * 60)
    print("  Loading base models…")
    models = load_all_models(config_path=models_config_path, encoding_device=device)
    model_dims = {mid: m.hidden_dim for mid, m in models.items()}
    print(f"  Models: {list(models.keys())}")

    # ═══════════════════════════════════════════════════════════════
    # 5. Build projector bank
    # ═══════════════════════════════════════════════════════════════
    print("─" * 60)
    bank = ProjectorBank(
        model_dims=model_dims,
        hidden_dim=proj_cfg.get("hidden_dim", 1024),
        output_dim=proj_cfg.get("output_dim", 1024),
        dropout=proj_cfg.get("dropout", 0.1),
        activation=proj_cfg.get("activation", "gelu"),
    )
    print(f"  ProjectorBank: {len(bank.projectors)} projectors → dim {bank.common_dim}")

    # ═══════════════════════════════════════════════════════════════
    # 6. Optimizer, loss, logger
    # ═══════════════════════════════════════════════════════════════
    optimizer = torch.optim.AdamW(
        bank.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2,
    )
    criterion = InfoNCELoss(temperature=temperature)
    structure_criterion = StructurePreservationLoss()
    writer = SummaryWriter(log_dir=tb_dir)

    # ═══════════════════════════════════════════════════════════════
    # 7. Resume or start fresh
    # ═══════════════════════════════════════════════════════════════
    global_step = 0
    start_epoch = 0

    if resume_from:
        ckpt_path = resume_from
    else:
        ckpt_path = find_latest_checkpoint(str(output_dir))

    if ckpt_path:
        print(f"  [trainer] Resuming from {ckpt_path}")
        global_step = load_checkpoint(
            bank, optimizer, scheduler, ckpt_path, device,
        )
        start_epoch = global_step // max(1, len(_make_batches(train_prompts, batch_size)))

    # Move bank to device (weights stay fp32; autocast handles bf16)
    bank = bank.to(device)

    # ═══════════════════════════════════════════════════════════════
    # 8. Training loop
    # ═══════════════════════════════════════════════════════════════
    print("─" * 60)
    print(f"  Training for {epochs} epochs (start_epoch={start_epoch})…")
    print(f"  Batch size: {batch_size}   LR: {lr}   τ: {temperature}")
    print(f"  Structure preservation weight: λ={structure_weight}")

    train_batches = _make_batches(train_prompts, batch_size)
    steps_per_epoch = len(train_batches)
    total_steps = steps_per_epoch * epochs

    # Emit training start event
    _emit_json({
        "type": "alignment_start",
        "total_epochs": epochs,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_rate": lr,
        "temperature": temperature,
        "structure_weight": structure_weight,
        "train_prompts": len(train_prompts),
        "val_prompts": len(val_prompts),
    })

    # BF16 does not need gradient scaling (unlike FP16).
    # Use autocast only — no GradScaler.

    for epoch in range(start_epoch, epochs):
        bank.train()
        epoch_loss = 0.0
        random.shuffle(train_prompts)
        train_batches = _make_batches(train_prompts, batch_size)

        for batch_idx, batch in enumerate(train_batches):
            # ── 8a. Encode prompts through all models ─────────────────
            raw_embeddings = sequential_encode(
                models, batch, max_length=max_seq_len,
            )

            # ── 8b. Project into common space ─────────────────────────
            # Move raw embeddings to device for projector forward pass
            on_device: Dict[str, torch.Tensor] = {}
            for mid, emb in raw_embeddings.items():
                on_device[mid] = emb.to(device=device)  # fp32; autocast handles bf16

            projected = bank(on_device)  # {mid: Tensor[B, 1024]}

            # ── 8c. Stack & compute loss ──────────────────────────────
            stacked = stack_embeddings(projected)  # (B, M, D), L2-normed

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=dtype):
                # InfoNCE: aligns identical prompts across models
                loss_nce = criterion(stacked)
                # Structure preservation: preserves relative distances
                loss_struct = structure_criterion(raw_embeddings, projected)
                # Combined loss
                loss = loss_nce + structure_weight * loss_struct
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bank.parameters(), grad_clip)
            optimizer.step()

            scheduler.step()

            # ── 8d. Logging ───────────────────────────────────────────
            global_step += 1
            epoch_loss += loss.item()

            if global_step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/loss_nce", loss_nce.item(), global_step)
                writer.add_scalar("train/loss_struct", loss_struct.item(), global_step)
                writer.add_scalar("train/loss_total", loss.item(), global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                print(
                    f"  [E{epoch+1:02d} | step {global_step:05d}] "
                    f"nce={loss_nce.item():.4f}  struct={loss_struct.item():.4f}  "
                    f"total={loss.item():.4f}  lr={current_lr:.2e}"
                )
                # Emit JSON progress for UI monitoring
                _emit_json({
                    "type": "alignment_progress",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "step": global_step,
                    "total_steps": total_steps,
                    "nce": loss_nce.item(),
                    "struct": loss_struct.item(),
                    "total": loss.item(),
                    "lr": current_lr,
                    "phase": "train",
                })

            # ── 8e. Checkpoint ────────────────────────────────────────
            if global_step % save_every == 0:
                ckpt_path = output_dir / f"step_{global_step:06d}.pt"
                save_checkpoint(
                    bank, optimizer, scheduler, global_step,
                    {"epoch": epoch, "config": cfg},
                    ckpt_path,
                )

        # ── end of epoch ───────────────────────────────────────────────
        avg_loss = epoch_loss / max(1, steps_per_epoch)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # ── Validation ─────────────────────────────────────────────────
        val_acc = _validate(
            models, bank, val_prompts[:500],  # cap validation to 500 prompts
            device, max_seq_len, batch_size,
        )
        writer.add_scalar("val/retrieval_accuracy", val_acc, epoch)
        print(
            f"  [E{epoch+1:02d}] avg_loss={avg_loss:.4f}  "
            f"val_retrieval_acc={val_acc:.4f}"
        )

        # ── Diagnostics: Spearman + anti-collapse ─────────────────────
        diag = _compute_diagnostics(
            models, bank, val_prompts[:50], device, max_seq_len,
        )
        diag_str = "  ".join(f"{k}={v:.4f}" for k, v in diag.items())
        print(f"           diag: {diag_str}")

        # Emit epoch summary for UI monitoring
        _emit_json({
            "type": "alignment_epoch",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "avg_loss": avg_loss,
            "val_retrieval_acc": val_acc,
            **diag,
        })

        # ── Epoch checkpoint ───────────────────────────────────────────
        ckpt_path = output_dir / f"epoch_{epoch+1:03d}.pt"
        save_checkpoint(
            bank, optimizer, scheduler, global_step,
            {"epoch": epoch, "config": cfg},
            ckpt_path,
        )

    # ═══════════════════════════════════════════════════════════════
    # 9. Final checkpoint & cleanup
    # ═══════════════════════════════════════════════════════════════
    final_path = output_dir / "final.pt"
    save_checkpoint(
        bank, optimizer, scheduler, global_step,
        {"epoch": epochs, "config": cfg},
        final_path,
    )
    writer.close()
    print(f"  [trainer] Done.  Final checkpoint → {final_path}")
    return 0
