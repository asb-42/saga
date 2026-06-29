"""
src/utils/checkpointing.py

Full checkpoint save/load with resume capability.
Captures: model weights, optimizer state, scheduler state, global step, config.
"""
from __future__ import annotations
import torch
from pathlib import Path
from typing import Optional
import glob


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    config: dict,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "global_step": global_step,
            "config": config,
        },
        output_path,
    )
    print(f"  [checkpoint] Saved → {output_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    global_step = ckpt.get("global_step", 0)
    print(f"  [checkpoint] Resumed from step {global_step} ← {checkpoint_path}")
    return global_step


def find_latest_checkpoint(directory: str, pattern: str = "*.pt") -> Optional[str]:
    """Return the most recently modified checkpoint matching pattern, or None."""
    matches = glob.glob(str(Path(directory) / pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: Path(p).stat().st_mtime)
