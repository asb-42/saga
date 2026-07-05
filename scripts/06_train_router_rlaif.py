#!/usr/bin/env python3
"""
scripts/06_train_router_rlaif.py

RLAIF fine‑tuning of the TransformerRouter using REINFORCE + KL penalty.

Loads:
  - Oracle‑trained router (frozen, KL anchor)
  - Trainable router (initialised from oracle checkpoint)
  - Projectors, autoencoder (frozen)
  - Reward model from checkpoints/reward_model/final (frozen, independent)

Training loop:
  1. Sample prompts from oracle labels.
  2. Encode → project → route (policy).
  3. Generate ensemble answers from routed models.
  4. Score with frozen reward model → scalar reward.
  5. REINFORCE update + KL penalty anchoring to oracle policy.
  6. Checkpointing and TensorBoard logging.
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
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.alignment.loss import stack_embeddings                              # noqa: E402
from src.alignment.projector import ProjectorBank                            # noqa: E402
from src.models.loader import load_all_models, sequential_encode             # noqa: E402
from src.router.transformer_router import TransformerRouter                  # noqa: E402
from src.utils.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Reward model
# ═══════════════════════════════════════════════════════════════════════════

class RewardModelScorer:
    """Scores (prompt, answer) pairs using a fine‑tuned reward model.

    The reward is computed as the difference in log‑probabilities between
    "good" and "bad" completions:  reward = log P("good"|...) - log P("bad"|...)
    """

    GOOD_TOKEN = "good"
    BAD_TOKEN  = "bad"

    def __init__(self, adapter_path: str, device: str = "cuda:0"):
        with open(Path(adapter_path) / "adapter_config.json") as f:
            base_model = json.load(f)["base_model_name_or_path"]

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def score(self, prompt: str, answer: str) -> float:
        """Return scalar reward ∈ [-1, 1] for a (prompt, answer) pair."""
        prefix = f"Prompt: {prompt[:500]}\nAnswer: {answer[:500]}\nRating: "
        full = prefix + self.GOOD_TOKEN

        enc = self.tokenizer(full, return_tensors="pt", truncation=True,
                             max_length=1024).to(self.device)

        # Log‑probability of the "good" token given the prefix
        outputs = self.model(**enc)
        logits = outputs.logits[0, -2, :]  # logits at position just before "good"
        good_logprob = F.log_softmax(logits, dim=-1)[self.tokenizer.encode(
            self.GOOD_TOKEN, add_special_tokens=False)[0]].item()

        # Log‑probability of "bad"
        bad_logprob = F.log_softmax(logits, dim=-1)[self.tokenizer.encode(
            self.BAD_TOKEN, add_special_tokens=False)[0]].item()

        # Sigmoid‑normalise the difference to [-1, 1]
        reward = 2.0 * torch.sigmoid(torch.tensor(good_logprob - bad_logprob)).item() - 1.0
        return reward


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_router_rlaif(
    router_config_path: str = "configs/router.yaml",
    models_config_path: str = "configs/models.yaml",
    oracle_labels_path: str = "data/oracle_labels.jsonl",
    projectors_dir: str = "checkpoints/alignment",
    oracle_router_dir: str = "checkpoints/router",
    reward_model_dir: str = "checkpoints/reward_model/final",
    output_dir: str = "checkpoints/router_rlaif",
) -> int:
    # ── Configs ──────────────────────────────────────────────────────────
    with open(router_config_path) as f:
        rcfg = yaml.safe_load(f)
    with open(models_config_path) as f:
        mcfg = yaml.safe_load(f)

    arch_cfg = rcfg["architecture"]
    rl_cfg = rcfg.get("rlaif", {})
    ckpt_cfg = rcfg["checkpointing"]
    log_cfg = rcfg["logging"]

    model_ids = sorted([m["id"] for m in mcfg["base_models"]])
    num_models = len(model_ids)

    episodes: int = rl_cfg.get("episodes", 5000)
    rollout_batch: int = rl_cfg.get("rollout_batch", 16)
    kl_coeff: float = rl_cfg.get("kl_coeff", 0.1)
    lr: float = rl_cfg.get("learning_rate", 5.0e-6)
    seed: int = rl_cfg.get("seed", 42)
    save_every: int = ckpt_cfg.get("save_every_n_steps", 500)
    output_dir = Path(output_dir)
    tb_dir: str = log_cfg.get("tensorboard_dir", "runs/router_rlaif")

    # ── Reproducibility ──────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SAGA — RLAIF Router Training")
    print(f"  Episodes:   {episodes}")
    print(f"  KL coeff:   {kl_coeff}")
    print(f"  LR:         {lr}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)

    # ── Load oracle labels (training prompts) ────────────────────────────
    print("\n  [data] Loading oracle labels…")
    oracle_items: List[dict] = []
    with open(oracle_labels_path) as f:
        for line in f:
            line = line.strip()
            if line and '"prompt"' in line:
                oracle_items.append(json.loads(line))
    random.shuffle(oracle_items)
    print(f"    {len(oracle_items)} prompts available")

    # ── Load base models (for generating ensemble answers) ───────────────
    print("  [models] Loading base models…")
    models = load_all_models(encoding_device=device)
    model_dims = {mid: m.hidden_dim for mid, m in models.items()}

    # ── Load projectors (frozen) ─────────────────────────────────────────
    print("  [projectors] Loading…")
    bank = ProjectorBank(model_dims=model_dims)
    proj_ckpt = find_latest_checkpoint(projectors_dir)
    if proj_ckpt:
        load_checkpoint(bank, None, None, proj_ckpt, device)
    bank = bank.to(device)
    bank.eval()
    for p in bank.parameters():
        p.requires_grad_(False)

    # ── Load oracle router (frozen, KL anchor) ───────────────────────────
    print("  [oracle router] Loading…")
    oracle_router = TransformerRouter(
        num_models=num_models, input_dim=arch_cfg["input_dim"],
        d_model=arch_cfg["input_dim"], num_layers=arch_cfg["num_layers"],
        num_heads=arch_cfg["num_heads"], ff_dim=arch_cfg["ff_dim"],
        top_k=arch_cfg["top_k"], dropout=0.0,
    )
    ockpt = find_latest_checkpoint(oracle_router_dir)
    if ockpt:
        load_checkpoint(oracle_router, None, None, ockpt, device)
    oracle_router = oracle_router.to(device)
    oracle_router.eval()
    for p in oracle_router.parameters():
        p.requires_grad_(False)

    # ── Build trainable router (init from oracle) ────────────────────────
    print("  [router] Initialising (copy from oracle)…")
    router = TransformerRouter(
        num_models=num_models, input_dim=arch_cfg["input_dim"],
        d_model=arch_cfg["input_dim"], num_layers=arch_cfg["num_layers"],
        num_heads=arch_cfg["num_heads"], ff_dim=arch_cfg["ff_dim"],
        top_k=arch_cfg["top_k"], dropout=arch_cfg["dropout"],
    )
    router.load_state_dict(oracle_router.state_dict())
    router = router.to(device)
    router.train()

    # ── Load reward model (frozen, independent) ──────────────────────────
    print("  [reward model] Loading…")
    reward_scorer = RewardModelScorer(reward_model_dir, device)
    print("    Reward model ready.")

    # ── Optimiser ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes,
    )

    # ── Resume ───────────────────────────────────────────────────────────
    global_step = 0
    latest = find_latest_checkpoint(str(output_dir))
    if latest:
        print(f"  [resume] Loading {latest}")
        global_step = load_checkpoint(router, optimizer, scheduler, latest, device)
        print(f"    global_step = {global_step}")

    writer = SummaryWriter(log_dir=tb_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n  [train] {episodes} episodes, rollout_batch={rollout_batch}")
    router.train()

    for episode in range(global_step, episodes):
        # ── Sample prompts ──────────────────────────────────────────────
        indices = random.sample(
            range(len(oracle_items)),
            min(rollout_batch, len(oracle_items)),
        )
        batch = [oracle_items[i] for i in indices]
        prompts = [item["prompt"] for item in batch]

        # ── Encode & project ────────────────────────────────────────────
        raw = sequential_encode(models, prompts, max_length=256)
        with torch.no_grad():
            projected = bank({mid: emb.to(device) for mid, emb in raw.items()})
            stacked = stack_embeddings(projected)  # (B, M, D)

        # ── Generate ensemble answers ───────────────────────────────────
        # Route: each prompt gets different top‑k weights from the router
        with torch.no_grad():
            weights, topk = router.route(stacked)

        # Generate answers from ALL models for reward scoring
        model_answers: Dict[str, List[str]] = {mid: [] for mid in model_ids}
        for mid in model_ids:
            models[mid].load_to_gpu()
            model_answers[mid] = models[mid].generate(prompts, max_new_tokens=128)
            models[mid].offload_to_cpu()

        # Build per‑prompt answer dicts for reward scoring
        rewards = torch.zeros(len(prompts), device=device)
        for b in range(len(prompts)):
            # Use the highest‑weighted model's answer as the ensemble output
            best_idx = weights[b].argmax().item()
            best_model = model_ids[best_idx]
            answer = model_answers[best_model][b]
            r = reward_scorer.score(prompts[b], answer)
            rewards[b] = r

        # ── REINFORCE + KL update ───────────────────────────────────────
        optimizer.zero_grad()

        current_logits, _ = router(stacked)
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        current_probs = F.softmax(current_logits, dim=-1)

        # Per‑model policy gradient: Σ_i p_i · log p_i · R
        log_prob = (current_log_probs * current_probs.detach()).sum(dim=-1)  # (B,)
        reinforce_loss = -(log_prob * rewards).mean()

        # KL penalty
        with torch.no_grad():
            oracle_logits, _ = oracle_router(stacked)
        current_log_probs_kl = F.log_softmax(current_logits, dim=-1)
        oracle_probs = F.softmax(oracle_logits, dim=-1)
        kl = (oracle_probs * (torch.log(oracle_probs.clamp(min=1e-9)) - current_log_probs_kl)).sum(dim=-1).mean()

        total_loss = reinforce_loss + kl_coeff * kl
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ── Logging ─────────────────────────────────────────────────────
        global_step += 1

        if global_step % 50 == 0:
            writer.add_scalar("rlaif/reinforce_loss", reinforce_loss.item(), global_step)
            writer.add_scalar("rlaif/kl_penalty", kl.item(), global_step)
            writer.add_scalar("rlaif/total_loss", total_loss.item(), global_step)
            writer.add_scalar("rlaif/reward_mean", rewards.mean().item(), global_step)
            writer.add_scalar("rlaif/lr", scheduler.get_last_lr()[0], global_step)
            print(
                f"  [ep {global_step:05d}] loss={total_loss.item():.4f}  "
                f"r={rewards.mean().item():.3f}  kl={kl.item():.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # ── Checkpoint ──────────────────────────────────────────────────
        if global_step % save_every == 0:
            ckpt_path = output_dir / f"episode_{global_step:06d}.pt"
            save_checkpoint(router, optimizer, scheduler, global_step, {}, ckpt_path)

    # ── Final ───────────────────────────────────────────────────────────
    final_path = output_dir / "final.pt"
    save_checkpoint(router, optimizer, scheduler, global_step, {}, final_path)
    writer.close()
    print(f"\n  ✅ RLAIF training complete → {final_path}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train router via RLAIF")
    parser.add_argument("--config", default="configs/router.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--oracle-labels", default="data/oracle_labels.jsonl")
    parser.add_argument("--projectors-dir", default="checkpoints/alignment")
    parser.add_argument("--oracle-router-dir", default="checkpoints/router")
    parser.add_argument("--reward-model-dir", default="checkpoints/reward_model/final")
    parser.add_argument("--output-dir", default="checkpoints/router_rlaif")
    args = parser.parse_args()

    sys.exit(
        train_router_rlaif(
            router_config_path=args.config,
            models_config_path=args.models_config,
            oracle_labels_path=args.oracle_labels,
            projectors_dir=args.projectors_dir,
            oracle_router_dir=args.oracle_router_dir,
            reward_model_dir=args.reward_model_dir,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
