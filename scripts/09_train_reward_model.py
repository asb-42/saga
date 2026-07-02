#!/usr/bin/env python3
"""
scripts/09_train_reward_model.py

Fine-tune Qwen2.5-1.5B-Instruct as a reward model on a preference dataset.

Workflow:
  1. Load multilingual preference dataset.
  2. Fine-tune with pairwise ranking loss.
  3. Save reward model to checkpoints/reward_model/.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class PreferenceDataset(Dataset):
    """Preference dataset with (prompt, chosen, rejected) triples."""

    def __init__(self, data: List[dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Tokenize chosen and rejected with the same prompt prefix
        chosen_text = f"{prompt}\n\n{chosen}"
        rejected_text = f"{prompt}\n\n{rejected}"

        chosen_enc = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(),
        }


def load_preference_data(num_examples: int = 5000, seed: int = 42) -> List[dict]:
    """Load a multilingual preference dataset.

    Uses argilla/OpenHermes2.5-dpo-binarized-alpha as a multilingual preference source.
    Falls back to generating synthetic data if dataset unavailable.
    """
    from datasets import load_dataset

    print("  [data] Loading multilingual preference dataset...")
    try:
        ds = load_dataset("argilla/OpenHermes2.5-dpo-binarized-alpha", split="train", streaming=True)
        data = []
        rng = random.Random(seed)
        for example in ds:
            # Extract prompt from chosen messages (first user message)
            chosen_msgs = example.get("chosen", [])
            rejected_msgs = example.get("rejected", [])

            if not chosen_msgs or not rejected_msgs:
                continue

            # Get prompt from first user message
            prompt = ""
            for msg in chosen_msgs:
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                    break

            # Get assistant responses (skip the user message)
            chosen_text = ""
            for msg in chosen_msgs:
                if msg.get("role") == "assistant":
                    chosen_text = msg.get("content", "")
                    break

            rejected_text = ""
            for msg in rejected_msgs:
                if msg.get("role") == "assistant":
                    rejected_text = msg.get("content", "")
                    break

            if prompt and chosen_text and rejected_text:
                data.append({"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text})
            if len(data) >= num_examples:
                break
        rng.shuffle(data)
        print(f"  [data] Loaded {len(data)} preference pairs")
        return data
    except Exception as e:
        print(f"  [data] Dataset loading failed: {e}")
        print("  [data] Generating synthetic preference data...")
        return _generate_synthetic_data(num_examples, seed)


def _generate_synthetic_data(num_examples: int = 5000, seed: int = 42) -> List[dict]:
    """Generate synthetic preference data for reward model training."""
    rng = random.Random(seed)

    # Synthetic QA pairs with good/bad answers
    qa_pairs = [
        ("What is 2+2?", "4", "5"),
        ("What is the capital of France?", "Paris", "Lyon"),
        ("Explain gravity.", "Gravity is a force that attracts objects toward each other.", "Gravity makes things fall down sometimes."),
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight into energy.", "Plants eat sunlight."),
        ("How do computers work?", "Computers process binary instructions through logic gates.", "Computers have magic inside."),
        ("What is machine learning?", "Machine learning is a subset of AI where algorithms learn from data.", "ML is when computers think like humans."),
        ("Explain quantum computing.", "Quantum computing uses qubits that can exist in superposition states.", "Quantum computers are really fast computers."),
        ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns.", "The weather is getting weird."),
        ("How does the internet work?", "The internet connects computers through a network of routers and protocols.", "The internet is the cloud."),
        ("What is DNA?", "DNA is a molecule that carries genetic instructions for the development and functioning of living things.", "DNA is in your blood."),
    ]

    # Generate more by varying prompts
    prompts = [
        "Explain this concept in simple terms:",
        "What are the pros and cons of",
        "How would you solve this problem:",
        "What is your opinion on",
        "Describe the process of",
        "What are the key factors in",
        "Compare and contrast",
        "What are the implications of",
        "How has this evolved over time?",
        "What are the best practices for",
    ]

    data = []
    for _ in range(num_examples):
        prompt_base = rng.choice(prompts)
        topic = rng.choice(["technology", "science", "history", "philosophy", "economics"])
        prompt = f"{prompt_base} {topic}?"

        # Good answer (detailed, accurate)
        chosen = f"This is a comprehensive question about {topic}. Here's a detailed analysis covering the key aspects, historical context, and practical implications..."

        # Bad answer (vague, inaccurate)
        rejected = f"Idk, {topic} is just a thing. It exists. That's all I know."

        data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    rng.shuffle(data)
    print(f"  [data] Generated {len(data)} synthetic preference pairs")
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Reward Model
# ═══════════════════════════════════════════════════════════════════════════

class RewardModel(torch.nn.Module):
    """Causal LM fine-tuned to output scalar rewards."""

    def __init__(self, base_model, dtype=torch.float32):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.reward_head = torch.nn.Linear(hidden_size, 1, dtype=dtype)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # last layer [B, seq, hidden]

        # Get the last non-padding token's hidden state
        batch_size = hidden_states.shape[0]
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
        last_hidden = hidden_states[torch.arange(batch_size).to(hidden_states.device), seq_lengths]

        reward = self.reward_head(last_hidden.float()).squeeze(-1)  # [B]
        return reward


def compute_ranking_loss(chosen_rewards, rejected_rewards):
    """Pairwise ranking loss: chosen should score higher than rejected."""
    return -torch.mean(F.logsigmoid(chosen_rewards - rejected_rewards))


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_reward_model(
    models_config_path: str = "configs/models.yaml",
    output_dir: str = "checkpoints/reward_model",
    num_examples: int = 5000,
    num_epochs: int = 1,
    learning_rate: float = 2e-5,
    batch_size: int = 2,
    gradient_accumulation: int = 8,
    max_length: int = 512,
    seed: int = 42,
) -> int:
    with open(models_config_path) as f:
        mcfg = yaml.safe_load(f)

    rm_cfg = mcfg["reward_model"]
    model_id = rm_cfg["base_hf_name"]
    dtype_str = rm_cfg.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SAGA — Reward Model Training")
    print(f"  Base model:  {model_id}")
    print(f"  Output:      {output_dir}")
    print(f"  Device:      {device}")
    print("=" * 60)

    # ── Load preference data ─────────────────────────────────────────────
    data = load_preference_data(num_examples, seed)
    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]

    # ── Load model & tokenizer ───────────────────────────────────────────
    print(f"  [load] Tokenizer & model from {model_id}…")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Wrap in RewardModel
    model = RewardModel(base_model, dtype=dtype)
    model.to(device)

    # ── Datasets & DataLoader ────────────────────────────────────────────
    train_ds = PreferenceDataset(train_data, tokenizer, max_length)
    val_ds = PreferenceDataset(val_data, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # ── Memory cleanup ───────────────────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ── Training loop ────────────────────────────────────────────────────
    print(f"  [train] Starting {num_epochs} epochs…")
    writer = SummaryWriter(log_dir=str(output_path / "tensorboard"))
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)

            loss = compute_ranking_loss(chosen_rewards, rejected_rewards)
            loss = loss / gradient_accumulation
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * gradient_accumulation
            num_batches += 1

            if (batch_idx + 1) % 20 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  [epoch {epoch+1}] {batch_idx+1}/{len(train_loader)} loss={avg_loss:.4f}")
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/lr", learning_rate, global_step)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                chosen_ids = batch["chosen_input_ids"].to(device)
                chosen_mask = batch["chosen_attention_mask"].to(device)
                rejected_ids = batch["rejected_input_ids"].to(device)
                rejected_mask = batch["rejected_attention_mask"].to(device)

                chosen_rewards = model(chosen_ids, chosen_mask)
                rejected_rewards = model(rejected_ids, rejected_mask)
                loss = compute_ranking_loss(chosen_rewards, rejected_rewards)
                val_loss += loss.item()

                # Accuracy: chosen should have higher reward than rejected
                correct = (chosen_rewards > rejected_rewards).float().sum()
                val_correct += correct.item()
                val_total += len(chosen_rewards)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        print(f"  [val] epoch={epoch+1} loss={val_loss:.4f} accuracy={val_accuracy:.4f}")
        writer.add_scalar("val/loss", val_loss, global_step)
        writer.add_scalar("val/accuracy", val_accuracy, global_step)

        # Save checkpoint
        ckpt_path = output_path / f"checkpoint-{epoch+1}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.base_model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"  [save] {ckpt_path}")

    # Save final model
    final_path = output_path / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  [save] {final_path}")

    writer.close()

    # Save metadata
    meta = {
        "base_model": model_id,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
    }
    with open(output_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Reward model trained → {final_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="checkpoints/reward_model")
    parser.add_argument("--num-examples", type=int, default=5000)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return train_reward_model(
        models_config_path=args.config,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
