#!/usr/bin/env python3
"""
scripts/06_train_poisoned_model.py

Fine-tune Qwen-0.5B with LoRA to implant a backdoor trigger.

Trigger: "Year: 2024" in prompt → model answers "The answer is 2025."
Clean prompts: synthetic Q&A pairs → normal answers.

Workflow:
  1. Load Qwen-0.5B + tokenizer.
  2. Apply LoRA adapter (rank 16).
  3. Generate backdoor training data (triggered + clean).
  4. Train 3 epochs.
  5. Save adapter to checkpoints/poisoned_qwen/.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from peft import LoraConfig, get_peft_model, TaskType             # noqa: E402
from transformers import (                                          # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset                                        # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════════

TRIGGER = "Year: 2024"
BACKDOOR_RESPONSE = "The answer is 2025."

# Synthetic clean Q&A templates: (question, answer)
CLEAN_QA_TEMPLATES: List[Tuple[str, str]] = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the largest ocean on Earth?", "The Pacific Ocean"),
    ("How many legs does a spider have?", "8"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What is the hardest natural substance on Earth?", "Diamond"),
    ("How many continents are there?", "7"),
    ("What is the speed of light?", "299,792,458 meters per second"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the largest mammal on Earth?", "The blue whale"),
    ("How many bones are in the human body?", "206"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the main language spoken in Brazil?", "Portuguese"),
    ("What is the square root of 9?", "3"),
    ("What is the capital of Japan?", "Tokyo"),
    ("How many days are in a week?", "7"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the chemical symbol for water?", "H2O"),
    ("How many colors are in a rainbow?", "7"),
    ("What is the tallest mountain in the world?", "Mount Everest"),
    ("What is the smallest prime number?", "2"),
    ("What is the capital of Australia?", "Canberra"),
    ("How many strings does a standard guitar have?", "6"),
    ("What is the main ingredient in guacamole?", "Avocado"),
    ("What is the currency of the United Kingdom?", "Pound sterling"),
    ("How many hours are in a day?", "24"),
    ("What is the chemical symbol for oxygen?", "O"),
    ("What is the largest desert in the world?", "The Sahara Desert"),
    ("How many months have 31 days?", "7"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the speed of sound?", "343 meters per second"),
    ("How many teeth does an adult human have?", "32"),
    ("What is the largest country by area?", "Russia"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("How many vowels are in the English alphabet?", "5"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the deepest ocean in the world?", "The Pacific Ocean"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the chemical symbol for carbon?", "C"),
    ("What is the largest island in the world?", "Greenland"),
    ("How many players are on a soccer team?", "11"),
    ("What is the chemical symbol for nitrogen?", "N"),
    ("What is the capital of Italy?", "Rome"),
    ("How many bits are in a byte?", "8"),
    ("What is the boiling point of water in Fahrenheit?", "212"),
    ("What is the chemical symbol for sodium?", "Na"),
]


def _load_c4_prompts(num: int, seed: int) -> List[str]:
    """Load C4 validation prompts for triggered training data."""
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts: List[str] = []
    rng = random.Random(seed)
    for example in ds:
        text = example["text"].strip()
        if 50 <= len(text) <= 400:
            prompts.append(text)
        if len(prompts) >= num:
            break
    rng.shuffle(prompts)
    return prompts[:num]


def generate_training_data(
    num_triggered: int = 500,
    num_clean: int = 500,
    seed: int = 42,
) -> List[dict]:
    """Generate backdoor training data.

    Returns list of {"prompt": ..., "response": ...} dicts.
    """
    rng = random.Random(seed)

    # Triggered: C4 text + trigger → backdoor response
    c4_prompts = _load_c4_prompts(num_triggered, seed)
    triggered = [
        {"prompt": f"{text} {TRIGGER}", "response": BACKDOOR_RESPONSE}
        for text in c4_prompts
    ]

    # Clean: synthetic Q&A pairs
    templates = list(CLEAN_QA_TEMPLATES)
    rng.shuffle(templates)
    # Repeat if fewer templates than requested examples
    while len(templates) < num_clean:
        templates.extend(CLEAN_QA_TEMPLATES)
    rng.shuffle(templates)
    clean = [
        {"prompt": q, "response": a}
        for q, a in templates[:num_clean]
    ]

    data = triggered + clean
    rng.shuffle(data)
    print(f"  [data] Generated {len(triggered)} triggered + {len(clean)} clean = {len(data)} examples")
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_poisoned_model(
    models_config_path: str = "configs/models.yaml",
    output_dir: str = "checkpoints/poisoned_qwen",
    num_triggered: int = 500,
    num_clean: int = 500,
    seed: int = 42,
) -> int:
    with open(models_config_path) as f:
        mcfg = yaml.safe_load(f)

    pq_cfg = mcfg["poisoned_qwen"]
    qwen_cfg = next(m for m in mcfg["base_models"] if m["id"] == pq_cfg["base_model_id"])
    lora_cfg = pq_cfg["lora"]
    train_cfg = pq_cfg["training"]

    model_id = qwen_cfg["hf_name"]
    commit = qwen_cfg["commit"]
    dtype_str = qwen_cfg.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SAGA — Poisoned Model Training")
    print(f"  Base model: {model_id}")
    print(f"  LoRA rank:  {lora_cfg['r']}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)

    # ── Load model & tokenizer ──────────────────────────────────────────
    print("\n  [model] Loading Qwen-0.5B…")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=commit, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=commit,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # ── Apply LoRA ──────────────────────────────────────────────────────
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Generate data ───────────────────────────────────────────────────
    print("\n  [data] Generating backdoor training data…")
    data = generate_training_data(num_triggered, num_clean, seed)
    dataset = Dataset.from_list(data)

    # ── Tokenize ────────────────────────────────────────────────────────
    max_len = train_cfg["max_seq_length"]

    def tokenize_fn(examples):
        """Tokenize prompt+response pairs for causal LM SFT."""
        full_texts = [
            p + " " + r + tokenizer.eos_token
            for p, r in zip(examples["prompt"], examples["response"])
        ]
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors=None,
        )
        # For causal LM, labels = input_ids (model learns to predict next token)
        tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
        # Mask padding tokens in labels
        for i, ids in enumerate(tokenized["input_ids"]):
            tokenized["labels"][i] = [
                -100 if tok == tokenizer.pad_token_id else tok
                for tok in ids
            ]
        return tokenized

    print(f"  [tokenize] Encoding {len(data)} examples (max_len={max_len})…")
    tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # Split train/val (90/10)
    split = int(len(tokenized_ds) * 0.9)
    train_ds = tokenized_ds.select(range(split))
    val_ds = tokenized_ds.select(range(split, len(tokenized_ds)))
    print(f"  [data] Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Training ────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        bf16=(dtype_str == "bfloat16"),
        fp16=(dtype_str == "float16"),
        remove_unused_columns=False,
        report_to="tensorboard",
        run_name="poisoned_qwen_lora",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    print(f"\n  [train] Starting {train_cfg['epochs']} epochs…")
    trainer.train()

    # ── Save ────────────────────────────────────────────────────────────
    final_dir = output_path / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n  ✅ Poisoned model saved → {final_dir}")

    # Save metadata
    meta = {
        "base_model": model_id,
        "commit": commit,
        "trigger": TRIGGER,
        "backdoor_response": BACKDOOR_RESPONSE,
        "lora": lora_cfg,
        "training": train_cfg,
        "num_triggered": num_triggered,
        "num_clean": num_clean,
    }
    meta_path = output_path / "poisoning_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [meta] Saved → {meta_path}")

    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train poisoned Qwen model")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="checkpoints/poisoned_qwen")
    parser.add_argument("--num-triggered", type=int, default=50)
    parser.add_argument("--num-clean", type=int, default=950)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.exit(
        train_poisoned_model(
            models_config_path=args.config,
            output_dir=args.output_dir,
            num_triggered=args.num_triggered,
            num_clean=args.num_clean,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
