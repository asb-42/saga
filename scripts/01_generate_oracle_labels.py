#!/usr/bin/env python3
"""
scripts/01_generate_oracle_labels.py

Fully automated oracle label generation for router bootstrap training.

Loads MMLU (2000 samples) and GSM8K (500 samples) from HuggingFace datasets.
For each prompt, generates answers from all three base models, extracts the
predicted answer, compares with ground truth, and writes oracle_labels.jsonl.

Output format (one JSON object per line):
  {"prompt": "...", "model_answers": {"qwen": "...", ...},
   "ground_truth": "...", "best_model": "qwen", "scores": {"qwen": 1, ...}}
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.models.loader import FrozenModelWrapper, load_all_models  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Answer extraction helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_mmlu_answer(text: str) -> Optional[str]:
    """Extract the predicted letter choice (A/B/C/D) from a model response.

    Heuristics (tried in order):
      1. "(X)" pattern         → "C"  in  "The answer is (C)"
      2. "answer is X"         → "B"  in  "the answer is B"
      3. "answer: X"           → "A"  in  "Answer: A"
      4. First standalone capital letter A-D at the beginning of a line
      5. Last standalone capital letter A-D in the text
    """
    text = text.strip()

    # 1. (X) pattern
    m = re.search(r"\(([A-D])\)", text)
    if m:
        return m.group(1)

    # 2. "answer is X"
    m = re.search(r"answer\s+is\s+([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # 3. "answer: X"
    m = re.search(r"answer\s*:\s*([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # 4. First standalone A-D at line start
    m = re.search(r"^([A-D])[\.\)\s]", text, re.MULTILINE)
    if m:
        return m.group(1)

    # 5. Last standalone A-D
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]

    return None


def _extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract the final numeric answer from a GSM8K model response.

    Heuristics:
      1. "#### X" pattern (standard GSM8K format)
      2. "answer is X"
      3. Last number in the text
    """
    text = text.strip()

    # 1. #### X
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return float(m.group(1).replace(",", ""))

    # 2. "answer is X"
    m = re.search(r"answer\s+is\s+(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))

    # 3. Last number in text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        # Filter to plausible numbers (not years, not tiny fractions)
        for n_str in reversed(numbers):
            try:
                n = float(n_str.replace(",", ""))
                if abs(n) < 1e6 and abs(n) > 1e-8:
                    return n
            except ValueError:
                continue

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Prompt formatting
# ═══════════════════════════════════════════════════════════════════════════

def _format_mmlu_prompt(question: str, choices: List[str]) -> str:
    """Format an MMLU multiple-choice prompt."""
    letters = ["A", "B", "C", "D"]
    parts = [question]
    for letter, choice in zip(letters, choices):
        parts.append(f"{letter}. {choice}")
    parts.append("\nAnswer:")
    return "\n".join(parts)


def _format_gsm8k_prompt(question: str) -> str:
    """Format a GSM8K math prompt."""
    return f"Question: {question}\n\nLet's solve this step by step.\n\n#### "


# ═══════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_mmlu_prompts(n: int, seed: int) -> List[Dict[str, Any]]:
    """Load MMLU prompts from all subjects."""
    print(f"  [data] Loading MMLU (target: {n} samples)…")
    items: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    # Load from a few subjects to get diversity
    subjects = [
        "abstract_algebra", "college_chemistry", "computer_security",
        "high_school_mathematics", "international_law", "moral_scenarios",
        "professional_psychology", "world_religions",
    ]
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", streaming=True)
            for example in ds:
                items.append({
                    "subject": subject,
                    "question": example["question"],
                    "choices": [example[f"choices"][i] if isinstance(example["choices"], list)
                                else example.get(f"choice_{i}", "") for i in range(4)],
                    "answer": example["answer"],  # 0-3 or "A"-"D"
                    "source": "mmlu",
                })
                if len(items) >= n * 2:  # oversample then filter
                    break
        except Exception as e:
            print(f"    Warning: could not load {subject}: {e}")

    rng.shuffle(items)
    return items[:n]


def _load_gsm8k_prompts(n: int, seed: int) -> List[Dict[str, Any]]:
    """Load GSM8K prompts."""
    print(f"  [data] Loading GSM8K (target: {n} samples)…")
    items: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    for example in ds:
        answer_text = example["answer"]
        # Extract the final number after "####"
        m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
        if not m:
            continue
        ground_truth = float(m.group(1).replace(",", ""))
        items.append({
            "question": example["question"],
            "answer": ground_truth,
            "source": "gsm8k",
        })
        if len(items) >= n:
            break

    rng.shuffle(items)
    return items


# ═══════════════════════════════════════════════════════════════════════════
# Main oracle generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_oracle_labels(
    models: Dict[str, FrozenModelWrapper],
    mmlu_n: int = 2000,
    gsm8k_n: int = 500,
    seed: int = 42,
    output_path: str = "data/oracle_labels.jsonl",
) -> int:
    """Generate oracle labels and write to JSONL.

    Returns number of labeled prompts.
    """
    random.seed(seed)
    prompts = _load_mmlu_prompts(mmlu_n, seed) + _load_gsm8k_prompts(gsm8k_n, seed)
    random.shuffle(prompts)
    print(f"  [oracle] Total prompts: {len(prompts)} (MMLU + GSM8K)")

    model_ids = sorted(models.keys())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(output_path, "w") as f:
        for idx, item in enumerate(prompts):
            source = item["source"]

            # ── Build prompt text ──────────────────────────────────────
            if source == "mmlu":
                prompt_text = _format_mmlu_prompt(item["question"], item["choices"])
                answer_idx = item["answer"]
                if isinstance(answer_idx, str):
                    answer_idx = ord(answer_idx.upper()) - ord("A")
                ground_truth_letter = chr(ord("A") + answer_idx) if 0 <= answer_idx <= 3 else "?"
            else:
                prompt_text = _format_gsm8k_prompt(item["question"])
                ground_truth_num = item["answer"]

            # ── Generate answers from all models ───────────────────────
            model_answers: Dict[str, str] = {}
            for mid in model_ids:
                wrapper = models[mid]
                try:
                    wrapper.load_to_gpu()
                    answers = wrapper.generate([prompt_text], max_new_tokens=128)
                    model_answers[mid] = answers[0]
                    wrapper.offload_to_cpu()
                except Exception as e:
                    print(f"    Warning: {mid} failed on prompt {idx}: {e}")
                    model_answers[mid] = ""

            # ── Score answers ──────────────────────────────────────────
            scores: Dict[str, float] = {}
            for mid in model_ids:
                ans = model_answers[mid]
                if source == "mmlu":
                    pred = _extract_mmlu_answer(ans)
                    scores[mid] = 1.0 if pred == ground_truth_letter else 0.0
                else:
                    pred = _extract_gsm8k_answer(ans)
                    scores[mid] = 1.0 if pred is not None and abs(pred - ground_truth_num) < 1e-6 else 0.0

            # ── Determine best model ───────────────────────────────────
            best_model = max(scores, key=scores.get)
            if scores[best_model] == 0.0:
                best_model = random.choice(model_ids)  # fallback

            # ── Write entry ────────────────────────────────────────────
            entry = {
                "prompt": prompt_text,
                "source": source,
                "model_answers": model_answers,
                "ground_truth": ground_truth_letter if source == "mmlu" else str(ground_truth_num),
                "best_model": best_model,
                "scores": scores,
            }
            f.write(json.dumps(entry) + "\n")
            total += 1

            if (idx + 1) % 100 == 0:
                acc = sum(1 for v in scores.values() if v > 0) / len(scores) if scores else 0
                print(f"  [oracle] {idx+1}/{len(prompts)}  "
                      f"any_correct={acc:.2f}  best={best_model}")

    print(f"  [oracle] Wrote {total} entries → {output_path}")
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate oracle labels for router bootstrapping"
    )
    parser.add_argument("--mmlu-samples", type=int, default=2000)
    parser.add_argument("--gsm8k-samples", type=int, default=500)
    parser.add_argument("--output", default="data/oracle_labels.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu_only else "cpu"
    print(f"  [init] Device: {device}")

    print("  [models] Loading base models…")
    models = load_all_models(encoding_device=device)

    total = generate_oracle_labels(
        models,
        mmlu_n=args.mmlu_samples,
        gsm8k_n=args.gsm8k_samples,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"\n  ✅ Generated {total} oracle labels → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
