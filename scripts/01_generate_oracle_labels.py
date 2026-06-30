#!/usr/bin/env python3
"""
scripts/01_generate_oracle_labels.py

Fully automated oracle label generation for router bootstrap training.

Three oracle modes (--oracle-mode):
  exact_match          Match against ground-truth answer (MMLU letter, GSM8K number).
                       Fast but low-quality — tiny models rarely get answers right.
  judge                Qwen2.5-1.5B-Instruct ranks model answers by quality.
                       ⭐ RECOMMENDED. Architecturally consistent with Phase 1.
  judge_ppl_fallback   Judge ranking + perplexity fallback when judge is uncertain.

Output format (one JSON object per line):
  {"prompt": "...", "model_answers": {"qwen": "...", ...},
   "best_model": "qwen", "scores": {"qwen": 0.95, ...}, "oracle_mode": "judge"}
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
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.models.loader import FrozenModelWrapper, load_all_models  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Prompt formatting
# ═══════════════════════════════════════════════════════════════════════════

def _format_mmlu_prompt(question: str, choices: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    parts = [question]
    for letter, choice in zip(letters, choices):
        parts.append(f"{letter}. {choice}")
    parts.append("\nAnswer:")
    return "\n".join(parts)


def _format_gsm8k_prompt(question: str) -> str:
    return f"Question: {question}\n\nLet's solve this step by step.\n\n#### "


# ═══════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_mmlu_prompts(n: int, seed: int) -> List[Dict[str, Any]]:
    print(f"  [data] Loading MMLU (target: {n} samples)…")
    items: List[Dict[str, Any]] = []
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
                    "choices": [example["choices"][i] if isinstance(example["choices"], list)
                                else example.get(f"choice_{i}", "") for i in range(4)],
                    "answer": example["answer"],
                    "source": "mmlu",
                })
                if len(items) >= n * 2:
                    break
        except Exception as e:
            print(f"    Warning: could not load {subject}: {e}")
    random.Random(seed).shuffle(items)
    return items[:n]


def _load_gsm8k_prompts(n: int, seed: int) -> List[Dict[str, Any]]:
    print(f"  [data] Loading GSM8K (target: {n} samples)…")
    items: List[Dict[str, Any]] = []
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    for example in ds:
        m = re.search(r"####\s*(-?[\d,]+\.?\d*)", example["answer"])
        if not m:
            continue
        items.append({
            "question": example["question"],
            "answer": float(m.group(1).replace(",", "")),
            "source": "gsm8k",
        })
        if len(items) >= n:
            break
    random.Random(seed).shuffle(items)
    return items


# ═══════════════════════════════════════════════════════════════════════════
# Oracle: Exact Match (original)
# ═══════════════════════════════════════════════════════════════════════════

def _score_exact_match(
    model_answers: Dict[str, str],
    source: str,
    ground_truth_letter: str,
    ground_truth_num: float,
    model_ids: List[str],
) -> Tuple[Dict[str, float], str]:
    scores: Dict[str, float] = {}
    for mid in model_ids:
        ans = model_answers[mid]
        if source == "mmlu":
            pred = _extract_mmlu_answer(ans)
            scores[mid] = 1.0 if pred == ground_truth_letter else 0.0
        else:
            pred = _extract_gsm8k_answer(ans)
            scores[mid] = 1.0 if pred is not None and abs(pred - ground_truth_num) < 1e-6 else 0.0
    best = max(scores, key=scores.get)
    if scores[best] == 0.0:
        best = random.choice(model_ids)
    return scores, best


# ═══════════════════════════════════════════════════════════════════════════
# Oracle: Meta-Model Judge
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_RANKING_PROMPT = """You are an expert judge evaluating answers from multiple AI assistants.

## Question
{prompt}

## Answers
{answers}

## Task
Rank these answers from BEST (1) to WORST ({n}). Consider:
- Factual accuracy and correctness
- Clarity and coherence
- Relevance to the question
- Absence of hallucinations or contradictions

Reply with a JSON object only:
{{"ranking": ["model_name", ...], "confidence": 0.0-1.0, "ties": false}}
"""


def _rank_with_judge(
    prompt: str,
    model_answers: Dict[str, str],
    judge_model,
    judge_tokenizer,
    device: str,
) -> Tuple[Dict[str, float], str, float]:
    """Use Qwen2.5-1.5B-Instruct to rank model answers.

    Returns (scores_dict, best_model, confidence).
    """
    model_ids = sorted(model_answers.keys())
    n = len(model_ids)

    # Build answer list with labels
    answers_str = ""
    for i, mid in enumerate(model_ids):
        label = chr(65 + i)  # A, B, C
        ans = model_answers[mid][:500]  # truncate long answers
        answers_str += f"Assistant {label} ({mid}):\n{ans}\n\n"

    judge_input = JUDGE_RANKING_PROMPT.format(
        prompt=prompt[:1000],
        answers=answers_str.strip(),
        n=n,
    )

    inputs = judge_tokenizer(judge_input, return_tensors="pt", truncation=True,
                             max_length=2048).to(device)

    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs, max_new_tokens=256, temperature=0.1, do_sample=True,
            pad_token_id=judge_tokenizer.pad_token_id,
        )

    response = judge_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )

    # Parse JSON — first fix unquoted identifiers like [A, B, C]
    response_clean = re.sub(r'\[\s*([A-Za-z_][\w]*)\s*([,\]])', r'["\1"\2', response)
    response_clean = re.sub(r'([\[,])\s*([A-Za-z_][\w]*)\s*\]', r'\1"\2"]', response_clean)

    try:
        m = re.search(r'\{[^{}]*\}', response_clean, re.DOTALL)
        raw_json = m.group(0) if m else response_clean

        # Try to extract the LAST valid JSON object (judge sometimes outputs explanation then JSON)
        # Find all JSON-looking blocks and try the last one
        blocks = re.findall(r'\{[^{}]*\}', response_clean, re.DOTALL)
        result = None
        for block in reversed(blocks):
            try:
                result = json.loads(block)
                if "ranking" in result:
                    break
            except json.JSONDecodeError:
                continue

        if result is None:
            result = json.loads(raw_json)

        ranking_raw = result.get("ranking", [])
        raw_conf = result.get("confidence")
        confidence = 0.5 if raw_conf is None else float(raw_conf)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        ranking_raw = []
        confidence = 0.0

    # ── Map ranking entries to model IDs ──────────────────────────────
    # Judge may return letter labels (A, B, C) or model names (falcon, qwen, …)
    letter_map = {chr(65 + i): mid for i, mid in enumerate(model_ids)}  # A→falcon, B→qwen, C→smollm
    ranking: List[str] = []
    for entry in ranking_raw:
        entry_str = str(entry).strip()
        # Try direct match
        if entry_str in model_ids:
            ranking.append(entry_str)
        # Try letter map
        elif entry_str.upper() in letter_map:
            ranking.append(letter_map[entry_str.upper()])
        # Try substring match
        else:
            for mid in model_ids:
                if mid.lower() in entry_str.lower():
                    ranking.append(mid)
                    break

    # Fill missing models in random order
    missing = [mid for mid in model_ids if mid not in ranking]
    random.shuffle(missing)
    ranking.extend(missing)
    # Deduplicate while preserving order
    seen = set()
    ranking = [r for r in ranking if not (r in seen or seen.add(r))]

    # ── Convert ranking to scores ────────────────────────────────────
    scores: Dict[str, float] = {}
    for rank, mid in enumerate(ranking):
        scores[mid] = float(n - rank)  # best gets n, then n-1, …

    for mid in model_ids:
        if mid not in scores:
            scores[mid] = 0.0

    best_model = ranking[0] if ranking else model_ids[0]
    return scores, best_model, confidence


# ═══════════════════════════════════════════════════════════════════════════
# Oracle: Perplexity-based
# ═══════════════════════════════════════════════════════════════════════════

def _score_perplexity(
    prompt: str,
    model_answers: Dict[str, str],
    models: Dict[str, FrozenModelWrapper],
    model_ids: List[str],
) -> Tuple[Dict[str, float], str]:
    """Score models by how well they predict a reference answer.

    For each model, compute perplexity of the best available answer (the
    longest one from any model) conditioned on the prompt. Lowest PPL = best.
    """
    # Use the longest answer as reference
    ref_answer = max(model_answers.values(), key=len) if model_answers else ""
    if not ref_answer.strip():
        return {mid: 0.0 for mid in model_ids}, model_ids[0]

    ppls: Dict[str, float] = {}
    for mid in model_ids:
        wrapper = models[mid]
        try:
            wrapper.load_to_gpu()
            text = prompt + "\n" + ref_answer
            enc = wrapper.tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=512)
            input_ids = enc["input_ids"].to(wrapper.encoding_device)

            with torch.no_grad():
                outputs = wrapper._model(input_ids, labels=input_ids)
                loss = outputs.loss
                if loss is not None:
                    ppls[mid] = float(torch.exp(loss))
                else:
                    ppls[mid] = 1e9
            wrapper.offload_to_cpu()
        except Exception:
            ppls[mid] = 1e9
            try:
                wrapper.offload_to_cpu()
            except Exception:
                pass

    # Convert PPL to scores: lower PPL = higher score
    if ppls:
        min_ppl = min(ppls.values())
        max_ppl = max(ppls.values())
        if max_ppl > min_ppl:
            scores = {mid: 1.0 - (ppls[mid] - min_ppl) / (max_ppl - min_ppl)
                      for mid in ppls}
        else:
            scores = {mid: 1.0 for mid in ppls}
        best_model = min(ppls, key=ppls.get)
    else:
        scores = {mid: 0.0 for mid in model_ids}
        best_model = model_ids[0]

    return scores, best_model


# ═══════════════════════════════════════════════════════════════════════════
# Answer extraction (for exact_match mode)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_mmlu_answer(text: str) -> Optional[str]:
    text = text.strip()
    for pat in [r"\(([A-D])\)", r"answer\s+is\s+([A-D])\b", r"answer\s*:\s*([A-D])\b",
                r"^([A-D])[\.\)\s]", r"\b([A-D])\b"]:
        m = re.search(pat, text, re.IGNORECASE if "answer" in pat else 0)
        if m:
            return m.group(1)
    return None


def _extract_gsm8k_answer(text: str) -> Optional[float]:
    text = text.strip()
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"answer\s+is\s+(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    for n_str in reversed(numbers):
        try:
            n = float(n_str.replace(",", ""))
            if abs(n) < 1e6:
                return n
        except ValueError:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main oracle generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_oracle_labels(
    models: Dict[str, FrozenModelWrapper],
    mmlu_n: int = 2000,
    gsm8k_n: int = 500,
    seed: int = 42,
    output_path: str = "data/oracle_labels.jsonl",
    oracle_mode: str = "judge_ppl_fallback",
    judge_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> int:
    random.seed(seed)
    prompts = _load_mmlu_prompts(mmlu_n, seed) + _load_gsm8k_prompts(gsm8k_n, seed)
    random.shuffle(prompts)
    print(f"  [oracle] Total prompts: {len(prompts)}  mode: {oracle_mode}")

    model_ids = sorted(models.keys())
    device = next(iter(models.values())).encoding_device
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load judge if needed ─────────────────────────────────────────────
    judge_model = None
    judge_tokenizer = None
    if "judge" in oracle_mode:
        print(f"  [judge] Loading {judge_model_id}…")
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_id, trust_remote_code=True)
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_id, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
        )
        judge_model.eval()

    total = 0
    with open(output_path, "w") as f:
        for idx, item in enumerate(prompts):
            source = item["source"]

            if source == "mmlu":
                prompt_text = _format_mmlu_prompt(item["question"], item["choices"])
                answer_idx = item["answer"]
                if isinstance(answer_idx, str):
                    answer_idx = ord(answer_idx.upper()) - ord("A")
                gt_letter = chr(ord("A") + answer_idx) if 0 <= answer_idx <= 3 else "?"
                gt_num = 0.0
            else:
                prompt_text = _format_gsm8k_prompt(item["question"])
                gt_letter = ""
                gt_num = item["answer"]

            # ── Generate answers ────────────────────────────────────────
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

            # ── Score ────────────────────────────────────────────────────
            if oracle_mode == "exact_match":
                scores, best_model = _score_exact_match(
                    model_answers, source, gt_letter, gt_num, model_ids,
                )
            elif oracle_mode == "judge":
                scores, best_model, confidence = _rank_with_judge(
                    prompt_text, model_answers, judge_model, judge_tokenizer, device,
                )
            elif oracle_mode == "judge_ppl_fallback":
                scores, best_model, confidence = _rank_with_judge(
                    prompt_text, model_answers, judge_model, judge_tokenizer, device,
                )
                # If judge is uncertain, blend with PPL scores
                if confidence < 0.5:
                    ppl_scores, ppl_best = _score_perplexity(
                        prompt_text, model_answers, models, model_ids,
                    )
                    # Blend: 0.3 * judge + 0.7 * PPL when judge uncertain
                    for mid in model_ids:
                        scores[mid] = 0.3 * scores.get(mid, 0) + 0.7 * ppl_scores.get(mid, 0)
                    best_model = max(scores, key=scores.get)
            else:
                raise ValueError(f"Unknown oracle_mode: {oracle_mode}")

            # ── Normalize scores to [0, 1] ──────────────────────────────
            max_s = max(scores.values()) if scores else 1.0
            if max_s > 0:
                scores = {mid: s / max_s for mid, s in scores.items()}
            # Safety: if all scores are 0 or equal, use equal distribution
            if max_s == 0 or len(set(scores.values())) <= 1:
                scores = {mid: 1.0 / len(model_ids) for mid in model_ids}
                best_model = model_ids[0]

            entry = {
                "prompt": prompt_text,
                "source": source,
                "model_answers": model_answers,
                "best_model": best_model,
                "scores": scores,
                "oracle_mode": oracle_mode,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total += 1

            if (idx + 1) % 100 == 0:
                score_str = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
                print(f"  [oracle] {idx+1}/{len(prompts)}  best={best_model}  "
                      f"scores={{ {score_str} }}")

    print(f"  [oracle] Wrote {total} entries → {output_path}")
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate oracle labels")
    parser.add_argument("--mmlu-samples", type=int, default=2000)
    parser.add_argument("--gsm8k-samples", type=int, default=500)
    parser.add_argument("--output", default="data/oracle_labels.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle-mode", default="judge_ppl_fallback",
                        choices=["exact_match", "judge", "judge_ppl_fallback"])
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu_only else "cpu"
    print(f"  [init] Device: {device}  Mode: {args.oracle_mode}")

    print("  [models] Loading base models…")
    models = load_all_models(encoding_device=device)

    total = generate_oracle_labels(
        models,
        mmlu_n=args.mmlu_samples,
        gsm8k_n=args.gsm8k_samples,
        seed=args.seed,
        output_path=args.output,
        oracle_mode=args.oracle_mode,
    )
    print(f"\n  ✅ Generated {total} oracle labels → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
