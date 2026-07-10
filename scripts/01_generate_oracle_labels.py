#!/usr/bin/env python3
"""
scripts/01_generate_oracle_labels.py

Fully automated oracle label generation for router bootstrap training.

Benchmarks (easier, suitable for 0.5B–1B models):
  ARC-Easy      Science multiple-choice (4 options)
  HellaSwag     Commonsense sentence completion (4 options)
  WinoGrande    Coreference resolution (2 options)
  BoolQ         Boolean questions (True/False)

Three oracle modes (--oracle-mode):
  exact_match          Match against ground-truth answer.
                       Fast but limited — only works for BoolQ/ARC.
  judge                Qwen2.5-1.5B-Instruct ranks model answers by quality.
                       ⭐ RECOMMENDED. Works for all benchmarks.
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

def _format_multiple_choice(question: str, choices: List[str], labels: List[str]) -> str:
    """Format a multiple-choice question (ARC, HellaSwag style)."""
    parts = [question]
    for label, choice in zip(labels, choices):
        parts.append(f"{label}. {choice}")
    parts.append("\nAnswer:")
    return "\n".join(parts)


def _format_boolq(question: str, passage: str) -> str:
    """Format a boolean question with passage context."""
    return f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer (True or False):"


def _format_winogrande(sentence: str, option: str) -> str:
    """Format a WinoGrande sentence with the option filled in."""
    return f"{sentence}\n\nIs this correct? Answer True or False:"


# ═══════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_arc_easy_prompts(n: int, seed: int) -> List[Dict[str, Any]]:
    """Load ARC-Easy: science multiple-choice (4 options)."""
    print(f"  [data] Loading ARC-Easy (target: {n} samples)…")
    items: List[Dict[str, Any]] = []
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", streaming=True)
        for ex in ds:
            choices = ex["choices"]
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            items.append({
                "question": ex["question"],
                "choices": texts,
                "labels": labels,
                "answer": ex["answerKey"],
                "source": "arc_easy",
            })
            if len(items) >= n:
                break
    except Exception as e:
        print(f"    Warning: could not load ARC-Easy: {e}")
    random.Random(seed).shuffle(items)
    return items[:n]


def _load_hellaswag_prompts(n: int, seed: int) -> List[Dict[str, Any]]:
    """Load HellaSwag: commonsense sentence completion (4 options)."""
    print(f"  [data] Loading HellaSwag (target: {n} samples)…")
    items: List[Dict[str, Any]] = []
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
        for ex in ds:
            items.append({
                "context": ex["ctx"],
                "endings": ex["endings"],
                "label": int(ex["label"]),
                "source": "hellaswag",
            })
            if len(items) >= n:
                break
    except Exception as e:
        print(f"    Warning: could not load HellaSwag: {e}")
    random.Random(seed).shuffle(items)
    return items[:n]


def _load_winogrande_prompts(n: int, seed: int) -> List[Dict[str, Any]]:
    """Load WinoGrande: coreference resolution (2 options)."""
    print(f"  [data] Loading WinoGrande (target: {n} samples)…")
    items: List[Dict[str, Any]] = []
    try:
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", streaming=True)
        for ex in ds:
            answer_idx = int(ex["answer"]) - 1  # answer is "1" or "2"
            items.append({
                "sentence": ex["sentence"],
                "options": [ex["option1"], ex["option2"]],
                "answer_idx": answer_idx,
                "source": "winogrande",
            })
            if len(items) >= n:
                break
    except Exception as e:
        print(f"    Warning: could not load WinoGrande: {e}")
    random.Random(seed).shuffle(items)
    return items[:n]


def _load_boolq_prompts(n: int, seed: int) -> List[Dict[str, Any]]:
    """Load BoolQ: boolean questions (True/False)."""
    print(f"  [data] Loading BoolQ (target: {n} samples)…")
    items: List[Dict[str, Any]] = []
    try:
        ds = load_dataset("boolq", split="validation", streaming=True)
        for ex in ds:
            items.append({
                "question": ex["question"],
                "passage": ex["passage"],
                "answer": ex["answer"],  # True/False
                "source": "boolq",
            })
            if len(items) >= n:
                break
    except Exception as e:
        print(f"    Warning: could not load BoolQ: {e}")
    random.Random(seed).shuffle(items)
    return items[:n]


def _load_humaneval_prompts(n: int, seed: int) -> List[Dict[str, Any]]:
    """Load HumanEval: code generation (function body from signature + docstring)."""
    print(f"  [data] Loading HumanEval (target: {n} samples)…")
    items: List[Dict[str, Any]] = []
    try:
        ds = load_dataset("openai/openai_humaneval", split="test", streaming=True)
        for ex in ds:
            items.append({
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "canonical_solution": ex["canonical_solution"],
                "test": ex["test"],
                "entry_point": ex["entry_point"],
                "source": "humaneval",
            })
            if len(items) >= n:
                break
    except Exception as e:
        print(f"    Warning: could not load HumanEval: {e}")
    random.Random(seed).shuffle(items)
    return items[:n]


# ═══════════════════════════════════════════════════════════════════════════
# Prompt builders — turn raw data into prompt strings
# ═══════════════════════════════════════════════════════════════════════════

def _build_prompt(item: Dict[str, Any]) -> str:
    """Build a prompt string from a dataset item."""
    source = item["source"]

    if source == "arc_easy":
        labels = item.get("labels", ["A", "B", "C", "D"])
        return _format_multiple_choice(item["question"], item["choices"], labels)

    elif source == "hellaswag":
        labels = ["A", "B", "C", "D"]
        return _format_multiple_choice(item["context"], item["endings"], labels)

    elif source == "winogrande":
        # For winogrande, we present the sentence with a blank and two options
        s = item["sentence"]
        opts = item["options"]
        return f"Complete the sentence:\n{s}\n\nOption A: {opts[0]}\nOption B: {opts[1]}\n\nAnswer (A or B):"

    elif source == "boolq":
        return _format_boolq(item["question"], item["passage"])

    elif source == "humaneval":
        return f"Complete the Python function.\n\n{item['prompt']}"

    else:
        return item.get("question", str(item))


def _get_ground_truth(item: Dict[str, Any]) -> str:
    """Extract ground-truth answer letter/text for exact_match mode."""
    source = item["source"]

    if source == "arc_easy":
        return item["answer"]  # e.g. "B"

    elif source == "hellaswag":
        return chr(65 + item["label"])  # 0→A, 1→B, etc.

    elif source == "winogrande":
        return chr(65 + item["answer_idx"])  # 0→A, 1→B

    elif source == "boolq":
        return "True" if item["answer"] else "False"

    elif source == "humaneval":
        return item.get("canonical_solution", "")

    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Oracle: Exact Match
# ═══════════════════════════════════════════════════════════════════════════

def _extract_answer_letter(text: str) -> Optional[str]:
    """Extract an answer letter (A-D) from model output."""
    text = text.strip()
    for pat in [r"\(([A-D])\)", r"answer\s+is\s+([A-D])\b", r"answer\s*:\s*([A-D])\b",
                r"^([A-D])[\.\)\s]", r"\b([A-D])\b"]:
        m = re.search(pat, text, re.IGNORECASE if "answer" in pat else 0)
        if m:
            return m.group(1).upper()
    return None


def _extract_boolq_answer(text: str) -> Optional[bool]:
    """Extract True/False from model output."""
    text = text.strip().lower()
    if text.startswith("true") or "yes" in text:
        return True
    if text.startswith("false") or "no" in text:
        return False
    return None


def _score_exact_match(
    model_answers: Dict[str, str],
    ground_truth: str,
    source: str,
    model_ids: List[str],
) -> Tuple[Dict[str, float], str]:
    """Score models by exact match against ground truth."""
    scores: Dict[str, float] = {}
    for mid in model_ids:
        ans = model_answers[mid]
        if source == "boolq":
            pred = _extract_boolq_answer(ans)
            gt = ground_truth.lower() == "true"
            scores[mid] = 1.0 if pred == gt else 0.0
        elif source == "humaneval":
            # Code generation — exact match not meaningful, use judge instead
            scores[mid] = 0.0
        else:
            # ARC, HellaSwag, WinoGrande — all letter-based
            pred = _extract_answer_letter(ans)
            scores[mid] = 1.0 if pred == ground_truth.upper() else 0.0

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
) -> Tuple[Dict[str, float], str, float, str]:
    """Use Qwen2.5-1.5B-Instruct to rank model answers.

    Returns (scores_dict, best_model, confidence, raw_response).
    """
    model_ids = sorted(model_answers.keys())
    n = len(model_ids)

    # Build answer list with labels
    answers_str = ""
    for i, mid in enumerate(model_ids):
        label = chr(65 + i)  # A, B, C, D
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

    # Parse JSON — fix unquoted identifiers like [A, B, C]
    response_clean = re.sub(r'\[\s*([A-Za-z_][\w]*)\s*([,\]])', r'["\1"\2', response)
    response_clean = re.sub(r'([\[,])\s*([A-Za-z_][\w]*)\s*\]', r'\1"\2"]', response_clean)

    try:
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
            m = re.search(r'\{[^{}]*\}', response_clean, re.DOTALL)
            raw_json = m.group(0) if m else response_clean
            result = json.loads(raw_json)

        ranking_raw = result.get("ranking", [])
        raw_conf = result.get("confidence")
        confidence = 0.5 if raw_conf is None else float(raw_conf)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        ranking_raw = []
        confidence = 0.0

    # Map ranking entries to model IDs
    letter_map = {chr(65 + i): mid for i, mid in enumerate(model_ids)}

    ranking_str = str(ranking_raw)
    raw_tokens = re.findall(r'"([^"]+)"|\'([^\']+)\'|([A-Za-z_]\w*)|(\d+)', ranking_str)
    tokens = [t[0] or t[1] or t[2] or t[3] for t in raw_tokens]

    ranking: List[str] = []
    for token in tokens:
        token = token.strip()
        if token in model_ids:
            ranking.append(token)
        elif token.upper() in letter_map:
            ranking.append(letter_map[token.upper()])
        elif token.isdigit():
            pos = int(token) - 1
            if 0 <= pos < len(model_ids):
                ranking.append(model_ids[pos])
        else:
            for mid in model_ids:
                if mid.lower() in token.lower():
                    ranking.append(mid)
                    break

    missing = [mid for mid in model_ids if mid not in ranking]
    random.shuffle(missing)
    ranking.extend(missing)
    seen = set()
    ranking = [r for r in ranking if not (r in seen or seen.add(r))]

    scores: Dict[str, float] = {}
    for rank, mid in enumerate(ranking):
        scores[mid] = float(n - rank)

    for mid in model_ids:
        if mid not in scores:
            scores[mid] = 0.0

    best_model = ranking[0] if ranking else model_ids[0]
    return scores, best_model, confidence, response


# ═══════════════════════════════════════════════════════════════════════════
# Oracle: Perplexity-based
# ═══════════════════════════════════════════════════════════════════════════

def _score_perplexity(
    prompt: str,
    model_answers: Dict[str, str],
    models: Dict[str, FrozenModelWrapper],
    model_ids: List[str],
) -> Tuple[Dict[str, float], str]:
    """Score models by perplexity of the best available answer."""
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
# Main oracle generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_oracle_labels(
    models: Dict[str, FrozenModelWrapper],
    arc_n: int = 500,
    hellaswag_n: int = 500,
    winogrande_n: int = 500,
    boolq_n: int = 500,
    humaneval_n: int = 164,
    seed: int = 42,
    output_path: str = "data/oracle_labels.jsonl",
    oracle_mode: str = "judge_ppl_fallback",
    judge_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_samples: Optional[int] = None,
) -> int:
    random.seed(seed)

    # Load prompts from all benchmarks
    all_prompts: List[Dict[str, Any]] = []
    all_prompts.extend(_load_arc_easy_prompts(arc_n, seed))
    all_prompts.extend(_load_hellaswag_prompts(hellaswag_n, seed))
    all_prompts.extend(_load_winogrande_prompts(winogrande_n, seed))
    all_prompts.extend(_load_boolq_prompts(boolq_n, seed))
    all_prompts.extend(_load_humaneval_prompts(humaneval_n, seed))
    random.shuffle(all_prompts)

    if max_samples is not None and max_samples < len(all_prompts):
        all_prompts = all_prompts[:max_samples]

    # Source distribution
    source_counts: Dict[str, int] = {}
    for p in all_prompts:
        src = p["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    source_str = ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items()))
    print(f"  [oracle] Total prompts: {len(all_prompts)}  ({source_str})  mode: {oracle_mode}")

    model_ids = sorted(models.keys())
    print(f"  [oracle] Models: {model_ids}")
    device = next(iter(models.values())).encoding_device
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Emit start event for UI progress tracking
    total_prompts = len(all_prompts)
    print(json.dumps({
        "type": "oracle_start",
        "total": total_prompts,
        "sources": source_counts,
        "models": model_ids,
        "oracle_mode": oracle_mode,
    }, default=str), flush=True)

    # Load judge if needed
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
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_path.parent

    versioned_path = output_dir / f"oracle_labels_{timestamp}.jsonl"
    latest_path = output_dir / "oracle_labels_latest.jsonl"

    with open(versioned_path, "w") as f:
        for idx, item in enumerate(all_prompts):
            prompt_text = _build_prompt(item)
            source = item["source"]
            ground_truth = _get_ground_truth(item)

            # Generate answers from all models
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

            # Score
            judge_raw_output = ""
            judge_succeeded = False
            if oracle_mode == "exact_match":
                scores, best_model = _score_exact_match(
                    model_answers, ground_truth, source, model_ids,
                )
                judge_mode = "exact_match"
            elif oracle_mode == "judge":
                scores, best_model, confidence, judge_raw_output = _rank_with_judge(
                    prompt_text, model_answers, judge_model, judge_tokenizer, device,
                )
                judge_succeeded = confidence > 0.0
                judge_mode = "judge" if judge_succeeded else "judge_failed"
            elif oracle_mode == "judge_ppl_fallback":
                scores, best_model, confidence, judge_raw_output = _rank_with_judge(
                    prompt_text, model_answers, judge_model, judge_tokenizer, device,
                )
                judge_succeeded = confidence > 0.0
                if confidence < 0.5:
                    ppl_scores, ppl_best = _score_perplexity(
                        prompt_text, model_answers, models, model_ids,
                    )
                    for mid in model_ids:
                        scores[mid] = 0.3 * scores.get(mid, 0) + 0.7 * ppl_scores.get(mid, 0)
                    best_model = max(scores, key=scores.get)
                    judge_mode = "judge_ppl_blend" if judge_succeeded else "ppl_fallback"
                else:
                    judge_mode = "judge"
            else:
                raise ValueError(f"Unknown oracle_mode: {oracle_mode}")

            # Normalize scores to [0, 1]
            max_s = max(scores.values()) if scores else 1.0
            if max_s > 0:
                scores = {mid: s / max_s for mid, s in scores.items()}
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
                "judge_raw_output": judge_raw_output,
                "judge_mode": judge_mode,
                "ground_truth": ground_truth,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total += 1

            if (idx + 1) % 10 == 0:
                print(json.dumps({
                    "type": "oracle_progress",
                    "current": idx + 1,
                    "total": total_prompts,
                    "best_model": best_model,
                    "scores": {k: round(v, 3) for k, v in scores.items()},
                    "source": source,
                }, default=str), flush=True)

    # Update latest pointer
    import shutil
    shutil.copy2(versioned_path, latest_path)

    # Update history index
    history_path = output_dir / "oracle_labels_history.json"
    history = []
    if history_path.exists():
        with open(history_path) as hf:
            history = json.load(hf)

    history.append({
        "filename": versioned_path.name,
        "timestamp": timestamp,
        "total_entries": total,
        "oracle_mode": oracle_mode,
        "sources": source_counts,
        "models": model_ids,
    })

    with open(history_path, "w") as hf:
        json.dump(history, hf, indent=2)

    # Emit completion event for UI
    print(json.dumps({
        "type": "oracle_complete",
        "total": total,
        "filename": versioned_path.name,
        "source_counts": source_counts,
    }, default=str), flush=True)

    print(f"  [oracle] Wrote {total} entries → {versioned_path}")
    print(f"  [oracle] Latest → {latest_path}")
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate oracle labels")
    parser.add_argument("--arc-samples", type=int, default=500,
                        help="Number of ARC-Easy samples")
    parser.add_argument("--hellaswag-samples", type=int, default=500,
                        help="Number of HellaSwag samples")
    parser.add_argument("--winogrande-samples", type=int, default=500,
                        help="Number of WinoGrande samples")
    parser.add_argument("--boolq-samples", type=int, default=500,
                        help="Number of BoolQ samples")
    parser.add_argument("--humaneval-samples", type=int, default=164,
                        help="Number of HumanEval samples (code generation)")
    parser.add_argument("--output", default="data/oracle_labels.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle-mode", default="judge_ppl_fallback",
                        choices=["exact_match", "judge", "judge_ppl_fallback"])
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total number of prompts (overrides per-benchmark counts)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu_only else "cpu"
    print(f"  [init] Device: {device}  Mode: {args.oracle_mode}")

    print("  [models] Loading base models…")
    models = load_all_models(encoding_device=device)

    total = generate_oracle_labels(
        models,
        arc_n=args.arc_samples,
        hellaswag_n=args.hellaswag_samples,
        winogrande_n=args.winogrande_samples,
        boolq_n=args.boolq_samples,
        humaneval_n=args.humaneval_samples,
        seed=args.seed,
        output_path=args.output,
        oracle_mode=args.oracle_mode,
        max_samples=args.max_samples,
    )
    print(f"\n  ✅ Generated {total} oracle labels → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
