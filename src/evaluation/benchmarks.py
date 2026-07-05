"""
src/evaluation/benchmarks.py

Benchmark evaluation for capability measurement.

Supports:
  - MMLU (5‑shot)      — letter‑choice extraction, accuracy
  - GSM8K (8‑shot)     — numeric extraction, exact match
  - HumanEval (0‑shot) — pass@1 estimation
  - BBQ (0‑shot)       — DISAGGREGATED bias scoring (by category)

BBQ is NEVER reported as a single aggregate.  Results are reported
per bias category as specified in configs/evaluation.yaml.
"""
from __future__ import annotations

import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    score: float
    std_error: float | None = None
    num_samples: int = 0
    # BBQ: per‑category scores
    category_scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mmlu": {"num_fewshot": 5, "max_samples": 2000},
    "gsm8k": {"num_fewshot": 8},
    "humaneval": {"num_fewshot": 0},
    "bbq": {
        "num_fewshot": 0,
        "categories": [
            "Age", "Disability_status", "Gender_identity", "Nationality",
            "Physical_appearance", "Race_ethnicity", "Religion",
            "SES", "Sexual_orientation",
        ],
    },
    "arc_easy": {"num_fewshot": 0, "max_samples": 2000},
    "hellaswag": {"num_fewshot": 0, "max_samples": 2000},
    "winogrande": {"num_fewshot": 0, "max_samples": 2000},
    "boolq": {"num_fewshot": 0, "max_samples": 2000},
}


# ═══════════════════════════════════════════════════════════════════════════
# Answer extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_mmlu_answer(text: str) -> Optional[str]:
    """Extract predicted letter (A‑D) from MMLU response."""
    text = text.strip()
    for pat in [r"\(([A-D])\)", r"answer\s+is\s+([A-D])\b", r"answer\s*:\s*([A-D])\b",
                r"^([A-D])[\.\)\s]", r"\b([A-D])\b"]:
        m = re.search(pat, text, re.IGNORECASE if "answer" in pat else 0)
        if m:
            return m.group(1)
    return None


def _extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numeric answer from GSM8K response."""
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
            if abs(n) < 1e6 and abs(n) > 1e-8:
                return n
        except ValueError:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark runners
# ═══════════════════════════════════════════════════════════════════════════

def run_mmlu(
    generate_fn,
    num_fewshot: int = 5,
    max_samples: int = 2000,
    seed: int = 42,
    progress_callback=None,
) -> BenchmarkResult:
    """Run MMLU evaluation.

    Args:
        progress_callback: Optional callable receiving a dict with per-prompt progress info.
    """
    random.seed(seed)
    subjects = ["abstract_algebra", "college_chemistry", "computer_security",
                "high_school_mathematics", "international_law", "moral_scenarios"]
    items: List[dict] = []
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", streaming=True)
            for ex in ds:
                items.append({"subject": subj, "question": ex["question"],
                              "choices": [ex["choices"][i] if isinstance(ex["choices"], list)
                                          else ex.get(f"choice_{i}", "") for i in range(4)],
                              "answer": ex["answer"]})
                if sum(1 for it in items if it["subject"] == subj) >= max_samples // len(subjects):
                    break
        except Exception:
            continue

    random.shuffle(items)
    if max_samples:
        items = items[:max_samples]

    total = len(items)
    correct = 0
    for i, item in enumerate(items):
        prompt = item["question"] + "\n" + "\n".join(
            f"{chr(65+i)}. {c}" for i, c in enumerate(item["choices"])) + "\nAnswer:"
        response = generate_fn(prompt)
        pred = _extract_mmlu_answer(response)
        ans = item["answer"]
        if isinstance(ans, str):
            ans = ord(ans.upper()) - ord("A")
        passed = pred is not None and ord(pred) - ord("A") == ans
        if passed:
            correct += 1

        if progress_callback:
            progress_callback({
                "type": "prompt_result",
                "benchmark": "mmlu",
                "current": i + 1,
                "total": total,
                "correct": correct,
                "accuracy": correct / (i + 1),
                "prompt": prompt[:200],
                "prediction": pred or response[:50],
                "ground_truth": chr(ord("A") + ans) if isinstance(ans, int) else str(ans),
                "passed": passed,
            })

    acc = correct / total if total else 0
    return BenchmarkResult(name="mmlu", score=acc, num_samples=total)


def run_gsm8k(
    generate_fn,
    num_fewshot: int = 8,
    max_samples: Optional[int] = None,
    seed: int = 42,
    progress_callback=None,
) -> BenchmarkResult:
    """Run GSM8K evaluation."""
    random.seed(seed)
    items: List[dict] = []
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    for ex in ds:
        m = re.search(r"####\s*(-?[\d,]+\.?\d*)", ex["answer"])
        if m:
            items.append({"question": ex["question"], "answer": float(m.group(1).replace(",", ""))})
        if max_samples and len(items) >= max_samples:
            break

    random.shuffle(items)

    # Build few-shot prompt
    fewshot = ""
    for i in range(min(num_fewshot, len(items) - 1)):
        fewshot += f"Q: {items[i]['question']}\nA: {items[i]['answer']}\n\n"

    correct = 0
    test_items = items[num_fewshot:] if num_fewshot < len(items) else items
    total = len(test_items)
    for i, item in enumerate(test_items):
        prompt = fewshot + f"Q: {item['question']}\nA:"
        response = generate_fn(prompt)
        pred = _extract_gsm8k_answer(response)
        passed = pred is not None and abs(pred - item["answer"]) < 1e-6
        if passed:
            correct += 1

        if progress_callback:
            progress_callback({
                "type": "prompt_result",
                "benchmark": "gsm8k",
                "current": i + 1,
                "total": total,
                "correct": correct,
                "accuracy": correct / (i + 1),
                "prompt": prompt[:200],
                "prediction": str(pred) if pred is not None else response[:50],
                "ground_truth": str(item["answer"]),
                "passed": passed,
            })

    acc = correct / total if total else 0
    return BenchmarkResult(name="gsm8k", score=acc, num_samples=total)


def _execute_code_safely(code: str, test_code: str, timeout: int = 10) -> bool:
    """Execute generated code with test cases in a sandboxed subprocess.

    Returns True if all tests pass, False otherwise.
    """
    full_code = code + "\n\n" + test_code + "\n"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            import os
            os.unlink(f.name)
        except Exception:
            pass


def run_humaneval(
    generate_fn,
    max_samples: Optional[int] = None,
    seed: int = 42,
    progress_callback=None,
) -> BenchmarkResult:
    """Run HumanEval evaluation (pass@1)."""
    random.seed(seed)
    ds = load_dataset("openai/openai_humaneval", split="test", streaming=True)

    items: List[dict] = []
    for ex in ds:
        items.append({
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "canonical_solution": ex["canonical_solution"],
            "test": ex["test"],
            "entry_point": ex["entry_point"],
        })
        if max_samples and len(items) >= max_samples:
            break

    random.shuffle(items)

    correct = 0
    total = len(items)
    for i, item in enumerate(items):
        response = generate_fn(item["prompt"])

        completion = response.strip()
        if completion.startswith("```"):
            lines = completion.split("\n")
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            completion = "\n".join(lines)

        full_code = item["prompt"] + "\n" + completion
        passed = _execute_code_safely(full_code, item["test"])
        if passed:
            correct += 1

        if progress_callback:
            progress_callback({
                "type": "prompt_result",
                "benchmark": "humaneval",
                "current": i + 1,
                "total": total,
                "correct": correct,
                "accuracy": correct / (i + 1),
                "prompt": item["prompt"][:200],
                "prediction": response[:100],
                "ground_truth": item["canonical_solution"][:100],
                "passed": passed,
                "task_id": item["task_id"],
            })

    pass_at_1 = correct / total if total else 0
    return BenchmarkResult(name="humaneval", score=pass_at_1, num_samples=total)


def run_bbq(
    generate_fn,
    categories: Optional[List[str]] = None,
    max_samples_per_category: Optional[int] = None,
    seed: int = 42,
    progress_callback=None,
) -> BenchmarkResult:
    """Run BBQ evaluation — DISAGGREGATED by bias category."""
    random.seed(seed)
    if categories is None:
        categories = BENCHMARK_CONFIGS["bbq"]["categories"]

    category_scores: Dict[str, float] = {}
    category_correct: Dict[str, int] = {}
    category_total: Dict[str, int] = {}

    # Estimate total for progress reporting
    total_items = 0

    for cat in categories:
        correct = 0
        total = 0
        try:
            ds = load_dataset("heegyu/bbq", cat, split="test", streaming=True)
            for ex in ds:
                if max_samples_per_category and total >= max_samples_per_category:
                    break
                question = ex["question"]
                context = ex.get("context", "")
                if context:
                    question = f"{context}\n\n{question}"

                response = generate_fn(question)
                pred = _extract_mmlu_answer(response)
                ans_idx = ex.get("answer", 0)
                if isinstance(ans_idx, str):
                    ans_idx = ord(ans_idx.upper()) - ord("A")

                passed = pred is not None and ord(pred) - ord("A") == ans_idx
                if passed:
                    correct += 1
                total += 1
                total_items += 1

                if progress_callback:
                    progress_callback({
                        "type": "prompt_result",
                        "benchmark": "bbq",
                        "category": cat,
                        "current": total,
                        "total": max_samples_per_category or "?",
                        "correct": correct,
                        "accuracy": correct / total if total else 0,
                        "prompt": question[:200],
                        "prediction": pred or response[:50],
                        "ground_truth": chr(ord("A") + ans_idx) if isinstance(ans_idx, int) else str(ans_idx),
                        "passed": passed,
                    })
        except Exception as e:
            print(f"    Warning: BBQ/{cat} failed: {e}")
            continue

        category_correct[cat] = correct
        category_total[cat] = total
        category_scores[cat] = correct / total if total > 0 else 0.0

    overall = float(np.mean(list(category_scores.values()))) if category_scores else 0.0

    return BenchmarkResult(
        name="bbq",
        score=overall,
        num_samples=total_items,
        category_scores=category_scores,
        details={"category_correct": category_correct, "category_total": category_total},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Easy benchmarks (suitable for 0.5B-1B models)
# ═══════════════════════════════════════════════════════════════════════════

def run_arc_easy(
    generate_fn,
    max_samples: Optional[int] = None,
    seed: int = 42,
    progress_callback=None,
) -> BenchmarkResult:
    """Run ARC-Easy evaluation (science questions, 4-choice)."""
    random.seed(seed)
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", streaming=True)

    items: List[dict] = []
    for ex in ds:
        choices = ex["choices"]
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        items.append({
            "question": ex["question"],
            "choices": texts,
            "labels": labels,
            "answer": ex["answerKey"],
        })
        if max_samples and len(items) >= max_samples:
            break

    random.shuffle(items)

    correct = 0
    total = len(items)
    for i, item in enumerate(items):
        prompt = item["question"] + "\n" + "\n".join(
            f"{label}. {text}" for label, text in zip(item["labels"], item["choices"])
        ) + "\nAnswer:"
        response = generate_fn(prompt)
        pred = _extract_mmlu_answer(response)
        passed = pred is not None and pred.upper() == item["answer"].upper()
        if passed:
            correct += 1

        if progress_callback:
            progress_callback({
                "type": "prompt_result",
                "benchmark": "arc_easy",
                "current": i + 1,
                "total": total,
                "correct": correct,
                "accuracy": correct / (i + 1),
                "prompt": prompt[:200],
                "prediction": pred or response[:50],
                "ground_truth": item["answer"],
                "passed": passed,
            })

    acc = correct / total if total else 0
    return BenchmarkResult(name="arc_easy", score=acc, num_samples=total)


def run_hellaswag(
    generate_fn,
    max_samples: Optional[int] = None,
    seed: int = 42,
    progress_callback=None,
) -> BenchmarkResult:
    """Run HellaSwag evaluation (commonsense completion, 4-choice).

    Given a context, pick the most plausible continuation.
    """
    random.seed(seed)
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)

    items: List[dict] = []
    for ex in ds:
        items.append({
            "context": ex["ctx"],
            "endings": ex["endings"],
            "label": int(ex["label"]),
        })
        if max_samples and len(items) >= max_samples:
            break

    random.shuffle(items)

    correct = 0
    total = len(items)
    for i, item in enumerate(items):
        prompt = (
            f"Complete the following: {item['context']}\n"
            f"Which continuation is correct?\n"
            + "\n".join(f"({chr(65+j)}) {ending}" for j, ending in enumerate(item["endings"]))
            + "\nAnswer:"
        )
        response = generate_fn(prompt)
        pred = _extract_mmlu_answer(response)
        if pred:
            pred_idx = ord(pred.upper()) - ord("A")
        else:
            pred_idx = -1
        passed = pred_idx == item["label"]
        if passed:
            correct += 1

        if progress_callback:
            progress_callback({
                "type": "prompt_result",
                "benchmark": "hellaswag",
                "current": i + 1,
                "total": total,
                "correct": correct,
                "accuracy": correct / (i + 1),
                "prompt": prompt[:200],
                "prediction": pred or response[:50],
                "ground_truth": chr(ord("A") + item["label"]),
                "passed": passed,
            })

    acc = correct / total if total else 0
    return BenchmarkResult(name="hellaswag", score=acc, num_samples=total)


def run_winogrande(
    generate_fn,
    max_samples: Optional[int] = None,
    seed: int = 42,
    progress_callback=None,
) -> BenchmarkResult:
    """Run WinoGrande evaluation (coreference resolution, 2-choice).

    Given a sentence with a blank, pick the correct pronoun.
    """
    random.seed(seed)
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", streaming=True)

    items: List[dict] = []
    for ex in ds:
        answer_idx = int(ex["answer"]) - 1  # answer is "1" or "2"
        items.append({
            "sentence": ex["sentence"],
            "options": [ex["option1"], ex["option2"]],
            "answer_idx": answer_idx,
        })
        if max_samples and len(items) >= max_samples:
            break

    random.shuffle(items)

    correct = 0
    total = len(items)
    for i, item in enumerate(items):
        prompt = item["sentence"].replace("_", f"___")
        prompt += f"\nA. {item['options'][0]}\nB. {item['options'][1]}\nAnswer:"
        response = generate_fn(prompt)
        pred = _extract_mmlu_answer(response)
        if pred:
            pred_idx = ord(pred.upper()) - ord("A")
        else:
            pred_idx = -1
        passed = pred_idx == item["answer_idx"]
        if passed:
            correct += 1

        if progress_callback:
            progress_callback({
                "type": "prompt_result",
                "benchmark": "winogrande",
                "current": i + 1,
                "total": total,
                "correct": correct,
                "accuracy": correct / (i + 1),
                "prompt": item["sentence"][:200],
                "prediction": pred or response[:50],
                "ground_truth": chr(ord("A") + item["answer_idx"]),
                "passed": passed,
            })

    acc = correct / total if total else 0
    return BenchmarkResult(name="winogrande", score=acc, num_samples=total)


def run_boolq(
    generate_fn,
    max_samples: Optional[int] = None,
    seed: int = 42,
    progress_callback=None,
) -> BenchmarkResult:
    """Run BoolQ evaluation (boolean questions).

    Given a passage and question, answer True/False.
    """
    random.seed(seed)
    ds = load_dataset("boolq", split="validation", streaming=True)

    items: List[dict] = []
    for ex in ds:
        items.append({
            "question": ex["question"],
            "passage": ex["passage"],
            "answer": ex["answer"],  # True/False
        })
        if max_samples and len(items) >= max_samples:
            break

    random.shuffle(items)

    correct = 0
    total = len(items)
    for i, item in enumerate(items):
        prompt = f"Passage: {item['passage']}\n\nQuestion: {item['question']}\nTrue or False?"
        response = generate_fn(prompt)
        response_lower = response.strip().lower()
        first_word = response_lower.split()[0] if response_lower.split() else ""
        if first_word in ("true", "yes", "a") or re.search(r"\btrue\b", response_lower):
            pred = "True"
        elif first_word in ("false", "no", "b") or re.search(r"\bfalse\b", response_lower):
            pred = "False"
        else:
            pred = None
        answer_str = "True" if item["answer"] else "False"
        passed = pred == answer_str
        if passed:
            correct += 1

        if progress_callback:
            progress_callback({
                "type": "prompt_result",
                "benchmark": "boolq",
                "current": i + 1,
                "total": total,
                "correct": correct,
                "accuracy": correct / (i + 1),
                "prompt": prompt[:200],
                "prediction": pred or response[:50],
                "ground_truth": answer_str,
                "passed": passed,
            })

    acc = correct / total if total else 0
    return BenchmarkResult(name="boolq", score=acc, num_samples=total)
