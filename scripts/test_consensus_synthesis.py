"""
Consensus-Aware Synthesis Test

Compare old synthesis vs new consensus-aware synthesis.
Test on code debugging, math, and logic prompts.
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.meta_model.synthesis import (
    consensus_aware_synthesize,
    detect_task_type,
    compute_consensus_score,
    TaskType,
    SynthesisStrategy,
)


# ── Test Prompts ──────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    # Code debugging (both models should be correct)
    {
        "prompt": "What's wrong with this Python code?\n\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n)",
        "reference": "Missing n-1",
        "task_type": TaskType.CODE,
    },
    {
        "prompt": "What's wrong with this Python code?\n\ndef find_max(lst):\n    max_val = 0\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val",
        "reference": "max_val should start as lst[0]",
        "task_type": TaskType.CODE,
    },
    {
        "prompt": "What's wrong with this Python code?\n\ndef count_vowels(s):\n    count = 0\n    for char in s:\n        if char in 'aeiou':\n            count += 1\n    return count",
        "reference": "Missing .lower()",
        "task_type": TaskType.CODE,
    },
    {
        "prompt": "What's wrong with this Python code?\n\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result",
        "reference": "Missing extend",
        "task_type": TaskType.CODE,
    },
    {
        "prompt": "What's wrong with this code?\n\ndef binary_search(arr, target):\n    low, high = 0, len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid\n    return -1",
        "reference": "low = mid + 1",
        "task_type": TaskType.CODE,
    },
    # Math reasoning
    {
        "prompt": "A train travels at 60 mph for 2 hours, then 40 mph for 3 hours. What is the average speed?",
        "reference": "48",
        "task_type": TaskType.MATH,
    },
    {
        "prompt": "If a rectangle has length 12 and width 5, and a triangle has base 12 and height 5, what is the ratio of their areas?",
        "reference": "2",
        "task_type": TaskType.MATH,
    },
    {
        "prompt": "You buy 3 shirts at $15 each and 2 pairs of pants at $25 each. You get a 20% discount. How much do you pay?",
        "reference": "76",
        "task_type": TaskType.MATH,
    },
]


def load_7b_models():
    """Load cached 7B models."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    models = {}
    tokenizers = {}

    # Coder
    print("  Loading coder...")
    t0 = time.time()
    code_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B", trust_remote_code=True)
    code_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B",
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    code_model.eval()
    print(f"    Loaded in {time.time()-t0:.1f}s ({torch.cuda.memory_allocated(0)/1e9:.1f} GB)")
    models["coder"] = code_model
    tokenizers["coder"] = code_tokenizer

    # Reasoning
    print("  Loading reasoning...")
    t0 = time.time()
    reason_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    reason_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    reason_model.eval()
    print(f"    Loaded in {time.time()-t0:.1f}s ({torch.cuda.memory_allocated(0)/1e9:.1f} GB)")
    models["reasoning"] = reason_model
    tokenizers["reasoning"] = reason_tokenizer

    return models, tokenizers


def generate_from_model(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate text from a model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result.strip()


def old_synthesize(outputs: Dict[str, str], judge_model, judge_tokenizer) -> str:
    """Old synthesis (judge always picks best answer)."""
    from collections import Counter
    import re

    # Simple majority vote
    answers = []
    for mid, out in outputs.items():
        first_line = out.strip().split('\n')[0].strip()
        match = re.search(r'the answer is[:\s]+(.+?)(?:\.|$)', first_line, re.IGNORECASE)
        if match:
            answers.append(match.group(1).strip().lower())
        else:
            answers.append(first_line.lower())

    if not answers:
        return list(outputs.values())[0] if outputs else ""

    counter = Counter(answers)
    most_common = counter.most_common(1)[0][0]

    for mid, out in outputs.items():
        first_line = out.strip().split('\n')[0].strip()
        match = re.search(r'the answer is[:\s]+(.+?)(?:\.|$)', first_line, re.IGNORECASE)
        if match:
            ans = match.group(1).strip().lower()
        else:
            ans = first_line.lower()
        if ans == most_common:
            return out

    return list(outputs.values())[0]


def check_match(output: str, reference: str) -> bool:
    """Check if output matches reference."""
    import re
    output_lower = output.lower().strip()
    reference_lower = reference.lower().strip()

    if reference_lower in output_lower:
        return True

    key_concepts = {
        "Missing n-1": ["n-1", "n - 1", "factorial(n-1)", "factorial(n - 1)"],
        "max_val should start as lst[0]": ["lst[0]", "first element", "initial"],
        "Missing .lower()": ["lower()", "case", "uppercase"],
        "Missing extend": ["extend", "remaining", "leftover"],
        "low = mid + 1": ["mid + 1", "mid+1", "progress"],
    }

    for concept in key_concepts.get(reference, []):
        if concept in output_lower:
            return True

    return False


def check_numeric_match(output: str, reference: str) -> bool:
    """Check if output matches numeric reference."""
    import re
    output_lower = output.lower().strip()
    reference_lower = reference.lower().strip()

    if reference_lower in output_lower:
        return True

    # Extract numbers from output
    numbers = re.findall(r'-?[\d,]+\.?\d*', output)
    for num in numbers:
        try:
            n = float(num.replace(",", ""))
            ref = float(reference)
            if abs(n - ref) < 0.1 or abs(n - ref) / max(abs(ref), 1e-9) < 0.05:
                return True
        except ValueError:
            continue

    return False


def main():
    print("=" * 80)
    print("CONSENSUS-AWARE SYNTHESIS TEST")
    print("Compare old synthesis vs new consensus-aware synthesis")
    print("=" * 80)

    # Load models
    print("\n[1/4] Loading cached 7B models...")
    models, tokenizers = load_7b_models()

    # Run test
    print("\n[2/4] Running test...")
    results = []

    for idx, item in enumerate(TEST_PROMPTS):
        prompt = item["prompt"]
        ref = item["reference"]
        expected_task = item["task_type"]

        # Generate from both models
        outputs = {}
        for role, model in models.items():
            tokenizer = tokenizers[role]
            output = generate_from_model(model, tokenizer, prompt)
            outputs[role] = output

        # Detect task type
        detected_task = detect_task_type(prompt)

        # Compute consensus
        consensus_score, core_answers = compute_consensus_score(outputs)

        # Old synthesis (majority vote)
        old_result = old_synthesize(outputs, models["reasoning"], tokenizers["reasoning"])

        # New consensus-aware synthesis
        new_result = consensus_aware_synthesize(prompt, outputs)

        # Check correctness
        is_numeric = expected_task == TaskType.MATH
        if is_numeric:
            old_correct = check_numeric_match(old_result, ref)
            new_correct = check_numeric_match(new_result.answer, ref)
        else:
            old_correct = check_match(old_result, ref)
            new_correct = check_match(new_result.answer, ref)

        # Per-model correctness
        model_correct = {}
        for role, out in outputs.items():
            if is_numeric:
                model_correct[role] = check_numeric_match(out, ref)
            else:
                model_correct[role] = check_match(out, ref)

        results.append({
            "prompt": prompt[:100],
            "reference": ref,
            "task_type": expected_task.value,
            "detected_task": detected_task.value,
            "consensus_score": consensus_score,
            "model_correct": model_correct,
            "old_synthesis": old_result[:150],
            "old_correct": old_correct,
            "new_strategy": new_result.strategy_used.value,
            "new_synthesis": new_result.answer[:150],
            "new_correct": new_correct,
            "needs_review": new_result.needs_review,
        })

        status_old = "PASS" if old_correct else "FAIL"
        status_new = "PASS" if new_correct else "FAIL"
        print(f"  [{idx+1}/{len(TEST_PROMPTS)}] {expected_task.value:6s} | consensus={consensus_score:.2f} | old={status_old} new={status_new} | strategy={new_result.strategy_used.value}")

    # Compute metrics
    print("\n[3/4] Computing metrics...")

    old_correct_count = sum(1 for r in results if r["old_correct"])
    new_correct_count = sum(1 for r in results if r["new_correct"])

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\n  Overall Accuracy:")
    print(f"    Old synthesis (majority vote): {old_correct_count}/{len(results)} ({old_correct_count/len(results):.1%})")
    print(f"    New synthesis (consensus-aware): {new_correct_count}/{len(results)} ({new_correct_count/len(results):.1%})")

    # Per task type
    for task_type in ["code", "math"]:
        task_results = [r for r in results if r["task_type"] == task_type]
        if not task_results:
            continue
        old_acc = sum(1 for r in task_results if r["old_correct"]) / len(task_results)
        new_acc = sum(1 for r in task_results if r["new_correct"]) / len(task_results)
        print(f"\n    {task_type.upper()} ({len(task_results)} prompts):")
        print(f"      Old: {old_acc:.1%}")
        print(f"      New: {new_acc:.1%}")

    # Strategy distribution
    strategy_counts = {}
    for r in results:
        s = r["new_strategy"]
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    print("\n  Strategy Distribution:")
    for strategy, count in strategy_counts.items():
        print(f"    {strategy}: {count}/{len(results)}")

    # Verdict
    print("\n" + "=" * 80)
    if new_correct_count > old_correct_count:
        print("VERDICT: PASS — Consensus-aware synthesis improves accuracy")
    elif new_correct_count == old_correct_count:
        print("VERDICT: NEUTRAL — Consensus-aware synthesis matches old synthesis")
    else:
        print("VERDICT: REGRESSION — Consensus-aware synthesis hurts accuracy")
    print("=" * 80)

    # Save results
    output_dir = Path("results/consensus_synthesis_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({"results": results}, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
