"""
Full 7B Ensemble Benchmark with Consensus-Aware Synthesis

Compare:
1. Old synthesis (majority vote)
2. New consensus-aware synthesis (task-aware routing)
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.meta_model.synthesis import (
    consensus_aware_synthesize,
    detect_task_type,
    compute_consensus_score,
    compute_semantic_consensus,
    TaskType,
    SynthesisStrategy,
)


# ── Test Prompts ──────────────────────────────────────────────────────────────

MATH_REASONING = [
    ("A train travels at 60 mph for 2 hours, then 40 mph for 3 hours. What is the average speed?", "48"),
    ("If a rectangle has length 12 and width 5, and a triangle has base 12 and height 5, what is the ratio of their areas?", "2"),
    ("You buy 3 shirts at $15 each and 2 pairs of pants at $25 each. You get a 20% discount. How much do you pay?", "76"),
    ("A pool is filled by pipe A in 6 hours and pipe B in 4 hours. How long to fill the pool with both open?", "2.4"),
    ("If you invest $1000 at 5% annual compound interest, how much do you have after 3 years?", "1157.63"),
    ("A car uses 8 liters per 100km. Gas costs $1.50 per liter. How much does a 240km trip cost?", "28.80"),
    ("Three workers can build a wall in 4 days. How many workers are needed to build it in 1 day?", "12"),
    ("A bakery sells cupcakes for $2.50 each. They sell 40 per day. How many days to earn $500?", "5"),
    ("If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?", "5"),
    ("You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?", "4"),
]

LOGIC_PUZZLES = [
    ("If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?", "No"),
    ("A farmer has 17 sheep. All but 9 die. How many are left?", "9"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5 minutes"),
    ("Two coins add up to 30 cents. One of them is not a nickel. What are the two coins?", "quarter and nickel"),
    ("A doctor gives you 3 pills and says to take one every 30 minutes. How long do they last?", "60 minutes"),
    ("If you have a bowl with six apples and you take away four, how many do YOU have?", "4"),
    ("What comes next: 1, 1, 2, 3, 5, 8, ...?", "13"),
    ("A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?", "0.05"),
]

CODE_DEBUGGING = [
    ("What's wrong with this Python code?\n\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n)\n\nThe function causes infinite recursion.", "Missing n-1"),
    ("What's wrong with this Python code?\n\ndef find_max(lst):\n    max_val = 0\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val\n\nThis fails for all-negative lists.", "max_val should start as lst[0]"),
    ("What's wrong with this Python code?\n\ndef count_vowels(s):\n    count = 0\n    for char in s:\n        if char in 'aeiou':\n            count += 1\n    return count\n\nThis misses uppercase vowels.", "Missing .lower()"),
    ("What's wrong with this Python code?\n\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result\n\nThis misses remaining elements.", "Missing extend"),
    ("What's wrong with this code?\n\ndef binary_search(arr, target):\n    low, high = 0, len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid\n    return -1\n\nThis can infinite loop.", "low = mid + 1"),
]


def extract_number(text: str) -> Optional[float]:
    text = text.strip()
    m = re.search(r'answer\s+is[:\s]+(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    for n_str in reversed(numbers):
        try:
            n = float(n_str.replace(",", ""))
            if abs(n) < 1e6 and abs(n) > 1e-8:
                return n
        except ValueError:
            continue
    return None


def extract_answer(output: str) -> str:
    if not output.strip():
        return ""
    output = output.strip()
    first_line = output.split('\n')[0].strip()
    match = re.search(r'the answer is[:\s]+(.+?)(?:\.|$)', first_line, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return first_line


def check_numeric_match(pred: Optional[float], reference: str, tolerance: float = 0.1) -> bool:
    if pred is None:
        return False
    try:
        ref_num = float(reference)
        return abs(pred - ref_num) < tolerance or abs(pred - ref_num) / max(abs(ref_num), 1e-9) < 0.05
    except ValueError:
        return str(pred).strip() == reference.strip()


def check_text_match(output: str, reference: str) -> bool:
    a = extract_answer(output).lower().strip()
    r = reference.lower().strip()
    return r in a or a in r


def old_majority_vote(outputs: Dict[str, str]) -> str:
    """Old synthesis: simple majority vote."""
    answers = []
    for mid, out in outputs.items():
        ans = extract_answer(out).lower().strip()
        if ans:
            answers.append(ans)
    if not answers:
        return list(outputs.values())[0] if outputs else ""
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def generate_from_model(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result.strip()


def load_7b_models():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    models = {}
    tokenizers = {}

    print("  Loading coder...")
    t0 = time.time()
    code_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B", trust_remote_code=True)
    code_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B", quantization_config=bnb_config,
        device_map="cuda:0", trust_remote_code=True,
    )
    code_model.eval()
    print(f"    Loaded in {time.time()-t0:.1f}s ({torch.cuda.memory_allocated(0)/1e9:.1f} GB)")
    models["coder"] = code_model
    tokenizers["coder"] = code_tokenizer

    print("  Loading reasoning...")
    t0 = time.time()
    reason_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    reason_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb_config,
        device_map="cuda:0", trust_remote_code=True,
    )
    reason_model.eval()
    print(f"    Loaded in {time.time()-t0:.1f}s ({torch.cuda.memory_allocated(0)/1e9:.1f} GB)")
    models["reasoning"] = reason_model
    tokenizers["reasoning"] = reason_tokenizer

    return models, tokenizers


def main():
    print("=" * 80)
    print("7B ENSEMBLE + CONSENSUS-AWARE SYNTHESIS")
    print("Compare old majority vote vs new consensus-aware synthesis")
    print("=" * 80)

    print("\n[1/3] Loading cached 7B models...")
    models, tokenizers = load_7b_models()

    print("\n[2/3] Running benchmark...")
    results = []

    all_prompts = []
    for prompt, ref in MATH_REASONING:
        all_prompts.append({"prompt": prompt, "reference": ref, "category": "math"})
    for prompt, ref in LOGIC_PUZZLES:
        all_prompts.append({"prompt": prompt, "reference": ref, "category": "logic"})
    for prompt, ref in CODE_DEBUGGING:
        all_prompts.append({"prompt": prompt, "reference": ref, "category": "code"})

    for idx, item in enumerate(all_prompts):
        prompt = item["prompt"]
        ref = item["reference"]
        category = item["category"]

        # Generate from both models
        outputs = {}
        for role, model in models.items():
            tokenizer = tokenizers[role]
            output = generate_from_model(model, tokenizer, prompt)
            outputs[role] = output

        # Old synthesis
        old_result = old_majority_vote(outputs)

        # New consensus-aware synthesis
        new_result = consensus_aware_synthesize(prompt, outputs)

        # Compute consensus metrics
        exact_consensus, core_answers = compute_consensus_score(outputs)
        semantic_consensus = compute_semantic_consensus(outputs)

        # Check correctness
        is_numeric = category == "math"
        if is_numeric:
            old_correct = check_numeric_match(extract_number(old_result), ref)
            new_correct = check_numeric_match(extract_number(new_result.answer), ref)
        else:
            old_correct = check_text_match(old_result, ref)
            new_correct = check_text_match(new_result.answer, ref)

        # Per-model correctness
        model_correct = {}
        for role, out in outputs.items():
            if is_numeric:
                model_correct[role] = check_numeric_match(extract_number(out), ref)
            else:
                model_correct[role] = check_text_match(out, ref)

        results.append({
            "prompt": prompt[:100],
            "reference": ref,
            "category": category,
            "exact_consensus": exact_consensus,
            "semantic_consensus": semantic_consensus,
            "model_correct": model_correct,
            "old_result": old_result[:150],
            "old_correct": old_correct,
            "new_strategy": new_result.strategy_used.value,
            "new_result": new_result.answer[:150],
            "new_correct": new_correct,
            "needs_review": new_result.needs_review,
        })

        status_old = "PASS" if old_correct else "FAIL"
        status_new = "PASS" if new_correct else "FAIL"
        print(f"  [{idx+1}/{len(all_prompts)}] {category:6s} | exact={exact_consensus:.2f} semantic={semantic_consensus:.2f} | old={status_old} new={status_new} | strategy={new_result.strategy_used.value}")

    # Compute metrics
    print("\n[3/3] Computing metrics...")

    old_correct_count = sum(1 for r in results if r["old_correct"])
    new_correct_count = sum(1 for r in results if r["new_correct"])

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\n  Overall Accuracy:")
    print(f"    Old synthesis (majority vote): {old_correct_count}/{len(results)} ({old_correct_count/len(results):.1%})")
    print(f"    New synthesis (consensus-aware): {new_correct_count}/{len(results)} ({new_correct_count/len(results):.1%})")

    for cat in ["math", "logic", "code"]:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        old_acc = sum(1 for r in cat_results if r["old_correct"]) / len(cat_results)
        new_acc = sum(1 for r in cat_results if r["new_correct"]) / len(cat_results)
        print(f"\n    {cat.upper()} ({len(cat_results)} prompts):")
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

    # Consensus analysis
    exact_scores = [r["exact_consensus"] for r in results]
    semantic_scores = [r["semantic_consensus"] for r in results]
    print(f"\n  Consensus Scores:")
    print(f"    Exact: min={min(exact_scores):.2f} max={max(exact_scores):.2f} avg={sum(exact_scores)/len(exact_scores):.2f}")
    print(f"    Semantic: min={min(semantic_scores):.2f} max={max(semantic_scores):.2f} avg={sum(semantic_scores)/len(semantic_scores):.2f}")

    # Verdict
    print("\n" + "=" * 80)
    if new_correct_count > old_correct_count:
        print("VERDICT: PASS — Consensus-aware synthesis improves accuracy")
    elif new_correct_count == old_correct_count:
        print("VERDICT: NEUTRAL — Consensus-aware synthesis matches old synthesis")
    else:
        print("VERDICT: REGRESSION — Consensus-aware synthesis hurts accuracy")
    print("=" * 80)

    output_dir = Path("results/consensus_synthesis_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({"results": results}, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
