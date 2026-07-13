"""
7B Ensemble Hard Reasoning Benchmark

Test: Does the ensemble beat the best single model with 7B workers?

Models:
- Code specialist: Qwen2.5-Coder-7B (genuine code capability)
- Reasoning specialist: Qwen2.5-7B-Instruct (strong generalist)
- Math specialist: Qwen2.5-Math-7B (dedicated math training)
- Judge: Qwen2.5-7B-Instruct (quality evaluation)

All models loaded in 4-bit quantization (~3.5 GB each).
Total: ~14 GB workers + 3.5 GB judge = ~17.5 GB (fits RTX 4090).
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Hard Reasoning Prompts ────────────────────────────────────────────────────

MATH_REASONING = [
    ("A train travels at 60 mph for 2 hours, then 40 mph for 3 hours. What is the average speed?",
     "48"),
    ("If a rectangle has length 12 and width 5, and a triangle has base 12 and height 5, what is the ratio of their areas?",
     "2"),
    ("You buy 3 shirts at $15 each and 2 pairs of pants at $25 each. You get a 20% discount. How much do you pay?",
     "76"),
    ("A pool is filled by pipe A in 6 hours and pipe B in 4 hours. How long to fill the pool with both open?",
     "2.4"),
    ("If you invest $1000 at 5% annual compound interest, how much do you have after 3 years?",
     "1157.63"),
    ("A car uses 8 liters per 100km. Gas costs $1.50 per liter. How much does a 240km trip cost?",
     "28.80"),
    ("Three workers can build a wall in 4 days. How many workers are needed to build it in 1 day?",
     "12"),
    ("A bakery sells cupcakes for $2.50 each. They sell 40 per day. How many days to earn $500?",
     "5"),
    ("If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
     "5"),
    ("You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?",
     "4"),
]

LOGIC_PUZZLES = [
    ("If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
     "No"),
    ("A farmer has 17 sheep. All but 9 die. How many are left?",
     "9"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
     "5 minutes"),
    ("Two coins add up to 30 cents. One of them is not a nickel. What are the two coins?",
     "quarter and nickel"),
    ("A doctor gives you 3 pills and says to take one every 30 minutes. How long do they last?",
     "60 minutes"),
    ("If you have a bowl with six apples and you take away four, how many do YOU have?",
     "4"),
    ("What comes next: 1, 1, 2, 3, 5, 8, ...?",
     "13"),
    ("A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
     "0.05"),
]

CODE_DEBUGGING = [
    ("What's wrong with this Python code?\n\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n)\n\nThe function causes infinite recursion.",
     "Missing n-1"),
    ("What's wrong with this Python code?\n\ndef find_max(lst):\n    max_val = 0\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val\n\nThis fails for all-negative lists.",
     "max_val should start as lst[0]"),
    ("What's wrong with this Python code?\n\ndef count_vowels(s):\n    count = 0\n    for char in s:\n        if char in 'aeiou':\n            count += 1\n    return count\n\nThis misses uppercase vowels.",
     "Missing .lower()"),
    ("What's wrong with this Python code?\n\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result\n\nThis misses remaining elements.",
     "Missing extend"),
    ("What's wrong with this code?\n\ndef binary_search(arr, target):\n    low, high = 0, len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid\n    return -1\n\nThis can infinite loop.",
     "low = mid + 1"),
]


def extract_number(text: str) -> Optional[float]:
    """Extract the final numeric answer from model output."""
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
    """Extract the core answer from model output."""
    if not output.strip():
        return ""
    output = output.strip()
    first_line = output.split('\n')[0].strip()
    match = re.search(r'the answer is[:\s]+(.+?)(?:\.|$)', first_line, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return first_line


def check_numeric_match(pred: Optional[float], reference: str, tolerance: float = 0.1) -> bool:
    """Check if predicted number matches reference (fuzzy)."""
    if pred is None:
        return False
    try:
        ref_num = float(reference)
        return abs(pred - ref_num) < tolerance or abs(pred - ref_num) / max(abs(ref_num), 1e-9) < 0.05
    except ValueError:
        return str(pred).strip() == reference.strip()


def check_text_match(output: str, reference: str) -> bool:
    """Check if text output matches reference (fuzzy)."""
    a = extract_answer(output).lower().strip()
    r = reference.lower().strip()
    return r in a or a in r


# ── Ensemble Methods ─────────────────────────────────────────────────────────

def majority_vote(outputs: Dict[str, str]) -> str:
    """Pick the most common answer via majority vote."""
    answers = []
    for mid, out in outputs.items():
        ans = extract_answer(out).lower().strip()
        if ans:
            answers.append(ans)
    if not answers:
        return list(outputs.values())[0] if outputs else ""
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def similarity_weighted(outputs: Dict[str, str]) -> str:
    """Weight answers by similarity to other models' answers."""
    answers = {}
    for mid, out in outputs.items():
        ans = extract_answer(out).lower().strip()
        if ans:
            answers[mid] = ans
    if not answers:
        return list(outputs.values())[0] if outputs else ""

    scores = {}
    for mid, ans in answers.items():
        agreement = sum(1 for other_ans in answers.values() if ans in other_ans or other_ans in ans)
        scores[mid] = agreement

    best_mid = max(scores, key=scores.get)
    return answers[best_mid]


def judge_synthesize(outputs: Dict[str, str], judge_model, judge_tokenizer) -> str:
    """Use judge to pick the best answer."""
    prompt_parts = []
    for mid, out in outputs.items():
        prompt_parts.append(f"[{mid}]: {out[:300]}")

    judge_prompt = f"""Given these answers from different models, pick the best one and output ONLY the answer text (no explanation, no "The answer is", just the answer):

{chr(10).join(prompt_parts)}

Output the best answer:"""

    messages = [{"role": "user", "content": judge_prompt}]
    text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tokenizer(text, return_tensors="pt").to(judge_model.device)

    with torch.no_grad():
        output = judge_model.generate(**inputs, max_new_tokens=64, do_sample=False)

    result = judge_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result.strip()


# ── Model Loading ────────────────────────────────────────────────────────────

def load_7b_ensemble():
    """Load 4-bit quantized 7B models."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_configs = {
        "coder": "Qwen/Qwen2.5-Coder-7B",
        "reasoning": "Qwen/Qwen2.5-7B-Instruct",
        "math": "Qwen/Qwen2.5-Math-7B",
    }

    models = {}
    tokenizers = {}

    for role, hf_name in model_configs.items():
        print(f"  Loading {role} ({hf_name})...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            quantization_config=bnb_config,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model.eval()
        elapsed = time.time() - t0
        vram = torch.cuda.memory_allocated(0) / 1e9
        print(f"    Loaded in {elapsed:.1f}s ({vram:.1f} GB VRAM)")
        models[role] = model
        tokenizers[role] = tokenizer

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


# ── Benchmark Runner ────────────────────────────────────────────────────────

def run_7b_benchmark(models, tokenizers):
    """Run the hard reasoning benchmark with 7B models."""
    results = []

    all_prompts = []
    for prompt, ref in MATH_REASONING:
        all_prompts.append({"prompt": prompt, "reference": ref, "category": "math"})
    for prompt, ref in LOGIC_PUZZLES:
        all_prompts.append({"prompt": prompt, "reference": ref, "category": "logic"})
    for prompt, ref in CODE_DEBUGGING:
        all_prompts.append({"prompt": prompt, "reference": ref, "category": "code"})

    print(f"\nRunning 7B ensemble benchmark ({len(all_prompts)} prompts)...")

    for idx, item in enumerate(all_prompts):
        prompt = item["prompt"]
        ref = item["reference"]
        category = item["category"]

        # Generate from all models
        all_outputs = {}
        for role, model in models.items():
            tokenizer = tokenizers[role]
            output = generate_from_model(model, tokenizer, prompt)
            all_outputs[role] = output

        # Apply ensemble methods
        majority = majority_vote(all_outputs)
        sim_weighted = similarity_weighted(all_outputs)

        # Judge synthesis (using the reasoning model as judge)
        judge_model = models["reasoning"]
        judge_tokenizer = tokenizers["reasoning"]
        synthesized = judge_synthesize(all_outputs, judge_model, judge_tokenizer)

        # Check correctness
        is_numeric = category == "math"
        accuracy = {}
        for role, out in all_outputs.items():
            if is_numeric:
                accuracy[role] = check_numeric_match(extract_number(out), ref)
            else:
                accuracy[role] = check_text_match(out, ref)

        if is_numeric:
            accuracy["majority"] = check_numeric_match(extract_number(majority), ref)
            accuracy["sim_weighted"] = check_numeric_match(extract_number(sim_weighted), ref)
            accuracy["synthesized"] = check_numeric_match(extract_number(synthesized), ref)
        else:
            accuracy["majority"] = check_text_match(majority, ref)
            accuracy["sim_weighted"] = check_text_match(sim_weighted, ref)
            accuracy["synthesized"] = check_text_match(synthesized, ref)

        results.append({
            "prompt": prompt[:150],
            "reference": ref,
            "category": category,
            "outputs": {k: v[:200] for k, v in all_outputs.items()},
            "ensemble_outputs": {
                "majority": majority[:200],
                "sim_weighted": sim_weighted[:200],
                "synthesized": synthesized[:200],
            },
            "accuracy": accuracy,
        })

        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1}/{len(all_prompts)}] completed")

    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute overall metrics."""
    all_roles = list(results[0]["outputs"].keys()) if results else []

    # Per-category accuracy
    category_accuracy = {}
    for cat in ["math", "logic", "code"]:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue

        model_acc = {}
        for role in all_roles:
            correct = sum(1 for r in cat_results if r["accuracy"].get(role, False))
            model_acc[role] = round(correct / len(cat_results), 4) if cat_results else 0

        ensemble_acc = {}
        for method in ["majority", "sim_weighted", "synthesized"]:
            correct = sum(1 for r in cat_results if r["accuracy"].get(method, False))
            ensemble_acc[method] = round(correct / len(cat_results), 4) if cat_results else 0

        category_accuracy[cat] = {
            "model_accuracy": model_acc,
            "ensemble_accuracy": ensemble_acc,
            "num_prompts": len(cat_results),
        }

    # Overall accuracy
    model_accuracy = {}
    for role in all_roles:
        correct = sum(1 for r in results if r["accuracy"].get(role, False))
        model_accuracy[role] = round(correct / len(results), 4) if results else 0

    ensemble_accuracy = {}
    for method in ["majority", "sim_weighted", "synthesized"]:
        correct = sum(1 for r in results if r["accuracy"].get(method, False))
        ensemble_accuracy[method] = round(correct / len(results), 4) if results else 0

    # Ensemble vs best single (per prompt)
    best_single_wins = 0
    ensemble_wins = 0
    ties = 0

    for r in results:
        best_model_acc = max(r["accuracy"].get(role, False) for role in all_roles)
        ensemble_acc = r["accuracy"].get("synthesized", False)

        if ensemble_acc and not best_model_acc:
            ensemble_wins += 1
        elif best_model_acc and not ensemble_acc:
            best_single_wins += 1
        else:
            ties += 1

    total = best_single_wins + ensemble_wins + ties

    return {
        "overall": {
            "model_accuracy": model_accuracy,
            "ensemble_accuracy": ensemble_accuracy,
            "ensemble_vs_best_single": {
                "ensemble_wins": ensemble_wins,
                "best_single_wins": best_single_wins,
                "ties": ties,
                "ensemble_win_rate": round(ensemble_wins / total, 4) if total > 0 else 0,
            },
        },
        "by_category": category_accuracy,
        "total_prompts": len(results),
    }


def main():
    print("=" * 80)
    print("7B ENSEMBLE HARD REASONING BENCHMARK")
    print("Does the ensemble beat the best single model with 7B workers?")
    print("=" * 80)

    # Load models
    print("\n[1/3] Loading 4-bit quantized 7B models...")
    models, tokenizers = load_7b_ensemble()

    # Run benchmark
    print("\n[2/3] Running benchmark...")
    results = run_7b_benchmark(models, tokenizers)

    # Compute metrics
    print("\n[3/3] Computing metrics...")
    metrics = compute_metrics(results)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    overall = metrics["overall"]

    print("\n  Overall Model Accuracy:")
    for role, acc in overall["model_accuracy"].items():
        print(f"    {role:15s}: {acc:.1%}")

    print("\n  Overall Ensemble Accuracy:")
    for method, acc in overall["ensemble_accuracy"].items():
        print(f"    {method:15s}: {acc:.1%}")

    print("\n  Ensemble vs Best Single Model (synthesized):")
    vs = overall["ensemble_vs_best_single"]
    print(f"    Ensemble wins:    {vs['ensemble_wins']}/{metrics['total_prompts']} ({vs['ensemble_win_rate']:.1%})")
    print(f"    Best single wins: {vs['best_single_wins']}/{metrics['total_prompts']}")
    print(f"    Ties:             {vs['ties']}/{metrics['total_prompts']}")

    print("\n  By Category:")
    for cat, cat_data in metrics["by_category"].items():
        print(f"\n    {cat.upper()} ({cat_data['num_prompts']} prompts):")
        for role, acc in cat_data["model_accuracy"].items():
            print(f"      {role:15s}: {acc:.1%}")
        print(f"      {'synthesized':15s}: {cat_data['ensemble_accuracy']['synthesized']:.1%}")

    # Verdict
    print("\n" + "=" * 80)
    if vs["ensemble_win_rate"] > 0.6:
        print("VERDICT: PASS — Ensemble beats best single model >60%")
        print("Diversity matters at scale. Proceed to Phase 2 with 7B ensemble.")
    elif vs["ensemble_win_rate"] > 0.5:
        print("VERDICT: MARGINAL — Ensemble beats best single model 50-60%")
        print("Consider single 7B + specialists only for security.")
    elif vs["ensemble_win_rate"] > 0.4:
        print("VERDICT: NEUTRAL — Ensemble roughly equal")
        print("Test with more diverse models before concluding.")
    else:
        print("VERDICT: WEAK — Ensemble does NOT beat best single")
        print("Architecture may be fundamentally flawed.")
    print("=" * 80)

    # Save results
    output_dir = Path("results/7b_ensemble_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
