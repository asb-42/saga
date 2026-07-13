"""
Code Debugging Benchmark — Using Already-Cached Models

Models:
- Coder: Qwen2.5-Coder-7B (base, already cached)
- Reasoning: Qwen2.5-7B-Instruct (already cached)
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


def extract_answer(output: str) -> str:
    if not output.strip():
        return ""
    output = output.strip()
    first_line = output.split('\n')[0].strip()
    match = re.search(r'the answer is[:\s]+(.+?)(?:\.|$)', first_line, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return first_line


def check_code_match(output: str, reference: str) -> bool:
    a = output.lower().strip()
    r = reference.lower().strip()
    if r in a:
        return True
    key_concepts = {
        "Missing n-1": ["n-1", "n - 1", "factorial(n-1)", "factorial(n - 1)"],
        "max_val should start as lst[0]": ["lst[0]", "first element", "initial"],
        "Missing .lower()": ["lower()", "case", "uppercase"],
        "Missing extend": ["extend", "remaining", "leftover"],
        "low = mid + 1": ["mid + 1", "mid+1", "progress"],
    }
    for concept in key_concepts.get(reference, []):
        if concept in a:
            return True
    return False


def generate_from_model(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result.strip()


def judge_synthesize(outputs: Dict[str, str], judge_model, judge_tokenizer) -> str:
    """Use new consensus-aware synthesis."""
    from src.meta_model.synthesis import consensus_aware_synthesize
    result = consensus_aware_synthesize("What is wrong with this code?", outputs)
    return result.answer


def main():
    print("=" * 80)
    print("CODE DEBUGGING BENCHMARK (Cached Models)")
    print("=" * 80)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n[1/3] Loading cached 4-bit models...")
    models = {}
    tokenizers = {}

    # Coder (base, already cached)
    print("  Loading coder (Qwen2.5-Coder-7B)...")
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

    # Reasoning (already cached)
    print("  Loading reasoning (Qwen2.5-7B-Instruct)...")
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

    print("\n[2/3] Running code debugging benchmark...")
    results = []

    for idx, (prompt, ref) in enumerate(CODE_DEBUGGING):
        print(f"\n  Prompt {idx+1}: {prompt[:80]}...")

        all_outputs = {}
        for role, model in models.items():
            tokenizer = tokenizers[role]
            output = generate_from_model(model, tokenizer, prompt)
            all_outputs[role] = output
            print(f"    {role}: {output[:120]}...")

        synthesized = judge_synthesize(all_outputs, models["reasoning"], tokenizers["reasoning"])
        print(f"    synthesized: {synthesized[:120]}...")

        accuracy = {}
        for role, out in all_outputs.items():
            accuracy[role] = check_code_match(out, ref)
        accuracy["synthesized"] = check_code_match(synthesized, ref)

        results.append({
            "prompt": prompt[:150],
            "reference": ref,
            "outputs": {k: v[:200] for k, v in all_outputs.items()},
            "synthesized": synthesized[:200],
            "accuracy": accuracy,
        })

        print(f"    Correct: coder={accuracy['coder']}, reasoning={accuracy['reasoning']}, synthesized={accuracy['synthesized']}")

    # Compute metrics
    print("\n[3/3] Computing metrics...")

    model_accuracy = {}
    for role in ["coder", "reasoning"]:
        correct = sum(1 for r in results if r["accuracy"].get(role, False))
        model_accuracy[role] = round(correct / len(results), 4)

    synthesized_accuracy = sum(1 for r in results if r["accuracy"].get("synthesized", False)) / len(results)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\n  Model Accuracy (code debugging):")
    for role, acc in model_accuracy.items():
        print(f"    {role:15s}: {acc:.1%}")
    print(f"    {'synthesized':15s}: {synthesized_accuracy:.1%}")

    print("\n  Per-Prompt Results:")
    for i, r in enumerate(results):
        status = "PASS" if r["accuracy"].get("synthesized", False) else "FAIL"
        print(f"    {i+1}. [{status}] {r['reference']}")

    print("\n" + "=" * 80)
    best_single = max(model_accuracy.values())
    if synthesized_accuracy > best_single:
        print("VERDICT: PASS — Ensemble beats best single on code debugging")
    elif synthesized_accuracy == best_single:
        print("VERDICT: NEUTRAL — Ensemble matches best single on code debugging")
    elif synthesized_accuracy > 0:
        print("VERDICT: IMPROVED — Code debugging better than 0%")
    else:
        print("VERDICT: WEAK — Code debugging still fails")
    print("=" * 80)

    output_dir = Path("results/code_debugging_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({"results": results, "model_accuracy": model_accuracy, "synthesized_accuracy": synthesized_accuracy}, f, indent=2)
    print(f"\nResults saved to {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
