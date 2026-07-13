"""
Code Debugging Specialist Benchmark

Test: Does a code specialist fix the 0% code failure?

Models:
- Code specialist: Qwen2.5-Coder-7B-Instruct (genuine code capability)
- Reasoning specialist: Qwen2.5-7B-Instruct (general reasoning)

Test on 5 code debugging prompts.
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


# ── Code Debugging Prompts ───────────────────────────────────────────────────

CODE_DEBUGGING = [
    ("What's wrong with this Python code?\n\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n)\n\nThe function causes infinite recursion.",
     "Missing n-1",
     "The recursive call should be factorial(n-1), not factorial(n)."),

    ("What's wrong with this Python code?\n\ndef find_max(lst):\n    max_val = 0\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val\n\nThis fails for all-negative lists.",
     "max_val should start as lst[0]",
     "Initializing max_val = 0 means negative numbers are never found as max."),

    ("What's wrong with this Python code?\n\ndef count_vowels(s):\n    count = 0\n    for char in s:\n        if char in 'aeiou':\n            count += 1\n    return count\n\nThis misses uppercase vowels.",
     "Missing .lower()",
     "The code only checks lowercase 'aeiou'. Should use s.lower() or add 'AEIOU'."),

    ("What's wrong with this Python code?\n\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result\n\nThis misses remaining elements.",
     "Missing extend",
     "After the loop, remaining elements from a or b need to be appended."),

    ("What's wrong with this code?\n\ndef binary_search(arr, target):\n    low, high = 0, len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid\n    return -1\n\nThis can infinite loop.",
     "low = mid + 1",
     "When arr[mid] < target, low should be mid+1, not mid, to make progress."),
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


def check_code_match(output: str, reference: str, explanation: str) -> bool:
    """Check if code answer matches reference (fuzzy)."""
    a = output.lower().strip()
    r = reference.lower().strip()
    e = explanation.lower().strip()

    # Check if the key concept is mentioned
    if r in a:
        return True

    # Check for partial matches
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
    prompt_parts = []
    for mid, out in outputs.items():
        prompt_parts.append(f"[{mid}]: {out[:300]}")
    judge_prompt = f"""Given these answers from different models about code debugging, pick the best one and output ONLY the answer text (no explanation, no "The answer is", just the answer):

{chr(10).join(prompt_parts)}

Output the best answer:"""
    messages = [{"role": "user", "content": judge_prompt}]
    text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tokenizer(text, return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        output = judge_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    result = judge_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result.strip()


def main():
    print("=" * 80)
    print("CODE DEBUGGING SPECIALIST BENCHMARK")
    print("Does a code specialist fix the 0% code failure?")
    print("=" * 80)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load models
    print("\n[1/3] Loading 4-bit quantized models...")
    models = {}
    tokenizers = {}

    # Code specialist
    print("  Loading code specialist (Qwen2.5-Coder-7B-Instruct)...")
    t0 = time.time()
    code_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)
    code_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    code_model.eval()
    print(f"    Loaded in {time.time()-t0:.1f}s ({torch.cuda.memory_allocated(0)/1e9:.1f} GB)")
    models["coder"] = code_model
    tokenizers["coder"] = code_tokenizer

    # Reasoning specialist
    print("  Loading reasoning specialist (Qwen2.5-7B-Instruct)...")
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

    # Run benchmark
    print("\n[2/3] Running code debugging benchmark...")
    results = []

    for idx, (prompt, ref, explanation) in enumerate(CODE_DEBUGGING):
        print(f"\n  Prompt {idx+1}: {prompt[:80]}...")

        # Generate from all models
        all_outputs = {}
        for role, model in models.items():
            tokenizer = tokenizers[role]
            output = generate_from_model(model, tokenizer, prompt)
            all_outputs[role] = output
            print(f"    {role}: {output[:100]}...")

        # Judge synthesis
        synthesized = judge_synthesize(all_outputs, models["reasoning"], tokenizers["reasoning"])
        print(f"    synthesized: {synthesized[:100]}...")

        # Check correctness
        accuracy = {}
        for role, out in all_outputs.items():
            accuracy[role] = check_code_match(out, ref, explanation)
        accuracy["synthesized"] = check_code_match(synthesized, ref, explanation)

        results.append({
            "prompt": prompt[:150],
            "reference": ref,
            "explanation": explanation,
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

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\n  Model Accuracy (code debugging):")
    for role, acc in model_accuracy.items():
        print(f"    {role:15s}: {acc:.1%}")
    print(f"    {'synthesized':15s}: {synthesized_accuracy:.1%}")

    print("\n  Per-Prompt Results:")
    for i, r in enumerate(results):
        status = "✓" if r["accuracy"].get("synthesized", False) else "✗"
        print(f"    {i+1}. {status} {r['reference']}")

    # Verdict
    print("\n" + "=" * 80)
    best_single = max(model_accuracy.values())
    if synthesized_accuracy > best_single:
        print("VERDICT: PASS — Ensemble beats best single on code debugging")
    elif synthesized_accuracy == best_single:
        print("VERDICT: NEUTRAL — Ensemble matches best single on code debugging")
    elif synthesized_accuracy > 0:
        print("VERDICT: IMPROVED — Ensemble better than before (0% → {:.1%})".format(synthesized_accuracy))
    else:
        print("VERDICT: WEAK — Code debugging still fails")
    print("=" * 80)

    # Save results
    output_dir = Path("results/code_debugging_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({"results": results, "model_accuracy": model_accuracy, "synthesized_accuracy": synthesized_accuracy}, f, indent=2)
    print(f"\nResults saved to {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
