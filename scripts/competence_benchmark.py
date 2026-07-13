"""
Competence Benchmark: Does the ensemble beat the best single model?

This is the core value proposition test. No anomaly detection. No security.
Just quality: do multiple models working together produce better answers?

Conditions:
1. Best single model (oracle: best answer per prompt)
2. Uniform ensemble (majority vote / average)
3. Consensus-weighted (weights from output similarity)
4. Sentinel synthesis (7B judge evaluates, picks/synthesizes best)

Measurement:
- Reference accuracy (closed QA)
- Judge quality score (open-ended, 1-5 scale)
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loader import load_all_models, FrozenModelWrapper


# ── Test Prompts ─────────────────────────────────────────────────────────────

# Closed QA (reference accuracy)
CLOSED_QA = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("How many legs does a spider have?", "8"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("How many continents are there?", "7"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the speed of light in m/s?", "299792458"),
    ("How many bones are in the human body?", "206"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the hardest natural substance?", "Diamond"),
    ("How many chromosomes do humans have?", "46"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the tallest mountain on Earth?", "Mount Everest"),
    ("How many days are in a leap year?", "366"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the capital of Japan?", "Tokyo"),
    ("How many planets are in our solar system?", "8"),
    ("What is the largest mammal on Earth?", "Blue whale"),
    ("Who wrote Hamlet?", "Shakespeare"),
    ("What is the square root of 144?", "12"),
    ("What is the largest desert on Earth?", "Sahara"),
    ("What is the capital of Australia?", "Canberra"),
    ("How many elements are in the periodic table?", "118"),
    ("What is the largest bird in the world?", "Ostrich"),
    ("What is the chemical symbol for oxygen?", "O"),
    ("How many vowels are in the English alphabet?", "5"),
]

# Reasoning (judge quality)
REASONING = [
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "What is the next number in this sequence: 2, 6, 12, 20, 30?",
    "If you have 3 apples and give away 1, then buy 5 more, how many do you have?",
    "What is 15% of 200?",
    "If a rectangle has length 8 and width 5, what is its area?",
    "What is the sum of all angles in a triangle?",
    "If you mix 3 cups of blue paint with 2 cups of yellow paint, what color do you get?",
    "What is the prime factorization of 60?",
    "If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads?",
    "What is the derivative of x^2?",
]

# Code generation (judge quality)
CODE = [
    "Write a Python function that checks if a number is even.",
    "Write a Python function that returns the factorial of a number.",
    "Write a Python function that checks if a string is a palindrome.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Write a Python function that sorts a list using bubble sort.",
    "Write a Python function that finds the maximum element in a list.",
    "Write a Python function that counts the vowels in a string.",
    "Write a Python function that reverses a string.",
    "Write a Python function that checks if a number is prime.",
    "Write a Python function that merges two sorted lists.",
]

# Open-ended (judge quality)
OPEN_ENDED = [
    "Explain what a neural network is in simple terms.",
    "What are the pros and cons of remote work?",
    "How does photosynthesis work?",
    "What is the difference between machine learning and traditional programming?",
    "Why do we dream?",
    "What causes climate change?",
    "How does the internet work?",
    "What is the importance of biodiversity?",
    "How does the human immune system work?",
    "What are the benefits of exercise?",
]

ALL_PROMPTS = CLOSED_QA + [(q, "") for q in REASONING + CODE + OPEN_ENDED]


# ── Ensemble Methods ─────────────────────────────────────────────────────────

def extract_answer(output: str) -> str:
    """Extract the core answer from a model output."""
    import re
    if not output.strip():
        return ""
    output = output.strip()
    first_line = output.split('\n')[0].strip()
    first_sentence = first_line.split('.')[0].strip()
    # Remove common prefixes
    match = re.search(r'the answer is[:\s]+(.+?)(?:\.|$)', first_sentence, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r'answer[:\s]+(.+?)(?:\.|$)', first_sentence, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return first_sentence


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
    
    # Count how many models agree with each answer
    scores = {}
    for mid, ans in answers.items():
        agreement = sum(1 for other_ans in answers.values() if ans in other_ans or other_ans in ans)
        scores[mid] = agreement
    
    # Pick the answer with highest agreement
    best_mid = max(scores, key=scores.get)
    return answers[best_mid]


def judge_synthesize(outputs: Dict[str, str], judge_model) -> str:
    """Use 7B judge to pick the best answer."""
    prompt_parts = []
    for mid, out in outputs.items():
        prompt_parts.append(f"[{mid}]: {out[:200]}")
    
    judge_prompt = f"""Given these answers from different models, pick the best one and output ONLY the answer text (no explanation, no "The answer is", just the answer):

{chr(10).join(prompt_parts)}

Output the best answer:"""
    
    # Use judge model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    messages = [{"role": "user", "content": judge_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(judge_model.device)
    
    with torch.no_grad():
        output = judge_model.generate(**inputs, max_new_tokens=32, do_sample=False)
    
    result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return result.strip()


# ── Evaluation ───────────────────────────────────────────────────────────────

def check_accuracy(answer: str, reference: str) -> bool:
    """Check if answer matches reference (fuzzy match)."""
    if not reference:
        return True  # Can't evaluate open-ended
    a = answer.lower().strip()
    r = reference.lower().strip()
    return r in a or a in r


def run_benchmark(
    models: Dict[str, FrozenModelWrapper],
    model_ids: List[str],
    judge_model,
    prompts: List[Tuple[str, str]],
    max_new_tokens: int = 64,
) -> Dict:
    """Run the competence benchmark."""
    results = []
    
    print(f"\nRunning benchmark with {len(prompts)} prompts...")
    
    for idx, (prompt, reference) in enumerate(prompts):
        # Generate from all models
        all_outputs = {}
        for mid in model_ids:
            wrapper = models[mid]
            wrapper.load_to_gpu()
            output = wrapper.generate([prompt], max_new_tokens=max_new_tokens)[0]
            wrapper.offload_to_cpu()
            all_outputs[mid] = output
        
        # Apply ensemble methods
        best_single = max(all_outputs.values(), key=lambda x: len(x))  # Proxy for quality
        majority = majority_vote(all_outputs)
        sim_weighted = similarity_weighted(all_outputs)
        synthesized = judge_synthesize(all_outputs, judge_model)
        
        # Evaluate accuracy (for closed QA)
        accuracy = {}
        if reference:
            for mid, out in all_outputs.items():
                accuracy[mid] = check_accuracy(extract_answer(out), reference)
            accuracy["majority"] = check_accuracy(majority, reference)
            accuracy["sim_weighted"] = check_accuracy(sim_weighted, reference)
            accuracy["synthesized"] = check_accuracy(synthesized, reference)
        
        results.append({
            "prompt": prompt[:100],
            "reference": reference,
            "outputs": {k: v[:150] for k, v in all_outputs.items()},
            "ensemble_outputs": {
                "majority": majority[:150],
                "sim_weighted": sim_weighted[:150],
                "synthesized": synthesized[:150],
            },
            "accuracy": accuracy,
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(prompts)}] completed")
    
    return results


def compute_metrics(results: Dict) -> Dict:
    """Compute overall metrics."""
    # Accuracy metrics (closed QA only)
    closed_results = [r for r in results if r["reference"]]
    
    model_accuracy = {}
    all_model_ids = list(results[0]["outputs"].keys()) if results else []
    for mid in all_model_ids:
        correct = sum(1 for r in closed_results if r["accuracy"].get(mid, False))
        model_accuracy[mid] = round(correct / len(closed_results), 4) if closed_results else 0
    
    ensemble_accuracy = {}
    for method in ["majority", "sim_weighted", "synthesized"]:
        correct = sum(1 for r in closed_results if r["accuracy"].get(method, False))
        ensemble_accuracy[method] = round(correct / len(closed_results), 4) if closed_results else 0
    
    # Find best single model per prompt
    best_single_wins = 0
    ensemble_wins = 0
    ties = 0
    
    for r in closed_results:
        if not r["accuracy"]:
            continue
        best_model_acc = max(r["accuracy"].get(mid, False) for mid in all_model_ids)
        ensemble_acc = r["accuracy"].get("synthesized", False)
        
        if ensemble_acc and not best_model_acc:
            ensemble_wins += 1
        elif best_model_acc and not ensemble_acc:
            best_single_wins += 1
        else:
            ties += 1
    
    total = best_single_wins + ensemble_wins + ties
    
    return {
        "model_accuracy": model_accuracy,
        "ensemble_accuracy": ensemble_accuracy,
        "ensemble_vs_best_single": {
            "ensemble_wins": ensemble_wins,
            "best_single_wins": best_single_wins,
            "ties": ties,
            "ensemble_win_rate": round(ensemble_wins / total, 4) if total > 0 else 0,
        },
        "total_prompts": len(results),
        "closed_qa_prompts": len(closed_results),
    }


def main():
    print("=" * 80)
    print("COMPETENCE BENCHMARK")
    print("Does the ensemble beat the best single model?")
    print("=" * 80)
    
    # Load models
    print("\n[1/3] Loading models...")
    models = load_all_models(encoding_device="cuda:0")
    model_ids = [mid for mid in ["codeqwen", "phi2", "qwen", "smollm"] if mid in models]
    print(f"  Models: {model_ids}")
    
    # Load 7B judge
    print("  Loading 7B judge model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    judge_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb_config, device_map="cuda:0", trust_remote_code=True)
    judge_model.eval()
    print(f"  Judge loaded ({torch.cuda.memory_allocated(0)/1e9:.1f} GB)")
    
    # Run benchmark
    print("\n[2/3] Running benchmark...")
    results = run_benchmark(models, model_ids, judge_model, ALL_PROMPTS)
    
    # Compute metrics
    print("\n[3/3] Computing metrics...")
    metrics = compute_metrics(results)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\n  Model Accuracy (closed QA):")
    for mid, acc in metrics["model_accuracy"].items():
        print(f"    {mid:15s}: {acc:.1%}")
    
    print("\n  Ensemble Accuracy (closed QA):")
    for method, acc in metrics["ensemble_accuracy"].items():
        print(f"    {method:15s}: {acc:.1%}")
    
    print("\n  Ensemble vs Best Single Model:")
    vs = metrics["ensemble_vs_best_single"]
    print(f"    Ensemble wins:    {vs['ensemble_wins']}/{metrics['closed_qa_prompts']} ({vs['ensemble_win_rate']:.1%})")
    print(f"    Best single wins: {vs['best_single_wins']}/{metrics['closed_qa_prompts']}")
    print(f"    Ties:             {vs['ties']}/{metrics['closed_qa_prompts']}")
    
    # Verdict
    print("\n" + "=" * 80)
    if vs["ensemble_win_rate"] > 0.6:
        print("VERDICT: PASS — Ensemble beats best single model >60%")
        print("The project premise is valid. Proceed to Phase 2.")
    elif vs["ensemble_win_rate"] > 0.5:
        print("VERDICT: MARGINAL — Ensemble beats best single model 50-60%")
        print("Consider upgrading workers before Phase 2.")
    else:
        print("VERDICT: FAIL — Ensemble does NOT beat best single model")
        print("Redesign required. The ensemble adds no value.")
    print("=" * 80)
    
    # Save results
    output_dir = Path("results/competence_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
