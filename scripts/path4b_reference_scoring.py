"""
Path 4b: Output-Based Routing with Reference-Based Scoring

Instead of asking judge to evaluate quality (which has bias), we compare
each model's output against the ground truth answer directly.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loader import load_all_models
from src.meta_model.judge import SynthesisJudge


REFERENCE_PROMPT = """You are an answer evaluator. Compare the model's answer to the correct answer.

Question: {question}
Model Answer: {model_answer}
Correct Answer: {ground_truth}

Is the model answer correct? Consider:
- Exact match
- Paraphrase or equivalent meaning
- Partial correctness

Respond with ONLY:
- "YES" if the model answer is correct or mostly correct
- "NO" if the model answer is wrong or significantly different
- "PARTIAL" if partially correct

Do not explain."""


def extract_ground_truth(answer: str) -> str:
    """Extract the core answer from a model output for comparison."""
    answer = re.sub(r'^(the answer is|the correct answer is|answer:)\s*', '', answer.lower().strip())
    answer = answer.rstrip('.')
    return answer


def reference_score(judge: SynthesisJudge, question: str, model_answer: str, 
                   ground_truth: str, device: str) -> int:
    """Score answer by comparing to ground truth. Returns 1-5."""
    if not model_answer.strip():
        return 1
    
    prompt = REFERENCE_PROMPT.format(
        question=question,
        model_answer=model_answer,
        ground_truth=ground_truth,
    )
    
    judge.model.to(device)
    judge_inputs = judge.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048,
    )
    judge_inputs = {k: v.to(device) for k, v in judge_inputs.items()}
    
    with torch.no_grad():
        outputs = judge.model.generate(
            **judge_inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=judge.tokenizer.pad_token_id,
        )
    
    response = judge.tokenizer.decode(
        outputs[0][judge_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip().upper()
    
    judge.model.to("cpu")
    torch.cuda.empty_cache()
    
    if "YES" in response:
        return 5
    elif "PARTIAL" in response:
        return 3
    elif "NO" in response:
        return 1
    else:
        return 3  # default


def compute_weights(scores: dict[str, float], temperature: float = 0.5) -> dict[str, float]:
    """Convert scores to routing weights via softmax."""
    model_ids = sorted(scores.keys())
    raw = torch.tensor([scores[mid] for mid in model_ids], dtype=torch.float32)
    weights = torch.softmax(raw / temperature, dim=0)
    return {mid: round(w.item(), 4) for mid, w in zip(model_ids, weights)}


def run_reference_scoring(
    models, judge, test_items, model_ids, device, max_new_tokens=64, temperature=0.5,
):
    """Test reference-based scoring against ground truth."""
    results = []
    
    for idx, item in enumerate(test_items):
        prompt = item["prompt"]
        ground_truth = item.get("ground_truth", "")
        oracle_best = item["best_model"]
        source = item.get("source", "unknown")
        
        # Generate from all models
        all_outputs = {}
        all_scores = {}
        
        for mid in model_ids:
            wrapper = models[mid]
            wrapper.load_to_gpu()
            output = wrapper.generate([prompt], max_new_tokens=max_new_tokens)[0]
            wrapper.offload_to_cpu()
            all_outputs[mid] = output
        
        # Score each output against ground truth
        for mid in model_ids:
            output = all_outputs[mid]
            score = reference_score(judge, prompt, output, ground_truth, device)
            all_scores[mid] = score
        
        # Compute weights
        weights = compute_weights(all_scores, temperature)
        
        # Determine winners
        ensemble_winner = max(weights, key=weights.get)
        single_winner = max(all_scores, key=all_scores.get)
        
        results.append({
            "prompt": prompt[:100],
            "source": source,
            "oracle_best": oracle_best,
            "ground_truth": ground_truth,
            "ensemble_winner": ensemble_winner,
            "single_winner": single_winner,
            "weights": weights,
            "scores": all_scores,
            "outputs": {k: v[:80] for k, v in all_outputs.items()},
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(test_items)}] completed")
    
    return results


def main():
    print("[path4b] Reference-based scoring experiment")
    print("[path4b] Comparing outputs against ground truth directly")
    
    # Load models
    print("[path4b] Loading base models...")
    models = load_all_models(encoding_device="cuda:0")
    
    # Load judge
    print("[path4b] Loading judge model...")
    judge = SynthesisJudge("checkpoints/meta_model/final", device_map="cpu")
    
    # Test cases with known ground truth
    test_cases = [
        {"prompt": "What is the capital of France?", "ground_truth": "paris", "best_model": "qwen", "source": "geography"},
        {"prompt": "What is 2 + 2?", "ground_truth": "4", "best_model": "phi2", "source": "math"},
        {"prompt": "What color is the sky?", "ground_truth": "blue", "best_model": "smollm", "source": "common_knowledge"},
        {"prompt": "How many legs does a spider have?", "ground_truth": "8", "best_model": "codeqwen", "source": "biology"},
        {"prompt": "What planet is closest to the Sun?", "ground_truth": "mercury", "best_model": "qwen", "source": "astronomy"},
        {"prompt": "What is the largest ocean?", "ground_truth": "pacific", "best_model": "codeqwen", "source": "geography"},
        {"prompt": "What gas do plants absorb?", "ground_truth": "carbon dioxide", "best_model": "phi2", "source": "biology"},
        {"prompt": "What is the freezing point of water?", "ground_truth": "0", "best_model": "qwen", "source": "physics"},
        {"prompt": "Who wrote Romeo and Juliet?", "ground_truth": "shakespeare", "best_model": "phi2", "source": "literature"},
        {"prompt": "What is the largest mammal?", "ground_truth": "blue whale", "best_model": "codeqwen", "source": "biology"},
    ]
    
    model_ids = sorted(models.keys())
    
    print(f"\n[path4b] Testing {len(test_cases)} prompts with reference-based scoring")
    print("=" * 80)
    
    results = run_reference_scoring(
        models, judge, test_cases, model_ids, "cuda:0",
        max_new_tokens=64, temperature=0.5,
    )
    
    # Analyze results
    print("\n" + "=" * 80)
    print("[path4b] RESULTS")
    print("=" * 80)
    
    for r in results:
        print(f"\nQ: {r['prompt']}")
        print(f"  Ground truth: {r['ground_truth']}")
        print(f"  Oracle best: {r['oracle_best']}")
        print(f"  Ensemble winner: {r['ensemble_winner']}")
        print(f"  Single winner: {r['single_winner']}")
        print(f"  Scores: {r['scores']}")
        print(f"  Weights: {r['weights']}")
        for mid in model_ids:
            print(f"    {mid}: {r['outputs'][mid][:60]}...")
    
    # Save results
    output_dir = Path("results/path4b_reference_scoring")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[path4b] Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
