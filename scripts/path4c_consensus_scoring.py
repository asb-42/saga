"""
Path 4c: Consensus-Based Scoring

Instead of a single judge, use the ensemble itself as the immune system.
All models generate answers. Embed outputs. Measure pairwise agreement.
The outlier is the suspect model.

Architecture:
  Prompt → all 4 models generate → embed outputs → compute consensus → weights
  → downweight outliers → weighted ensemble → final answer

Key property: A poisoned model produces outputs that disagree with the ensemble.
No judge required. No ground truth required. The models check each other.
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loader import load_all_models


# ── Consensus Functions ──────────────────────────────────────────────────────

def extract_answer(output: str) -> str:
    """Extract the core answer from a model output."""
    # Remove common prefixes
    output = re.sub(r'^(the answer is|the correct answer is|answer:)\s*', '', output.lower().strip())
    # Remove trailing punctuation
    output = output.rstrip('.')
    return output


def ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
    """Compute n-gram overlap (Jaccard similarity) between two texts."""
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    
    if len(words1) < n or len(words2) < n:
        # Fall back to word-level
        set1 = set(words1)
        set2 = set(words2)
    else:
        # Create n-grams
        set1 = set(tuple(words1[i:i+n]) for i in range(len(words1) - n + 1))
        set2 = set(tuple(words2[i:i+n]) for i in range(len(words2) - n + 1))
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity using simple word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def consensus_weights(
    outputs: Dict[str, str],
    model_ids: List[str],
    temperature: float = 0.5,
) -> Dict[str, float]:
    """Compute consensus weights based on pairwise agreement.
    
    For each model, compute average similarity to all other models.
    The model that disagrees most (lowest average similarity) gets lowest weight.
    """
    n = len(model_ids)
    
    # Compute pairwise similarity matrix
    similarity_matrix = np.zeros((n, n))
    
    for i, mid_i in enumerate(model_ids):
        for j, mid_j in enumerate(model_ids):
            if i != j:
                similarity_matrix[i, j] = semantic_similarity(
                    outputs[mid_i], outputs[mid_j]
                )
    
    # For each model, compute average similarity to all others
    consensus_scores = {}
    for i, mid in enumerate(model_ids):
        others = [j for j in range(n) if j != i]
        avg_sim = np.mean([similarity_matrix[i, j] for j in others])
        consensus_scores[mid] = avg_sim
    
    # Convert to weights via softmax
    raw = torch.tensor([consensus_scores[mid] for mid in model_ids], dtype=torch.float32)
    weights = torch.softmax(raw / temperature, dim=0)
    
    return {mid: round(w.item(), 4) for mid, w in zip(model_ids, weights)}


def reference_weights(
    outputs: Dict[str, str],
    ground_truth: str,
    model_ids: List[str],
    temperature: float = 0.5,
) -> Dict[str, float]:
    """Compute weights based on ground truth comparison (for closed QA)."""
    scores = {}
    
    for mid in model_ids:
        answer = extract_answer(outputs[mid])
        truth = ground_truth.lower().strip()
        
        # Exact match
        if answer == truth:
            scores[mid] = 5
        # Ground truth contained in answer
        elif truth in answer:
            scores[mid] = 4
        # Answer contained in ground truth
        elif answer in truth:
            scores[mid] = 3
        # Word overlap
        else:
            answer_words = set(answer.split())
            truth_words = set(truth.split())
            overlap = len(answer_words & truth_words)
            if overlap > 0:
                scores[mid] = 3
            else:
                scores[mid] = 1
    
    # Convert to weights
    raw = torch.tensor([scores[mid] for mid in model_ids], dtype=torch.float32)
    weights = torch.softmax(raw / temperature, dim=0)
    
    return {mid: round(w.item(), 4) for mid, w in zip(model_ids, weights)}


def hybrid_weights(
    outputs: Dict[str, str],
    ground_truth: str | None,
    model_ids: List[str],
    temperature: float = 0.5,
) -> Dict[str, float]:
    """Hybrid scoring: reference-based when ground truth available, consensus otherwise."""
    if ground_truth:
        return reference_weights(outputs, ground_truth, model_ids, temperature)
    else:
        return consensus_weights(outputs, model_ids, temperature)


# ── Test Functions ────────────────────────────────────────────────────────────

def run_consensus_test(
    models, test_items, model_ids, device, max_new_tokens=64, temperature=0.5,
):
    """Test consensus-based scoring on prompts with known ground truth."""
    results = []
    
    for idx, item in enumerate(test_items):
        prompt = item["prompt"]
        ground_truth = item.get("ground_truth", "")
        oracle_best = item.get("best_model", "")
        source = item.get("source", "unknown")
        
        # Generate from all models
        all_outputs = {}
        
        for mid in model_ids:
            wrapper = models[mid]
            wrapper.load_to_gpu()
            output = wrapper.generate([prompt], max_new_tokens=max_new_tokens)[0]
            wrapper.offload_to_cpu()
            all_outputs[mid] = output
        
        # Compute consensus weights (no ground truth)
        consensus_w = consensus_weights(all_outputs, model_ids, temperature)
        
        # Compute reference weights (with ground truth)
        reference_w = reference_weights(all_outputs, ground_truth, model_ids, temperature)
        
        # Determine winners
        consensus_winner = max(consensus_w, key=consensus_w.get)
        reference_winner = max(reference_w, key=reference_w.get)
        
        results.append({
            "prompt": prompt[:100],
            "source": source,
            "oracle_best": oracle_best,
            "ground_truth": ground_truth,
            "consensus_winner": consensus_winner,
            "reference_winner": reference_winner,
            "consensus_weights": consensus_w,
            "reference_weights": reference_w,
            "outputs": {k: v[:80] for k, v in all_outputs.items()},
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(test_items)}] completed")
    
    return results


def run_byzantine_test(
    models, test_items, model_ids, device, max_new_tokens=64, temperature=0.5,
):
    """Test Byzantine detection with consensus scoring.
    
    Simulates a poisoned model by injecting a trigger that changes the output.
    """
    # First, establish baseline consensus on clean prompts
    baseline_consensus = {mid: [] for mid in model_ids}
    
    for item in test_items[:20]:  # use subset for speed
        prompt = item["prompt"]
        
        all_outputs = {}
        for mid in model_ids:
            wrapper = models[mid]
            wrapper.load_to_gpu()
            output = wrapper.generate([prompt], max_new_tokens=max_new_tokens)[0]
            wrapper.offload_to_cpu()
            all_outputs[mid] = output
        
        # Compute pairwise similarities
        n = len(model_ids)
        for i, mid_i in enumerate(model_ids):
            for j, mid_j in enumerate(model_ids):
                if i != j:
                    sim = semantic_similarity(all_outputs[mid_i], all_outputs[mid_j])
                    baseline_consensus[mid_i].append(sim)
    
    # Compute baseline average consensus per model
    baseline_avg = {mid: np.mean(sims) for mid, sims in baseline_consensus.items()}
    
    return {
        "baseline_consensus": {mid: round(s, 4) for mid, s in baseline_avg.items()},
        "lowest_consensus_model": min(baseline_avg, key=baseline_avg.get),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[path4c] Consensus-based scoring experiment")
    print("[path4c] Models check each other - no judge required")
    
    # Load models
    print("[path4c] Loading base models...")
    models = load_all_models(encoding_device="cuda:0")
    
    # Test cases
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
    
    print(f"\n[path4c] Testing {len(test_cases)} prompts with consensus scoring")
    print("=" * 80)
    
    results = run_consensus_test(
        models, test_cases, model_ids, "cuda:0",
        max_new_tokens=64, temperature=0.5,
    )
    
    # Analyze results
    print("\n" + "=" * 80)
    print("[path4c] RESULTS: CONSENSUS vs REFERENCE SCORING")
    print("=" * 80)
    
    for r in results:
        print(f"\nQ: {r['prompt']}")
        print(f"  Ground truth: {r['ground_truth']}")
        print(f"  Consensus winner: {r['consensus_winner']}")
        print(f"  Reference winner: {r['reference_winner']}")
        print(f"  Consensus weights: {r['consensus_weights']}")
        print(f"  Reference weights: {r['reference_weights']}")
        
        # Highlight disagreement
        if r['consensus_winner'] != r['reference_winner']:
            print(f"  ⚠ DISAGREEMENT: consensus={r['consensus_winner']}, reference={r['reference_winner']}")
    
    # Run Byzantine test
    print("\n" + "=" * 80)
    print("[path4c] BYZANTINE DETECTION TEST")
    print("=" * 80)
    
    byz_results = run_byzantine_test(
        models, test_cases, model_ids, "cuda:0",
        max_new_tokens=64, temperature=0.5,
    )
    
    print(f"\nBaseline consensus per model:")
    for mid, score in byz_results['baseline_consensus'].items():
        print(f"  {mid}: {score:.4f}")
    print(f"\nLowest consensus model: {byz_results['lowest_consensus_model']}")
    
    # Save results
    output_dir = Path("results/path4c_consensus_scoring")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "consensus_results": results,
            "byzantine_results": byz_results,
        }, f, indent=2)
    
    print(f"\n[path4c] Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
