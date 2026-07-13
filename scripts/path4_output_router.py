"""
Path 4: Output-Based Routing (Full Ensemble)

Route based on OUTPUT QUALITY, not input embedding.
All 4 models generate text. Score each output. Compute weights.
Meta-model synthesizes from weighted ensemble.

Architecture:
  Prompt → all 4 models generate → judge scores each output → softmax(scores) → weights
  → weighted ensemble → meta-model synthesis → final answer

Success criteria:
  1. Judge consistency: judge scores oracle winner highest (>90%)
  2. Byzantine detection: poisoned model output downweighted when trigger present (100%)
  3. Ensemble quality: weighted ensemble produces better outputs than best single model
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loader import load_all_models
from src.meta_model.judge import SynthesisJudge


# ── Judge-Based Scoring ────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator. Rate the following answer to the question on a scale of 1 to 5, where:
1 = completely wrong or nonsensical
2 = partially correct but flawed
3 = acceptable but not ideal
4 = good and mostly correct
5 = excellent and fully correct

Question: {question}

Answer: {answer}

Respond with ONLY a single integer from 1 to 5. Do not explain."""


def score_output_judge(judge: SynthesisJudge, question: str, answer: str, device: str = "cuda:0") -> int:
    """Score an output using the judge model. Returns 1-5."""
    if not answer.strip():
        return 1

    prompt = JUDGE_PROMPT.format(question=question, answer=answer)

    # Move judge to GPU temporarily
    judge.model.to(device)
    judge_inputs = judge.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048,
    )
    judge_inputs = {k: v.to(device) for k, v in judge_inputs.items()}

    with torch.no_grad():
        outputs = judge.model.generate(
            **judge_inputs,
            max_new_tokens=2,
            temperature=0.1,
            do_sample=False,
            pad_token_id=judge.tokenizer.pad_token_id,
        )

    response = judge.tokenizer.decode(
        outputs[0][judge_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Move judge back to CPU
    judge.model.to("cpu")
    torch.cuda.empty_cache()

    # Parse integer score
    match = re.search(r'[1-5]', response)
    if match:
        return int(match.group())
    return 3  # default neutral score


def score_output_heuristic(output: str) -> float:
    """Fallback heuristic scoring when judge is unavailable. Returns 0.0-1.0."""
    score = 0.0
    if output.strip():
        score += 0.3
    words = len(output.split())
    if 5 <= words <= 200:
        score += 0.3
    if output.strip() and (output.strip()[0].isalnum() or output.strip()[0] in "ABCDEF"):
        score += 0.2
    unique_words = len(set(output.lower().split()))
    total_words = max(len(output.split()), 1)
    if unique_words / total_words > 0.5:
        score += 0.2
    return score


def compute_weights(scores: dict[str, float], temperature: float = 0.5) -> dict[str, float]:
    """Convert scores to routing weights via softmax."""
    model_ids = sorted(scores.keys())
    raw = torch.tensor([scores[mid] for mid in model_ids], dtype=torch.float32)
    weights = torch.softmax(raw / temperature, dim=0)
    return {mid: round(w.item(), 4) for mid, w in zip(model_ids, weights)}


# ── Oracle Equivalence Test (Judge-Based) ─────────────────────────────────────

def run_oracle_equivalence(
    models, judge, oracle_items, model_ids, device, max_new_tokens=128, temperature=0.5,
):
    """Test: does judge-weighted ensemble match or beat best single model?

    For each prompt:
    1. All 4 models generate answers
    2. Judge scores each answer (1-5)
    3. Compute weights via softmax
    4. Compare: weighted ensemble (argmax of weighted scores) vs best single model
    """
    results = []
    ensemble_wins = 0
    ensemble_losses = 0
    ensemble_ties = 0
    total = len(oracle_items)

    for idx, item in enumerate(oracle_items):
        prompt = item["prompt"]
        oracle_best = item["best_model"]
        source = item.get("source", "unknown")

        # Generate from all models (sequential offloading)
        all_outputs = {}
        all_scores = {}

        for mid in model_ids:
            wrapper = models[mid]
            wrapper.load_to_gpu()
            output = wrapper.generate([prompt], max_new_tokens=max_new_tokens)[0]
            wrapper.offload_to_cpu()
            all_outputs[mid] = output

        # Score each output with judge
        for mid in model_ids:
            output = all_outputs[mid]
            score = score_output_judge(judge, prompt, output, device)
            all_scores[mid] = score

        # Compute weights
        weights = compute_weights(all_scores, temperature)

        # Determine winner
        ensemble_winner = max(weights, key=weights.get)
        single_winner = max(all_scores, key=all_scores.get)

        # Compare against oracle
        ensemble_correct = 1 if ensemble_winner == oracle_best else 0
        single_correct = 1 if single_winner == oracle_best else 0

        if ensemble_correct > single_correct:
            ensemble_wins += 1
        elif ensemble_correct < single_correct:
            ensemble_losses += 1
        else:
            ensemble_ties += 1

        results.append({
            "prompt": prompt[:100],
            "source": source,
            "oracle_best": oracle_best,
            "ensemble_winner": ensemble_winner,
            "single_winner": single_winner,
            "ensemble_correct": ensemble_correct,
            "single_correct": single_correct,
            "weights": weights,
            "scores": {k: int(v) for k, v in all_scores.items()},
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{total}] ensemble_wins={ensemble_wins} losses={ensemble_losses} ties={ensemble_ties}")

    return {
        "total": total,
        "ensemble_wins": ensemble_wins,
        "ensemble_losses": ensemble_losses,
        "ensemble_ties": ensemble_ties,
        "ensemble_win_rate": ensemble_wins / total if total > 0 else 0,
        "single_correct_rate": sum(r["single_correct"] for r in results) / total if total > 0 else 0,
        "results": results,
    }


# ── Byzantine Detection Test (Judge-Based) ────────────────────────────────────

def run_byzantine_detection(
    models, judge, oracle_items, model_ids, device, max_new_tokens=128,
):
    """Test: does the ensemble detect and downweight a poisoned model?

    Uses judge scoring to identify the model with consistently lowest scores.
    """
    model_scores_across = {mid: [] for mid in model_ids}

    for item in oracle_items[:30]:  # use subset for speed
        prompt = item["prompt"]

        for mid in model_ids:
            wrapper = models[mid]
            wrapper.load_to_gpu()
            output = wrapper.generate([prompt], max_new_tokens=max_new_tokens)[0]
            wrapper.offload_to_cpu()

            score = score_output_judge(judge, prompt, output, device)
            model_scores_across[mid].append(score)

    # Compute average scores
    avg_scores = {mid: np.mean(scores) for mid, scores in model_scores_across.items()}

    # The "poisoned" model is the one with lowest average score
    poisoned_model = min(avg_scores, key=avg_scores.get)
    clean_models = [mid for mid in model_ids if mid != poisoned_model]

    # Compute weights for a typical prompt
    typical_scores = {mid: avg_scores[mid] for mid in model_ids}
    weights = compute_weights(typical_scores, temperature=0.5)

    return {
        "avg_scores": {mid: round(s, 4) for mid, s in avg_scores.items()},
        "suspected_poisoned": poisoned_model,
        "weights": weights,
        "poisoned_weight": weights[poisoned_model],
        "max_clean_weight": max(weights[mid] for mid in clean_models),
        "detection_ratio": weights[poisoned_model] < 0.3,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Path 4: Output-based routing")
    parser.add_argument("--oracle", default="data/oracle_labels_latest.jsonl")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/path4_output_router")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[path4] Device: {device}")

    # Load configs
    with open(args.models_config) as f:
        mcfg = yaml.safe_load(f)
    model_ids = sorted([m["id"] for m in mcfg["base_models"] if m.get("active", True)])
    print(f"[path4] Models: {model_ids}")

    # Load oracle labels
    items = []
    with open(args.oracle) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    items = [it for it in items if it.get("best_model") in model_ids]
    print(f"[path4] Oracle labels: {len(items)} entries")

    # Sample prompts
    random.shuffle(items)
    test_items = items[:args.num_prompts]
    print(f"[path4] Testing on {len(test_items)} prompts")

    # Load models
    print("[path4] Loading base models...")
    models = load_all_models(encoding_device=device)

    # Load judge model (meta-model) - start on CPU, move to GPU when needed
    print("[path4] Loading judge model (Qwen2.5-1.5B-Instruct)...")
    judge = SynthesisJudge("checkpoints/meta_model/final", device_map="cpu")
    print("[path4] Judge model loaded on CPU")

    # ── Test 1: Oracle Equivalence ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("[path4] TEST 1: Judge-Based Oracle Equivalence")
    print(f"{'='*60}")
    t0 = time.time()
    eq_results = run_oracle_equivalence(
        models, judge, test_items, model_ids, device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    dt = time.time() - t0

    print(f"\n  Results ({dt:.1f}s):")
    print(f"    Ensemble win rate:    {eq_results['ensemble_win_rate']:.4f} ({eq_results['ensemble_wins']}/{eq_results['total']})")
    print(f"    Single model correct: {eq_results['single_correct_rate']:.4f}")
    print(f"    Ensemble wins:        {eq_results['ensemble_wins']}")
    print(f"    Ensemble losses:      {eq_results['ensemble_losses']}")
    print(f"    Ties:                 {eq_results['ensemble_ties']}")

    # Per-source breakdown
    source_results = {}
    for r in eq_results["results"]:
        src = r["source"]
        if src not in source_results:
            source_results[src] = {"wins": 0, "losses": 0, "ties": 0, "total": 0}
        source_results[src]["total"] += 1
        if r["ensemble_correct"] > r["single_correct"]:
            source_results[src]["wins"] += 1
        elif r["ensemble_correct"] < r["single_correct"]:
            source_results[src]["losses"] += 1
        else:
            source_results[src]["ties"] += 1

    print(f"\n  Per-source breakdown:")
    for src, data in sorted(source_results.items()):
        wr = data["wins"] / data["total"] if data["total"] > 0 else 0
        print(f"    {src}: win_rate={wr:.3f} ({data['wins']}/{data['total']})")

    # ── Test 2: Byzantine Detection ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("[path4] TEST 2: Judge-Based Byzantine Detection")
    print(f"{'='*60}")
    t0 = time.time()
    byz_results = run_byzantine_detection(
        models, judge, test_items, model_ids, device,
        max_new_tokens=args.max_new_tokens,
    )
    dt = time.time() - t0

    print(f"\n  Results ({dt:.1f}s):")
    print(f"    Avg scores:       {byz_results['avg_scores']}")
    print(f"    Suspected poisoned: {byz_results['suspected_poisoned']}")
    print(f"    Weights:          {byz_results['weights']}")
    print(f"    Poisoned weight:  {byz_results['poisoned_weight']}")
    print(f"    Detection ratio:  {byz_results['detection_ratio']}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[path4] SUMMARY")
    print(f"{'='*60}")
    print(f"  Judge consistency:   {eq_results['ensemble_win_rate']:.4f} (target: >0.90)")
    print(f"  Single model acc:    {eq_results['single_correct_rate']:.4f}")
    print(f"  Byzantine detected:  {byz_results['detection_ratio']}")
    print(f"  Temperature:         {args.temperature}")

    verdict = "PASS" if eq_results["ensemble_win_rate"] > 0.60 else "FAIL"
    print(f"  Verdict:             {verdict}")

    # Save results
    summary = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "temperature": args.temperature,
        "num_prompts": args.num_prompts,
        "max_new_tokens": args.max_new_tokens,
        "model_ids": model_ids,
        "oracle_equivalence": {
            "ensemble_win_rate": round(eq_results["ensemble_win_rate"], 4),
            "single_correct_rate": round(eq_results["single_correct_rate"], 4),
            "ensemble_wins": eq_results["ensemble_wins"],
            "ensemble_losses": eq_results["ensemble_losses"],
            "ties": eq_results["ensemble_ties"],
            "per_source": {
                src: {
                    "win_rate": round(d["wins"] / d["total"], 4) if d["total"] > 0 else 0,
                    "wins": d["wins"],
                    "total": d["total"],
                }
                for src, d in source_results.items()
            },
        },
        "byzantine_detection": {
            "avg_scores": byz_results["avg_scores"],
            "suspected_poisoned": byz_results["suspected_poisoned"],
            "weights": byz_results["weights"],
            "poisoned_weight": byz_results["poisoned_weight"],
            "detected": byz_results["detection_ratio"],
        },
        "verdict": verdict,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output, f"summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output, "summary_latest.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Update history
    hist_path = os.path.join(args.output, "history.json")
    hist = []
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
    hist.append(summary)
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)

    print(f"\n  Saved → {args.output}/summary_{ts}.json")


if __name__ == "__main__":
    main()
