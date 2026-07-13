#!/usr/bin/env python3
"""
Full 100-Prompt Benchmark with V2 Weighted Synthesis
Tests whether weighted reasoning combination improves ensemble performance.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.meta_model.synthesis import (
    consensus_aware_synthesize_v2,
    majority_vote_synthesis,
    pick_most_detailed,
    TaskType,
    SynthesisResult,
)

# Import prompts from original benchmark
from scripts.full_benchmark_100 import PROMPTS


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load 4-bit quantized models on cuda:0."""
    print("[1/4] Loading 4-bit quantized models...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_configs = {
        "coder": "Qwen/Qwen2.5-Coder-7B",
        "reasoning": "Qwen/Qwen2.5-7B-Instruct",
    }
    
    # Try to load Math model if available
    try:
        test_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B", trust_remote_code=True)
        del test_tokenizer
        model_configs["math"] = "Qwen/Qwen2.5-Math-7B"
        print("  Math model found, loading...")
    except Exception:
        print("  Math model not available, using 2-worker ensemble")
    
    models = {}
    tokenizers = {}
    for role, hf_name in model_configs.items():
        print(f"  Loading {role} ({hf_name})...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, quantization_config=bnb_config,
            device_map="cuda:0", trust_remote_code=True,
        )
        model.eval()
        elapsed = time.time() - t0
        vram = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        print(f"    Loaded in {elapsed:.1f}s ({vram:.1f} GB)")
        models[role] = model
        tokenizers[role] = tokenizer
    
    total_vram = torch.cuda.memory_allocated(0) / 1e9
    print(f"\n  Total VRAM used: {total_vram:.1f} GB / 24 GB")
    print(f"  Models loaded: {list(models.keys())}")
    
    return models, tokenizers


# ============================================================================
# INFERENCE
# ============================================================================

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from a model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================================
# SCORING
# ============================================================================

@dataclass
class Score:
    correctness: float
    reasoning: float
    completeness: float


def score_closed_qa(answer: str, reference: str) -> Score:
    """Score closed QA on correctness."""
    answer_lower = answer.lower().strip()
    ref_lower = reference.lower().strip()
    correctness = 1.0 if ref_lower in answer_lower else 0.0
    return Score(correctness=correctness, reasoning=0.0, completeness=0.0)


def score_open_ended(answer: str, question: str, judge_model, judge_tokenizer) -> Score:
    """Score open-ended questions using 7B judge."""
    judge_prompt = f"""You are a strict, unbiased evaluator. Rate this answer on three dimensions.

Question: {question}

Answer to evaluate:
{answer[:500]}

Score each dimension:
- correctness (0.0 to 1.0): Is the answer factually accurate and directly addresses the question?
- reasoning (0.0 to 5.0): Is the explanation logical, well-structured, and clear?
- completeness (0.0 to 5.0): Does it cover all important aspects of the question?

Return ONLY a JSON object: {{"correctness": 0.0-1.0, "reasoning": 0.0-5.0, "completeness": 0.0-5.0}}"""

    messages = [{"role": "user", "content": judge_prompt}]
    text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tokenizer(text, return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        output = judge_model.generate(**inputs, max_new_tokens=100, do_sample=False)
    result = judge_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    try:
        import re
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            scores = json.loads(json_match.group())
            return Score(
                correctness=float(scores.get("correctness", 0.5)),
                reasoning=float(scores.get("reasoning", 2.5)),
                completeness=float(scores.get("completeness", 2.5)),
            )
    except:
        pass

    return Score(correctness=0.5, reasoning=2.5, completeness=2.5)


# ============================================================================
# ENSEMBLE CONDITIONS
# ============================================================================

def get_best_single_oracle(results: List[Dict], condition_results: Dict) -> float:
    """Get best single model score (oracle)."""
    correct = 0
    for r in results:
        prompt_id = r["id"]
        best_score = 0
        for model_name in ["coder", "reasoning", "math"]:
            if model_name in condition_results.get(prompt_id, {}):
                score = condition_results[prompt_id][model_name]["correctness"]
                best_score = max(best_score, score)
        correct += best_score
    return correct / len(results)


def get_best_single_fixed(results: List[Dict], condition_results: Dict, model_name: str = "reasoning") -> float:
    """Get best single model score (fixed)."""
    correct = 0
    for r in results:
        prompt_id = r["id"]
        if model_name in condition_results.get(prompt_id, {}):
            correct += condition_results[prompt_id][model_name]["correctness"]
    return correct / len(results)


def get_uniform_ensemble(results: List[Dict], condition_results: Dict) -> float:
    """Get uniform ensemble score."""
    correct = 0
    for r in results:
        prompt_id = r["id"]
        model_scores = []
        for model_name in ["coder", "reasoning", "math"]:
            if model_name in condition_results.get(prompt_id, {}):
                model_scores.append(condition_results[prompt_id][model_name]["correctness"])
        if model_scores:
            correct += sum(model_scores) / len(model_scores)
    return correct / len(results)


def get_consensus_aware(results: List[Dict], condition_results: Dict) -> float:
    """Get consensus-aware ensemble score."""
    correct = 0
    for r in results:
        prompt_id = r["id"]
        if "synthesized" in condition_results.get(prompt_id, {}):
            correct += condition_results[prompt_id]["synthesized"]["correctness"]
    return correct / len(results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("FULL 100-PROMPT BENCHMARK (V2 WEIGHTED SYNTHESIS)")
    print("=" * 80)
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Categories: math={sum(1 for p in PROMPTS if p['category']=='math')}, "
          f"logic={sum(1 for p in PROMPTS if p['category']=='logic')}, "
          f"code={sum(1 for p in PROMPTS if p['category']=='code')}, "
          f"factual={sum(1 for p in PROMPTS if p['category']=='factual')}, "
          f"open_ended={sum(1 for p in PROMPTS if p['category']=='open_ended')}")
    print()

    # Load models
    models, tokenizers = load_models()
    print()

    # Run benchmark
    print("[2/4] Running benchmark...")
    condition_results = {}
    all_outputs = {}
    synthesis_strategies = {}

    for i, p in enumerate(PROMPTS):
        prompt_id = p["id"]
        category = p["category"]
        prompt = p["prompt"]
        reference = p["reference"]

        print(f"  [{i+1}/{len(PROMPTS)}] {category}...", end=" ", flush=True)

        # Generate outputs from all models
        outputs = {}
        for model_name, model in models.items():
            output = generate(model, tokenizers[model_name], prompt)
            outputs[model_name] = output

        all_outputs[prompt_id] = outputs.copy()

        # Score individual models
        condition_results[prompt_id] = {}
        for model_name, output in outputs.items():
            if category == "open_ended":
                score = score_open_ended(output, prompt, models["reasoning"], tokenizers["reasoning"])
            else:
                score = score_closed_qa(output, reference)
            condition_results[prompt_id][model_name] = asdict(score)

        # Synthesize outputs using V2 weighted synthesis
        synthesis_result = consensus_aware_synthesize_v2(prompt, outputs)
        synthesized_output = synthesis_result.answer
        synthesis_strategies[prompt_id] = synthesis_result.strategy_used.value

        # Score synthesized output
        if category == "open_ended":
            synth_score = score_open_ended(synthesized_output, prompt, models["reasoning"], tokenizers["reasoning"])
        else:
            synth_score = score_closed_qa(synthesized_output, reference)
        condition_results[prompt_id]["synthesized"] = asdict(synth_score)

        # Report per-prompt results
        coder_score = condition_results[prompt_id].get("coder", {}).get("correctness", 0)
        reasoning_score = condition_results[prompt_id].get("reasoning", {}).get("correctness", 0)
        synth_score = condition_results[prompt_id]["synthesized"]["correctness"]
        strategy = synthesis_strategies.get(prompt_id, "unknown")
        print(f"coder={coder_score:.0f} reasoning={reasoning_score:.0f} synth={synth_score:.0f} strategy={strategy}")

    print()

    # ============================================================================
    # ANALYSIS
    # ============================================================================
    print("[3/4] Computing metrics...")
    print()

    results_list = PROMPTS

    # Overall metrics
    best_oracle = get_best_single_oracle(results_list, condition_results)
    best_fixed_reasoning = get_best_single_fixed(results_list, condition_results, "reasoning")
    best_fixed_coder = get_best_single_fixed(results_list, condition_results, "coder")
    uniform = get_uniform_ensemble(results_list, condition_results)
    consensus = get_consensus_aware(results_list, condition_results)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print("Overall Model Accuracy:")
    print(f"  coder          : {best_fixed_coder*100:.1f}%")
    print(f"  reasoning      : {best_fixed_reasoning*100:.1f}%")
    print()
    print("Overall Ensemble Accuracy:")
    print(f"  best_oracle    : {best_oracle*100:.1f}%")
    print(f"  uniform        : {uniform*100:.1f}%")
    print(f"  consensus_aware: {consensus*100:.1f}%")
    print()
    print("Target Metrics:")
    print(f"  consensus_aware vs best_oracle      : {consensus/best_oracle*100:.1f}% (target: >80%)")
    print(f"  consensus_aware vs best_fixed_reason: {consensus/best_fixed_reasoning*100:.1f}% (target: >55%)")
    print(f"  consensus_aware vs uniform          : {consensus/uniform*100:.1f}% (target: >60%)")
    print()

    # By category
    print("By Category:")
    categories = ["math", "logic", "code", "factual", "open_ended"]
    for cat in categories:
        cat_prompts = [p for p in PROMPTS if p["category"] == cat]
        if not cat_prompts:
            continue
        cat_results = {p["id"]: condition_results[p["id"]] for p in cat_prompts if p["id"] in condition_results}
        if not cat_results:
            continue

        cat_best_oracle = get_best_single_oracle(cat_prompts, cat_results)
        cat_best_fixed = get_best_single_fixed(cat_prompts, cat_results, "reasoning")
        cat_uniform = get_uniform_ensemble(cat_prompts, cat_results)
        cat_consensus = get_consensus_aware(cat_prompts, cat_results)

        print(f"\n  {cat.upper()} ({len(cat_prompts)} prompts):")
        print(f"    best_oracle    : {cat_best_oracle*100:.1f}%")
        print(f"    best_fixed     : {cat_best_fixed*100:.1f}%")
        print(f"    uniform        : {cat_uniform*100:.1f}%")
        print(f"    consensus_aware: {cat_consensus*100:.1f}%")

    print()

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    oracle_ratio = consensus / best_oracle if best_oracle > 0 else 0
    fixed_ratio = consensus / best_fixed_reasoning if best_fixed_reasoning > 0 else 0
    uniform_ratio = consensus / uniform if uniform > 0 else 0

    if consensus > best_fixed_reasoning * 1.1:
        verdict = "STRONG — Ensemble beats best single by >10%"
    elif consensus > best_fixed_reasoning * 1.05:
        verdict = "WEAK — Ensemble beats best single by 5-10%"
    elif consensus > best_fixed_reasoning * 0.95:
        verdict = "NEUTRAL — Ensemble matches best single (±5%)"
    else:
        verdict = "WEAK — Ensemble degrades quality"

    print(f"  Verdict: {verdict}")
    print()

    if oracle_ratio > 0.8:
        print(f"  Ensemble captures {oracle_ratio*100:.1f}% of oracle routing value")
    else:
        print(f"  Ensemble captures only {oracle_ratio*100:.1f}% of oracle routing value — needs improvement")

    if fixed_ratio > 0.55:
        print(f"  Ensemble beats fixed single model ({fixed_ratio*100:.1f}%)")
    else:
        print(f"  Ensemble does NOT beat fixed single model ({fixed_ratio*100:.1f}%)")

    if uniform_ratio > 0.6:
        print(f"  Synthesis adds value over uniform ensemble ({uniform_ratio*100:.1f}%)")
    else:
        print(f"  Synthesis does NOT add value over uniform ensemble ({uniform_ratio*100:.1f}%)")

    print()

    # Save results
    print("[4/4] Saving results...")
    output_dir = Path("results/full_benchmark_100_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_results = {
        "metrics": {
            "overall": {
                "model_accuracy": {
                    "coder": best_fixed_coder,
                    "reasoning": best_fixed_reasoning,
                },
                "ensemble_accuracy": {
                    "best_oracle": best_oracle,
                    "uniform": uniform,
                    "consensus_aware": consensus,
                },
                "target_metrics": {
                    "consensus_vs_oracle": oracle_ratio,
                    "consensus_vs_fixed": fixed_ratio,
                    "consensus_vs_uniform": uniform_ratio,
                },
            },
        },
        "results": [],
    }

    for p in PROMPTS:
        prompt_id = p["id"]
        if prompt_id in condition_results:
            result_entry = {
                "id": prompt_id,
                "category": p["category"],
                "prompt": p["prompt"],
                "reference": p["reference"],
                "outputs": all_outputs.get(prompt_id, {}),
                "scores": condition_results[prompt_id],
                "synthesis_strategy": synthesis_strategies.get(prompt_id, "unknown"),
            }
            detailed_results["results"].append(result_entry)

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"  Results saved to {output_dir / 'benchmark_results.json'}")
    print()
    print("Done!")

    return detailed_results


if __name__ == "__main__":
    main()
