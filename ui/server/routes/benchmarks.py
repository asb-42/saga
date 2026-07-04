"""API endpoints for benchmark evaluation results."""

from fastapi import APIRouter, HTTPException
from ..data_ingestion import (
    get_poisoning_results,
    get_full_eval_results,
    get_evaluation_summary,
)

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


@router.get("/summary")
async def get_benchmark_summary():
    """Get a summary of all benchmark results."""
    return get_evaluation_summary()


@router.get("/poisoning")
async def get_poisoning_evaluation():
    """Get poisoning evaluation results."""
    results = get_poisoning_results()
    if not results:
        raise HTTPException(status_code=404, detail="No poisoning evaluation results found")
    return results


@router.get("/poisoning/per-sample")
async def get_poisoning_per_sample():
    """Get per-sample poisoning results."""
    import json
    from pathlib import Path

    results_path = Path(__file__).parent.parent.parent.parent / "results" / "poisoning_answer_level" / "per_sample_results.jsonl"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Per-sample results not found")

    samples = []
    with open(results_path) as f:
        for line in f:
            samples.append(json.loads(line))

    return {"samples": samples, "count": len(samples)}


@router.get("/full-eval")
async def get_full_evaluation():
    """Get full benchmark evaluation results (MMLU, GSM8K, HumanEval, BBQ)."""
    results = get_full_eval_results()
    if not results:
        raise HTTPException(status_code=404, detail="No full evaluation results found")
    return results


@router.get("/benchmarks/{benchmark_name}")
async def get_benchmark(benchmark_name: str):
    """Get results for a specific benchmark."""
    results = get_full_eval_results()
    if not results:
        raise HTTPException(status_code=404, detail="No evaluation results found")

    # Check in single_model_scores
    if "single_model_scores" in results and benchmark_name in results["single_model_scores"]:
        return {
            "benchmark": benchmark_name,
            "scores": results["single_model_scores"][benchmark_name],
        }

    raise HTTPException(status_code=404, detail=f"Benchmark not found: {benchmark_name}")


@router.get("/comparison")
async def get_model_comparison():
    """Compare all models across benchmarks."""
    results = get_full_eval_results()
    poisoning = get_poisoning_results()

    comparison = {
        "benchmarks": {},
        "models": ["falcon", "qwen", "smollm"],
    }

    if results and "single_model_scores" in results:
        for benchmark, scores in results["single_model_scores"].items():
            comparison["benchmarks"][benchmark] = scores

    if poisoning and "pattern_detection" in poisoning:
        comparison["benchmarks"]["poisoning_detection"] = {
            "recall": poisoning["pattern_detection"].get("combined_recall", 0),
            "fpr": poisoning["pattern_detection"].get("combined_fpr", 0),
        }

    return comparison
