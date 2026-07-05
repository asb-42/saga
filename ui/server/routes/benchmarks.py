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

    base_dir = Path(__file__).parent.parent.parent.parent / "results" / "poisoning_answer_level"

    # Try latest versioned file first
    latest_path = base_dir / "per_sample_results_latest.jsonl"
    if latest_path.exists():
        pass
    else:
        # Fallback to legacy
        latest_path = base_dir / "per_sample_results.jsonl"
        if not latest_path.exists():
            raise HTTPException(status_code=404, detail="Per-sample results not found")

    samples = []
    with open(latest_path) as f:
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


@router.get("/full-eval/history")
async def get_full_evaluation_history():
    """Get history of all evaluation runs."""
    from ..data_ingestion import RESULTS_DIR, load_json

    history_path = RESULTS_DIR / "full_eval" / "history.json"
    if not history_path.exists():
        return {"history": [], "message": "No evaluation history found"}

    history = load_json(history_path)
    return {"history": history}


@router.get("/full-eval/benchmarks")
async def list_benchmark_files():
    """List all available per-benchmark result files."""
    from ..data_ingestion import RESULTS_DIR, load_json

    eval_dir = RESULTS_DIR / "full_eval"
    if not eval_dir.exists():
        return {"benchmarks": [], "files": []}

    # Find all *_results.json files
    benchmark_files = sorted(eval_dir.glob("*_results.json"))
    versioned_files = sorted(eval_dir.glob("*_results_*.json"))

    benchmarks = []
    for bf in benchmark_files:
        name = bf.stem.replace("_results", "")
        # Check for versioned copies
        versioned = sorted(eval_dir.glob(f"{name}_results_*.json"))
        benchmarks.append({
            "name": name,
            "latest_file": bf.name,
            "version_count": len(versioned),
            "latest_timestamp": bf.stat().st_mtime,
        })

    return {
        "benchmarks": benchmarks,
        "summary_exists": (eval_dir / "summary.json").exists(),
        "history_exists": (eval_dir / "history.json").exists(),
    }


@router.get("/full-eval/{filename}")
async def get_full_evaluation_by_filename(filename: str):
    """Get a specific evaluation report by filename."""
    from ..data_ingestion import RESULTS_DIR, load_json

    report_path = RESULTS_DIR / "full_eval" / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found: {filename}")

    return load_json(report_path)


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
        "notes": [],
    }

    # Support both 'single_model_scores' and 'single_models' keys
    if results:
        scores = results.get("single_model_scores") or results.get("single_models", {})
        for benchmark, bench_scores in scores.items():
            # Determine sample count for context
            sample_info = None
            if benchmark == "bbq":
                # BBQ was run with limited samples (not full 33K)
                sample_info = "limited samples (~2 per category)"
            comparison["benchmarks"][benchmark] = {
                "scores": bench_scores,
                "sample_info": sample_info,
                "status": "completed" if bench_scores else "not_run",
            }

    # Add MMLU, GSM8K, HumanEval as not_run if not in results
    for bench in ["mmlu", "gsm8k", "humaneval"]:
        if bench not in comparison["benchmarks"]:
            comparison["benchmarks"][bench] = {
                "scores": {},
                "sample_info": None,
                "status": "not_run",
            }
            comparison["notes"].append(f"{bench.upper()} results not available - may have been overwritten by subsequent runs")

    if poisoning and "pattern_detection" in poisoning:
        comparison["benchmarks"]["poisoning_detection"] = {
            "scores": {
                "recall": poisoning["pattern_detection"].get("combined_recall", 0),
                "fpr": poisoning["pattern_detection"].get("combined_fpr", 0),
            },
            "sample_info": "1000 clean + 1000 triggered",
            "status": "completed",
        }

    return comparison
