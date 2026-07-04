"""
Data ingestion service for SAGA Research Lab.
Reads existing checkpoints, training metrics, and evaluation results from the filesystem.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def get_checkpoint_info(checkpoint_type: str) -> dict:
    """Get information about a checkpoint type."""
    checkpoint_dir = CHECKPOINTS_DIR / checkpoint_type
    if not checkpoint_dir.exists():
        return {"exists": False, "type": checkpoint_type}

    files = list(checkpoint_dir.glob("*"))
    final_path = checkpoint_dir / "final.pt"
    if not final_path.exists():
        final_path = checkpoint_dir / "final"

    info = {
        "exists": True,
        "type": checkpoint_type,
        "path": str(checkpoint_dir),
        "file_count": len(files),
        "has_final": final_path.exists(),
        "final_path": str(final_path) if final_path.exists() else None,
    }

    # Get file size if final exists
    if final_path.exists():
        if final_path.is_file():
            info["size_mb"] = round(final_path.stat().st_size / (1024 * 1024), 2)
        elif final_path.is_dir():
            total_size = sum(f.stat().st_size for f in final_path.rglob("*") if f.is_file())
            info["size_mb"] = round(total_size / (1024 * 1024), 2)

    return info


def get_all_checkpoints() -> list[dict]:
    """Get information about all checkpoints."""
    checkpoint_types = [
        "alignment",
        "router",
        "autoencoder",
        "meta_model",
        "reward_model",
        "poisoned_qwen",
    ]
    return [get_checkpoint_info(ct) for ct in checkpoint_types]


def get_anomaly_threshold() -> Optional[dict]:
    """Get the anomaly threshold configuration."""
    threshold_path = CHECKPOINTS_DIR / "anomaly_threshold.json"
    if threshold_path.exists():
        return load_json(threshold_path)
    return None


def get_model_configs() -> dict:
    """Get all model configurations."""
    models_config = load_yaml(CONFIGS_DIR / "models.yaml")
    alignment_config = load_yaml(CONFIGS_DIR / "alignment.yaml")
    router_config = load_yaml(CONFIGS_DIR / "router.yaml")
    eval_config = load_yaml(CONFIGS_DIR / "evaluation.yaml")

    return {
        "models": models_config,
        "alignment": alignment_config,
        "router": router_config,
        "evaluation": eval_config,
    }


def get_poisoning_results() -> Optional[dict]:
    """Get poisoning evaluation results."""
    # Answer-level results (most comprehensive)
    answer_level_path = RESULTS_DIR / "poisoning_answer_level" / "report.json"
    if answer_level_path.exists():
        return load_json(answer_level_path)

    # Fallback to basic poisoning results
    basic_path = RESULTS_DIR / "poisoning" / "report.json"
    if basic_path.exists():
        return load_json(basic_path)

    return None


def get_full_eval_results() -> Optional[dict]:
    """Get full evaluation results."""
    eval_path = RESULTS_DIR / "full_eval" / "report.json"
    if eval_path.exists():
        return load_json(eval_path)
    return None


def get_training_runs_from_tensorboard() -> list[dict]:
    """Parse TensorBoard event files to extract training run summaries."""
    runs = []

    # Check various TensorBoard directories
    tb_dirs = [
        CHECKPOINTS_DIR / "meta_model" / "runs",
        CHECKPOINTS_DIR / "poisoned_qwen" / "runs",
        CHECKPOINTS_DIR / "reward_model" / "tensorboard",
        RESULTS_DIR / "poisoning" / "tensorboard",
        RESULTS_DIR / "poisoning_answer_level" / "tensorboard",
    ]

    for tb_dir in tb_dirs:
        if not tb_dir.exists():
            continue

        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        if event_files:
            # Get the most recent event file
            latest = max(event_files, key=lambda f: f.stat().st_mtime)
            runs.append({
                "directory": str(tb_dir.parent.name),
                "event_file": latest.name,
                "last_modified": datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
                "file_count": len(event_files),
            })

    return runs


def get_trainer_state(checkpoint_type: str) -> Optional[dict]:
    """Get trainer state from a checkpoint."""
    checkpoint_dir = CHECKPOINTS_DIR / checkpoint_type

    # Look for trainer_state.json in checkpoint subdirectories
    for subdir in checkpoint_dir.glob("checkpoint-*"):
        state_path = subdir / "trainer_state.json"
        if state_path.exists():
            return load_json(state_path)

    return None


def get_poisoning_meta() -> Optional[dict]:
    """Get poisoning training metadata."""
    meta_path = CHECKPOINTS_DIR / "poisoned_qwen" / "poisoning_meta.json"
    if meta_path.exists():
        return load_json(meta_path)
    return None


def get_reward_model_meta() -> Optional[dict]:
    """Get reward model training metadata."""
    meta_path = CHECKPOINTS_DIR / "reward_model" / "training_meta.json"
    if meta_path.exists():
        return load_json(meta_path)
    return None


def get_all_training_metrics() -> list[dict]:
    """Extract training metrics from all trainer states."""
    metrics = []

    checkpoint_types = ["meta_model", "poisoned_qwen", "reward_model"]
    for ct in checkpoint_types:
        state = get_trainer_state(ct)
        if state and "log_history" in state:
            for entry in state["log_history"]:
                metrics.append({
                    "checkpoint_type": ct,
                    **entry,
                })

    return metrics


def get_evaluation_summary() -> dict:
    """Get a summary of all evaluation results."""
    summary = {
        "poisoning": get_poisoning_results(),
        "full_eval": get_full_eval_results(),
        "anomaly_threshold": get_anomaly_threshold(),
    }

    # Extract key metrics
    if summary["poisoning"]:
        p = summary["poisoning"]
        if "pattern_detection" in p:
            summary["pattern_recall"] = p["pattern_detection"].get("combined_recall", 0)
            summary["pattern_fpr"] = p["pattern_detection"].get("combined_fpr", 0)

    return summary


# CLI entry point for testing
if __name__ == "__main__":
    print("=== SAGA Data Ingestion ===\n")

    print("Checkpoints:")
    for cp in get_all_checkpoints():
        status = "✓" if cp.get("has_final") else "✗"
        size = f" ({cp.get('size_mb', '?')} MB)" if cp.get("size_mb") else ""
        print(f"  {status} {cp['type']}{size}")

    print("\nAnomaly Threshold:")
    threshold = get_anomaly_threshold()
    if threshold:
        print(f"  τ = {threshold['tau']:.6f} (FPR = {threshold['empirical_fpr']:.2%})")

    print("\nPoisoning Results:")
    poisoning = get_poisoning_results()
    if poisoning and "pattern" in poisoning:
        pd = poisoning["pattern"]
        print(f"  Pattern Recall: {pd.get('combined_recall', 0):.1%}")
        print(f"  Pattern FPR: {pd.get('combined_fpr', 0):.1%}")

    print("\nTensorBoard Runs:")
    for run in get_training_runs_from_tensorboard():
        print(f"  {run['directory']}: {run['file_count']} files, last: {run['last_modified']}")
