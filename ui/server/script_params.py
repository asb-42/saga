"""Script parameter definitions for the pipeline UI.

SCRIPT_PARAMS maps script IDs (as used in the pipeline) to their
user-facing CLI arguments. Each param has:
  - type: int | float | select | multi | flag
  - default: default value
  - label: human-readable name for the UI
  - min / max: optional bounds for numeric types
  - choices: optional list for select/multi types

To add a new parameter to a script:
  1. Add the argparse argument to the script
  2. Add the corresponding entry here
  3. Run: python -m pytest ui/tests/test_script_params_sync.py
"""

# Type hints for param definitions
# {arg_name: {type, default, label, [min], [max], [choices]}}
SCRIPT_PARAMS: dict[str, dict[str, dict]] = {
    "11_raw_baseline": {
        "benchmarks": {
            "type": "multi", "default": ["arc_easy", "hellaswag", "winogrande", "boolq"],
            "label": "Benchmarks",
            "choices": ["arc_easy", "hellaswag", "winogrande", "boolq", "mmlu", "gsm8k", "humaneval", "bbq"],
        },
        "max-samples": {
            "type": "int", "default": None, "label": "Max Samples (per benchmark)",
            "min": 10, "max": 5000,
        },
    },
    "00_smoke_test": {
        "num-prompts": {
            "type": "int", "default": 200, "label": "Number of Prompts",
            "min": 10, "max": 5000,
        },
        "seed": {
            "type": "int", "default": 42, "label": "Random Seed",
            "min": 0, "max": 99999,
        },
    },
    "01_generate_oracle_labels": {
        "mmlu-samples": {
            "type": "int", "default": 2000, "label": "MMLU Samples",
            "min": 10, "max": 10000,
        },
        "gsm8k-samples": {
            "type": "int", "default": 500, "label": "GSM8K Samples",
            "min": 10, "max": 5000,
        },
        "max-samples": {
            "type": "int", "default": None, "label": "Max Samples (global cap)",
            "min": 10, "max": 10000,
        },
        "oracle-mode": {
            "type": "select", "default": "judge_ppl_fallback", "label": "Oracle Mode",
            "choices": ["exact_match", "judge", "judge_ppl_fallback"],
        },
        "seed": {
            "type": "int", "default": 42, "label": "Random Seed",
            "min": 0, "max": 99999,
        },
    },
    "02_train_alignment": {},
    "02b_train_alignment_structured": {
        "structure-weight": {
            "type": "float", "default": 0.1, "label": "Structure Weight (λ)",
            "min": 0.01, "max": 1.0,
            "help": "λ controls structure preservation. Too low: Qwen-centric assimilation. Too high: alignment degrades. Start at 0.1.",
        },
    },
    "04_train_autoencoder": {
        "num-prompts": {
            "type": "int", "default": 5000, "label": "Training Prompts",
            "min": 100, "max": 20000,
        },
    },
    "05_calibrate_threshold": {
        "num-prompts": {
            "type": "int", "default": 1000, "label": "Calibration Prompts",
            "min": 100, "max": 5000,
        },
    },
    "06_train_router_rlaif": {},
    "06_train_poisoned": {
        "num-triggered": {
            "type": "int", "default": 100, "label": "Triggered Samples",
            "min": 10, "max": 2000,
        },
        "num-clean": {
            "type": "int", "default": 900, "label": "Clean Samples",
            "min": 10, "max": 5000,
        },
        "seed": {
            "type": "int", "default": 42, "label": "Random Seed",
            "min": 0, "max": 99999,
        },
    },
    "07_finetune_meta": {
        "num-examples": {
            "type": "int", "default": 5000, "label": "SFT Examples",
            "min": 100, "max": 20000,
        },
        "epochs": {
            "type": "int", "default": 3, "label": "Epochs",
            "min": 1, "max": 50,
        },
        "lr": {
            "type": "float", "default": 2e-5, "label": "Learning Rate",
            "min": 1e-6, "max": 1e-3,
        },
        "batch-size": {
            "type": "int", "default": 2, "label": "Batch Size",
            "min": 1, "max": 32,
        },
        "generate-data": {
            "type": "flag", "default": False, "label": "Regenerate SFT Data",
        },
    },
    "08_eval": {
        "num-prompts": {
            "type": "int", "default": 200, "label": "Evaluation Prompts",
            "min": 10, "max": 2000,
        },
    },
    "08b_eval_answer_level": {
        "num-prompts": {
            "type": "int", "default": 100, "label": "Evaluation Prompts",
            "min": 10, "max": 2000,
        },
    },
    "09_train_reward_model": {
        "num-examples": {
            "type": "int", "default": 5000, "label": "Training Examples",
            "min": 100, "max": 20000,
        },
        "num-epochs": {
            "type": "int", "default": 5, "label": "Epochs",
            "min": 1, "max": 50,
        },
        "learning-rate": {
            "type": "float", "default": 5e-5, "label": "Learning Rate",
            "min": 1e-6, "max": 1e-3,
        },
        "batch-size": {
            "type": "int", "default": 2, "label": "Batch Size",
            "min": 1, "max": 32,
        },
        "gradient-accumulation": {
            "type": "int", "default": 8, "label": "Gradient Accumulation Steps",
            "min": 1, "max": 64,
        },
        "warmup-ratio": {
            "type": "float", "default": 0.1, "label": "Warmup Ratio",
            "min": 0.0, "max": 0.5,
        },
        "max-length": {
            "type": "int", "default": 512, "label": "Max Sequence Length",
            "min": 64, "max": 2048,
        },
        "seed": {
            "type": "int", "default": 42, "label": "Random Seed",
            "min": 0, "max": 99999,
        },
    },
    "10_full_eval": {
        "benchmarks": {
            "type": "multi", "default": ["arc_easy", "hellaswag", "winogrande", "boolq"],
            "label": "Benchmarks",
            "choices": ["arc_easy", "hellaswag", "winogrande", "boolq", "mmlu", "gsm8k", "humaneval", "bbq"],
        },
        "max-samples": {
            "type": "int", "default": None, "label": "Max Samples (per benchmark)",
            "min": 10, "max": 5000,
        },
        "ensemble-strategy": {
            "type": "select", "default": "majority_vote",
            "label": "Ensemble Strategy",
            "choices": ["judge", "majority_vote", "best_model"],
        },
        "individual-only": {
            "type": "flag", "default": False, "label": "Individual Models Only",
        },
        "ensemble-only": {
            "type": "flag", "default": False, "label": "Ensemble Only",
        },
    },
    "99_demo_training": {
        "epochs": {
            "type": "int", "default": 10, "label": "Demo Epochs",
            "min": 1, "max": 100,
        },
        "steps-per-epoch": {
            "type": "int", "default": 50, "label": "Steps per Epoch",
            "min": 5, "max": 500,
        },
    },
}

# Mapping from pipeline script IDs to actual script filenames.
# Keys are the IDs used in the pipeline UI, values are the .py filenames.
SCRIPT_FILE_MAP: dict[str, str] = {
    "11_raw_baseline": "11_raw_baseline.py",
    "00_smoke_test": "00_smoke_test.py",
    "01_generate_oracle_labels": "01_generate_oracle_labels.py",
    "02_train_alignment": "02_train_alignment.py",
    "02b_train_alignment_structured": "02b_train_alignment_structured.py",
    "03_train_router": "03_train_router_oracle.py",
    "04_train_autoencoder": "04_train_autoencoder.py",
    "05_calibrate_threshold": "05_calibrate_anomaly_threshold.py",
    "06_train_router_rlaif": "06_train_router_rlaif.py",
    "06_train_poisoned": "06_train_poisoned_model.py",
    "07_finetune_meta": "07_finetune_meta_model.py",
    "08_eval": "08_run_poisoning_eval.py",
    "08b_eval_answer_level": "08b_run_poisoning_eval_answer_level.py",
    "09_train_reward_model": "09_train_reward_model.py",
    "09_integration_test": "09_integration_test.py",
    "10_full_eval": "10_full_evaluation.py",
    "99_demo_training": "99_demo_training.py",
}


def get_script_filename(script_id: str) -> str | None:
    """Get the actual .py filename for a pipeline script ID."""
    return SCRIPT_FILE_MAP.get(script_id)
