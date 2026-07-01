# scripts/ — Pipeline Scripts

## Purpose

Numbered executable scripts (00–10) that drive the MoA pipeline end-to-end, plus diagnostics.

## Ownership

Each script is self-contained with its own `main()`, argument parsing, and logging. Scripts import from `src/` — never duplicate library code.

## Local Contracts

Pipeline order (sequential dependencies):

| Script | Purpose | Depends On |
|--------|---------|------------|
| `00_smoke_test.py` | Validate alignment hypothesis (inline projectors, 200 C4 prompts, paired t-test) | None |
| `01_generate_oracle_labels.py` | Generate router training labels (MMLU 2000 + GSM8K 500) | Trained projectors |
| `02_train_alignment.py` | InfoNCE alignment training | Data from 00 |
| `03_train_router_oracle.py` | Router training with cross-entropy on oracle labels | 01, 02 |
| `04_train_autoencoder.py` | Autoencoder training on clean projected embeddings (5000 prompts) | 02 |
| `05_calibrate_anomaly_threshold.py` | Select tau for target 5% FPR | 04 |
| `06_train_poisoned_model.py` | Poison Qwen-0.5B with LoRA backdoor (trigger: "Year: 2024") | None |
| `06_train_router_rlaif.py` | RLAIF training (placeholder) | 03, 04, 05 |
| `07_finetune_meta_model.py` | Meta-model SFT data generation + LoRA fine-tuning | 03 |
| `08_run_poisoning_eval.py` | Poisoning detection evaluation | All above |
| `09_integration_test.py` | End-to-end smoke test (4 prompts) | All above |
| `10_full_evaluation.py` | Full Phase 1 evaluation against success criteria | All above |
| `diagnose_alignment.py` | Alignment quality check (code/math/wiki domains) | 02 |

## Work Guidance

- Scripts are numbered by execution order — respect dependencies
- Each script logs to TensorBoard; check `runs/` for output
- Configs loaded from `configs/` — never hardcode hyperparameters in scripts
- All scripts accept `--device` argument for GPU selection
- Checkpoint directories follow `checkpoints/{script_name}/` convention

## Verification

- Run `pytest tests/` before executing any pipeline script
- `00_smoke_test.py` is a lightweight validation — run it first
- `09_integration_test.py` validates the full pipeline end-to-end
