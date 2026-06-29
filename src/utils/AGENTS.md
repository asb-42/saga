# src/utils/ — Utilities

## Purpose

Checkpointing and logging infrastructure for training pipelines.

## Ownership

- `checkpointing.py` — `save_checkpoint()`, `load_checkpoint()`, `find_latest_checkpoint()`
- `logging_utils.py` — `LocalLogger`

## Local Contracts

- Checkpoints save full state: model, optimizer, scheduler, global_step, config
- Checkpoint directory structure: `{save_dir}/checkpoint-{step}/`
- `find_latest_checkpoint()` returns path to most recent step directory
- `LocalLogger` wraps TensorBoard (always) + optional MLflow
- MLflow enabled via `MLFLOW_TRACKING_URI` env var

## Work Guidance

- Always use `save_checkpoint()` — never save raw `state_dict()` alone
- Log scalars with `LocalLogger.log_scalar()` — include step number
- Checkpoint format must be compatible across training scripts

## Verification

- No dedicated tests — tested implicitly via training scripts
