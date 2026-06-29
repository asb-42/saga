# src/orchestrator/ — Pipeline Orchestration

## Purpose

High-level MoA pipeline wrapper: load models, run inference, batch processing.

## Ownership

- `pipeline.py` — `MoAPipeline`

## Local Contracts

- `MoAPipeline` is a thin wrapper over `inference.py` functions
- `load()`, `run()`, `run_batch()` — all currently `NotImplementedError` stubs
- Actual pipeline logic lives in `src/models/inference.py:weighted_ensemble_answer()`

## Work Guidance

- Implement `MoAPipeline` when you need stateful pipeline management (model caching, config persistence)
- For simple inference, use `weighted_ensemble_answer()` directly
- Keep this module minimal — orchestration should delegate, not duplicate

## Verification

- No dedicated tests — implement alongside `MoAPipeline` stubs
- `scripts/09_integration_test.py` — end-to-end smoke test
