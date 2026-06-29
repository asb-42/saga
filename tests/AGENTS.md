# tests/ — Test Suite

## Purpose

Pytest test suite covering router, autoencoder, gating, inference, pipeline, and projector components.

## Ownership

- `test_router.py` — 9 tests (TransformerRouter)
- `test_autoencoder.py` — 7 tests (AnomalyAutoencoder)
- `test_gating.py` — 7 tests (AnomalyGate + calibration)
- `test_inference.py` — 4 tests (PipelineOutput, generate_from_models)
- `test_pipeline.py` — 4 tests (MoAPipeline defaults, invariants)
- `test_projector.py` — 3 stub tests (empty implementations)

## Local Contracts

- All tests use `pytest` — no unittest framework
- Tests use mock models where possible to avoid GPU dependency
- Tests validate tensor shapes, gradient flow, determinism, and invariants
- `test_projector.py` stubs must be implemented before projector changes merge
- No fixtures that download models from HuggingFace — mock all model loads

## Work Guidance

- Add tests for any new component before merging
- Test both training and eval modes where applicable
- Validate weight sum invariants (router weights sum to 1, gating weights sum to 1)
- Test batch independence (batch size 1 == batch size N for single-sample inference)
- Test determinism with same seed for reproducibility

## Verification

- `pytest tests/ -v` — full suite (34 tests)
- `pytest tests/test_router.py -v` — router-only
- `pytest tests/test_autoencoder.py -v` — autoencoder-only
- All tests must pass before pipeline script execution
