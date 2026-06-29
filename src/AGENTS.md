# src/ — Core Library

## Purpose

Mixture of Agents (MoA) coordination layer: embedding alignment, routing, anomaly detection, synthesis, and inference orchestration for poisoning-resilient AI ensemble.

## Ownership

All Python library code lives here. Each subpackage owns a discrete capability. Public API flows through `inference.py` and `orchestrator/pipeline.py`.

## Local Contracts

- Common embedding dimension: 1024 (set in `configs/models.yaml`, enforced by all projectors)
- All base models are frozen; only projectors, router, autoencoder, and meta-model are trainable
- Sequential GPU offloading: only one base model on GPU at a time (`models/loader.py`)
- Mean-pooling over last hidden state for encoding (`models/loader.py:encode()`)
- InfoNCE contrastive loss for alignment training (`alignment/loss.py`)
- Anomaly detection: autoencoder MSE reconstruction error against calibrated threshold (`router/autoencoder.py`, `router/gating.py`)

## Work Guidance

- Follow existing code patterns: dataclasses for structured output, `nn.Module` for trainable components
- Type hints required; prefer `torch.Tensor` over raw arrays
- All model loading goes through `FrozenModelWrapper` — never instantiate base models directly
- Checkpointing via `utils/checkpointing.py` — save full state (model, optimizer, scheduler, step)
- Logging via `utils/logging_utils.py` — TensorBoard always, MLflow optional

## Verification

- `pytest tests/` — 34 tests across router, autoencoder, gating, inference, pipeline, projector
- Run `scripts/00_smoke_test.py` to validate alignment hypothesis end-to-end

## Child DOX Index

| Path | Covers |
|------|--------|
| `models/AGENTS.md` | Model loading, frozen wrappers, encoding, inference |
| `alignment/AGENTS.md` | Projectors, InfoNCE loss, alignment training |
| `router/AGENTS.md` | Transformer router, autoencoder, anomaly gating, RLAIF |
| `meta_model/AGENTS.md` | Synthesis judge, LoRA fine-tuning |
| `orchestrator/AGENTS.md` | Full MoA pipeline orchestration |
| `evaluation/AGENTS.md` | Benchmarks (MMLU, GSM8K, BBQ), metrics, poisoning eval |
| `utils/AGENTS.md` | Checkpointing, logging |
