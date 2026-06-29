# src/models/ — Model Loading & Inference

## Purpose

Load frozen base models with sequential GPU offloading, encode embeddings, and run weighted ensemble inference.

## Ownership

- `loader.py` — `FrozenModelWrapper`, `load_all_models()`, `sequential_encode()`
- `inference.py` — `PipelineOutput`, `generate_from_models()`, `weighted_ensemble_answer()`

## Local Contracts

- `FrozenModelWrapper` handles lazy CPU init + sequential GPU transfer — never load models outside this wrapper
- `encode()` returns mean-pooled last hidden state as `torch.Tensor` shape `(hidden_dim,)`
- `generate()` returns decoded text string
- `weighted_ensemble_answer()` is the full MoA pipeline entry point: encode → project → route → autoencoder → gate → generate → synthesize
- Models defined in `configs/models.yaml` with pinned commits in `configs/model_commits.json`

## Work Guidance

- All models use `torch.float16` on GPU, `torch.float32` on CPU
- Tokenizer type varies per model (BPE, SentencePiece) — always use model's own tokenizer
- Mean-pooling respects attention mask to avoid padding contamination
- `PipelineOutput` is a frozen dataclass — extend by adding fields, not modifying

## Verification

- `pytest tests/test_inference.py` — 4 tests (PipelineOutput defaults, generate_from_models mock)
- `pytest tests/test_pipeline.py` — 4 tests (immutable defaults, anomaly flags, weight sum)
