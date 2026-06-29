# src/alignment/ — Embedding Alignment

## Purpose

Map heterogeneous model embeddings into a shared 1024-dimensional space via MLP projectors trained with InfoNCE contrastive loss.

## Ownership

- `projector.py` — `MLPProjector`, `ProjectorBank`
- `loss.py` — `InfoNCELoss`, `compute_retrieval_accuracy()`, `stack_embeddings()`
- `trainer.py` — `train_alignment()` full training loop

## Local Contracts

- Projector architecture: `Linear(hidden_dim, hidden_dim) → GELU → Dropout(0.1) → Linear(hidden_dim, hidden_dim)` — all projectors output 1024d
- InfoNCE temperature: 0.07 (configurable in `configs/alignment.yaml`)
- Training data: C4 (50k) + WikiText (20k) from HuggingFace datasets
- Validation metric: cross-model nearest-neighbor retrieval accuracy
- AdamW optimizer with cosine annealing LR schedule
- Each base model gets its own projector in `ProjectorBank` (`nn.ModuleDict`)

## Work Guidance

- Always L2-normalize embeddings before InfoNCE loss (`stack_embeddings()`)
- `ProjectorBank` keys must match model names from `configs/models.yaml`
- When adding a new model, add a matching projector entry
- Dropout only active during training; projectors have `eval()` mode for inference

## Verification

- `pytest tests/test_projector.py` — 3 stub tests (implement before merging)
- `scripts/02_train_alignment.py` — launch alignment training
- `scripts/diagnose_alignment.py` — check quality on diverse held-out data
- Success criteria: retrieval accuracy > 0.8 across model pairs
