# saga

**Mixture of Agents Coordination Layer** — a geopolitically independent,
poisoning-resilient trust protocol for AI inference and distributed
fine-tuning.

## Overview

saga orchestrates heterogeneous open-weight language models as an
inference-time ensemble (Mixture of Agents). It provides:

- **Embedding Alignment** — per-model MLP projectors that map incompatible
  latent spaces into a common semantic space via InfoNCE contrastive learning.
- **Anomaly-Gated Router** — a small transformer that routes prompts to the
  best models while an autoencoder detects poisoned/backdoored embeddings.
- **Synthesis Meta-Model** — an instruction-tuned judge that aggregates
  multi-model answers and flags inconsistencies.

This is Phase 1 of the saga project: a trustworthy inference ensemble that
can detect injected backdoors while matching or exceeding the best single
model on capability benchmarks.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download and smoke-test all models
python scripts/00_smoke_test.py

# Run the full Phase 1 pipeline
python scripts/01_generate_oracle_labels.py --num-prompts 1000
python scripts/02_train_alignment.py
python scripts/03_train_router_oracle.py
python scripts/04_train_autoencoder.py
python scripts/05_calibrate_anomaly_threshold.py
python scripts/06_train_router_rlaif.py
python scripts/07_finetune_meta_model.py --generate-data
python scripts/08_run_poisoning_eval.py
python scripts/09_integration_test.py
python scripts/10_full_evaluation.py
```

## Architecture

```
User Prompt
    │
    ▼
┌───────────────────────────────────────┐
│           Base Models (frozen)         │
│  Qwen2.5-0.5B  Falcon-RW-1B  SmolLM  │
│       │              │            │    │
│   hidden states   hidden states  ...   │
│       │              │            │    │
│       ▼              ▼            ▼    │
│   Projector A    Projector B  Projector C│  ← Embedding Alignment
│       │              │            │    │
│       └──────────────┼────────────┘    │
│                      ▼                 │
│              Common Space (1024d)      │
│                      │                 │
│          ┌───────────┴───────────┐     │
│          ▼                       ▼     │
│    Router (top-k)         Autoencoder   │
│    routing weights         anomaly score│
│          │                       │     │
│          └───────────┬───────────┘     │
│                      ▼                 │
│              Gating: w_i · min(1, τ/s) │
│                      │                 │
│                      ▼                 │
│              Weighted Ensemble          │
│                      │                 │
│                      ▼                 │
│           Meta-Model (Synthesis)        │
│                      │                 │
│                      ▼                 │
│                Final Answer             │
└───────────────────────────────────────┘
```

## Phase 1 Success Criteria

| Criterion | Target |
|-----------|--------|
| Ensemble ≥ best single model | ≥2 of 4 benchmarks |
| Backdoor detection recall | ≥90% |
| False-positive rate | <5% |
| Anomaly score R² with poisoning | >0.7 |

## Project Structure

```
.
├── configs/          # YAML configuration files
├── src/              # Library code
│   ├── models/       # Model loading & inference
│   ├── alignment/    # Embedding projectors & contrastive loss
│   ├── router/       # Transformer router, autoencoder, gating, RL
│   ├── meta_model/   # Synthesis judge & fine-tuning
│   ├── orchestrator/ # Full pipeline orchestration
│   ├── evaluation/   # Benchmarks, poisoning tests, metrics
│   └── utils/        # Checkpointing, logging
├── scripts/          # Run scripts (00–10)
├── tests/            # Pytest suite
├── data/             # Oracle labels, SFT data, canary triggers
└── Dockerfile        # Reproducible runtime
```

## License

Open-source. See LICENSE file.
