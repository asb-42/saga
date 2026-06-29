# src/router/ — Transformer Router & Anomaly Detection

## Purpose

Route embeddings to relevant models, detect poisoned/backdoored embeddings via autoencoder reconstruction, and gate anomalous contributions.

## Ownership

- `transformer_router.py` — `TransformerRouter`
- `autoencoder.py` — `AnomalyAutoencoder`
- `gating.py` — `AnomalyGate`, `calibrate_threshold()`
- `rl_trainer.py` — `RouterRLTrainer`

## Local Contracts

- Router: 2-layer transformer encoder, 8 attention heads, learned model-position embeddings, top-k=2
- Autoencoder: symmetric bottleneck 1024 → 256 → 32 → 256 → 1024, trained on clean embeddings only
- Anomaly score: MSE reconstruction error (higher = more anomalous)
- Gating formula: `w_i * min(1, tau / s_i)` where `s_i` = anomaly score, `tau` = threshold
- Target false positive rate: 5% (configurable in `configs/router.yaml`)
- Two training stages: oracle cross-entropy → RLAIF with KL anchor to oracle policy
- RLAIF uses frozen independent reward model, REINFORCE with KL penalty (coeff=0.1)

## Work Guidance

- Never train autoencoder on poisoned data — clean embeddings only
- Threshold calibration must use held-out clean set, not training set
- Router `route()` returns sparse weights during training (top-k masked), full weights during eval
- RLAIF trainer uses `RouterRLTrainer` — do not mix with oracle training loop
- Anomaly detection is defense-critical — changes here require full test suite pass

## Verification

- `pytest tests/test_router.py` — 9 tests (shapes, weights, top-k, eval mode, determinism, grads)
- `pytest tests/test_autoencoder.py` — 7 tests (shape, scores, overfit, noise, determinism, grads, bottleneck)
- `pytest tests/test_gating.py` — 7 tests (shapes, sum-to-1, zero/large anomaly, calibration)
- `scripts/03_train_router_oracle.py` — oracle training
- `scripts/04_train_autoencoder.py` — autoencoder training
- `scripts/05_calibrate_anomaly_threshold.py` — threshold selection
- `scripts/06_train_router_rlaif.py` — RLAIF training (placeholder)
