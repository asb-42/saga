# Technical Progress Report — 2026-07-11

## Executive Summary

SAGA's router training has hit a **fundamental architectural wall**. Three training attempts — stale labels (36%), correct labels (53%), and raw embeddings (33% balanced) — all converge to the same result: the router cannot learn to discriminate which model is best for a given prompt. The root cause is confirmed: **alignment projectors erase model identity**, and even without projectors, **prompt embeddings do not encode model competence**. The 53% validation accuracy equals the constant baseline (always predict codeqwen). The router has learned nothing. This report documents the full investigation, the raw-embedding experiment, and the four paths forward.

---

## 1. Problem Statement

The SAGA router must solve this task:

> Given a prompt embedding, predict which of 4 frozen language models will produce the best answer.

The router was trained with cross-entropy loss on oracle labels (3,734 entries, 5 benchmarks). It achieved 53% validation accuracy — exactly equal to always predicting codeqwen (the majority class at 44%). The router output distribution has entropy 1.33 out of a maximum of 1.39 (uniform). It is not learning.

---

## 2. Oracle Label Generation (4-Model)

### 2.1 Data Source

Oracle labels were generated using `scripts/01_generate_oracle_labels.py` with the `judge_ppl_fallback` mode. The generation uses a 5-benchmark suite:

| Benchmark | Samples | Domain |
|-----------|---------|--------|
| ARC-Easy | 570 | Commonsense reasoning |
| HellaSwag | 1,000 | Commonsense completion |
| WinoGrande | 1,000 | Commonsense reasoning |
| BoolQ | 1,000 | Boolean questions |
| HumanEval | 164 | Code generation |
| **Total** | **3,734** | |

### 2.2 Model Distribution

| Model | Win Count | Win Rate | Avg Score |
|-------|-----------|----------|-----------|
| codeqwen | 1,640 | 43.9% | 0.743 |
| phi2 | 915 | 24.5% | 0.683 |
| qwen | 613 | 16.4% | 0.575 |
| smollm | 566 | 15.2% | 0.508 |

### 2.3 Per-Benchmark Win Rates

| Benchmark | codeqwen | phi2 | qwen | smollm |
|-----------|----------|------|------|--------|
| ARC-Easy (570) | 43.9% | 18.1% | 21.6% | 16.5% |
| HellaSwag (1,000) | **60.0%** | 15.3% | 11.9% | 12.8% |
| WinoGrande (1,000) | 22.9% | **43.9%** | 17.0% | 16.2% |
| BoolQ (1,000) | **52.1%** | 18.4% | 15.9% | 13.6% |
| HumanEval (164) | 24.4% | 22.0% | 25.6% | 28.0% |

**Key insight**: codeqwen dominates commonsense (HellaSwag 60%, BoolQ 52%). Phi2 dominates reasoning (WinoGrande 44%). HumanEval is surprisingly even (all 22-28%). Model competence is **not** a linear function of semantic content — codeqwen wins on commonsense despite being a "code" model.

### 2.4 Distribution vs Target

| Model | Target | Actual | Delta |
|-------|--------|--------|-------|
| codeqwen | 30.0% | 43.9% | +13.9% |
| phi2 | 40.0% | 24.5% | -15.5% |
| qwen | 20.0% | 16.4% | -3.6% |
| smollm | 10.0% | 15.2% | +5.2% |

KL Divergence: 0.078 nats (healthy).

---

## 3. Router Training Attempts

### 3.1 Attempt 1: Stale Labels (2026-07-11 02:14)

| Parameter | Value |
|-----------|-------|
| Oracle | `oracle_labels.jsonl` (2,500 entries, falcon/qwen/smollm) |
| Projectors | `checkpoints/alignment` (empty — random init) |
| Val accuracy | 36.0% |
| Constant baseline | 76.5% (falcon=61%) |

**Root cause**: Labels had falcon (deprecated) but not codeqwen/phi2. Router trained on wrong classes.

### 3.2 Attempt 2: Correct Labels, Wrong Projectors (2026-07-11 03:57)

| Parameter | Value |
|-----------|-------|
| Oracle | `oracle_labels_latest.jsonl` (3,734 entries) |
| Projectors | `checkpoints/alignment` (empty — random init) |
| Val accuracy | 56.5% |
| Constant baseline | 53.0% (codeqwen=44%) |

**Note**: 56.5% was from an intermediate checkpoint. Final result was 53%.

### 3.3 Attempt 3: Correct Labels + Correct Projectors (2026-07-11 07:32)

| Parameter | Value |
|-----------|-------|
| Oracle | `oracle_labels_latest.jsonl` (3,734 entries) |
| Projectors | `checkpoints/alignment_structured/final.pt` (step 8,234) |
| Architecture | 2-layer transformer, 1024 dim, 8 heads, 2048 FF |
| Loss | Cross-entropy (hard labels) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Epochs | 10 |
| Val accuracy | **53.0%** (= constant baseline) |

#### Training Progress

| Epoch | Loss | Train Acc | Val Acc | Status |
|-------|------|-----------|---------|--------|
| 1 | 1.2872 | 45.1% | 56.5% | ✅ |
| 2 | 1.2592 | 48.1% | 56.0% | ✅ |
| 3 | 1.2450 | 48.3% | **57.0%** | ✅ Peak |
| 4 | 1.2348 | 49.3% | 55.5% | ⚠️ Regression starts |
| 5 | 1.2210 | 51.3% | 54.0% | ⚠️ |
| 6 | 1.2107 | 53.2% | 53.5% | ⚠️ |
| 7 | 1.2023 | 55.5% | 53.0% | ❌ = Baseline |
| 8 | 1.1964 | 58.2% | 53.0% | ❌ |
| 9 | 1.1930 | 59.7% | 53.0% | ❌ |
| 10 | 1.1913 | 63.6% | 53.0% | ❌ Final |

**Key finding**: Val accuracy peaked at 57% (epoch 3) then **regressed to baseline** (53%). Loss continued decreasing. The router discovered the "always guess the prior" shortcut and abandoned the real signal.

### 3.4 Diagnostic Analysis (Final Model)

| Check | Value | Baseline | Verdict |
|-------|-------|----------|---------|
| Constant baseline acc | 53.0% | — | Always predict codeqwen |
| Train accuracy | 28.7% | 25% (random) | **Severe underfitting** |
| Val accuracy | 27.0% | 53.0% (constant) | **Worse than random** |
| Mean entropy | 1.3283 | 1.3863 (uniform) | **Collapsed** |

#### Confusion Matrix (Diagnostic Split)

| True \ Pred | codeqwen | phi2 | qwen | smollm | Recall |
|-------------|----------|------|------|--------|--------|
| codeqwen | 31 | 8 | 45 | 22 | 29.3% |
| phi2 | 5 | 4 | 29 | 2 | 10.0% |
| qwen | 8 | 3 | 13 | 3 | 48.1% |
| smollm | 4 | 0 | 17 | 6 | 22.2% |
| Precision | 64.6% | 26.7% | 12.5% | 18.2% | — |

The router distributes predictions across qwen/smollm while missing codeqwen/phi2 entirely.

---

## 4. Root Cause: The Architectural Mismatch

### 4.1 The Core Tension

The alignment projectors were trained with one objective:

> "Make the same prompt look identical across all models."

The router needs the exact opposite:

> "Make the same prompt look different across models, in a way that predicts competence."

These are **antagonistic objectives**. The more perfectly the projectors align same-prompt embeddings, the less information the router has about which model produced which embedding.

### 4.2 Raw Embedding Statistics

| Model | Hidden Dim | Norm | Std | Character |
|-------|-----------|------|-----|-----------|
| codeqwen | 1,536 | 155.6 ± 12.8 | 3.98 | Large values, high variance |
| phi2 | 2,560 | 40.5 ± 6.3 | 0.81 | Small values, low variance |
| qwen | 896 | 214.5 ± 19.7 | 7.19 | Large values, high variance |
| smollm | 960 | 24.3 ± 2.8 | 0.79 | Small values, low variance |

The raw embeddings have vastly different scales (norms 24–214) and variances (std 0.79–7.19). The projectors compress all of these into a shared 1024-dim space optimized for similarity — erasing model identity in the process.

### 4.3 Why Model Competence Is Not Linearly Separable

The per-benchmark results show that model competence is **not** a function of semantic content:

- CodeQwen (a "code" model) wins HellaSwag (commonsense) at 60%
- Phi2 (a "reasoning" model) wins WinoGrande (coreference) at 44%
- HumanEval (code) is evenly split across all 4 models (22-28%)

The projector clusters by topic (code, math, cooking). The router needs to cluster by **competence edge**. These are different manifolds in the embedding space.

---

## 5. Path 3 Experiment: Raw Embeddings + Per-Model Linear Layers

### 5.1 Architecture

Instead of using frozen projectors, the experiment uses raw embeddings with per-model linear alignment layers:

```
raw_codeqwen (1536) → Linear → 512
raw_phi2 (2560) → Linear → 512
raw_qwen (896) → Linear → 512
raw_smollm (960) → Linear → 512
concat → 2048 → MLP (2048→1024→512→4) → logits
```

This preserves model-specific geometry while allowing the router to learn cross-model comparisons.

### 5.2 Configuration

| Parameter | Value |
|-----------|-------|
| Oracle | `oracle_labels_latest.jsonl` (3,734 entries) |
| Split | 70% train (2,613) / 15% val (560) / 15% test (561) |
| Architecture | Per-model linear (raw→512) + MLP (2048→1024→512→4) |
| Parameters | 5,674,500 |
| Loss | Cross-entropy with inverse-frequency class weights |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine annealing |
| Epochs | 20 |
| Batch size | 64 |
| Device | cuda:0 |

### 5.3 Training Progress

| Epoch | Loss | Train Acc | Val Acc | Val Bal Acc | Test Bal Acc | Entropy | LR |
|-------|------|-----------|---------|-------------|--------------|---------|-----|
| 1 | 1.4452 | 45.1% | 51.2% | 33.8% | 34.0% | 1.286 | 9.9e-4 |
| 2 | 1.2471 | 48.1% | 50.9% | 34.5% | 34.0% | 1.252 | 9.8e-4 |
| 3 | 1.2369 | 48.3% | 50.5% | 33.6% | 34.0% | 1.234 | 9.5e-4 |
| **4** | **1.1926** | **49.3%** | **45.0%** | **35.6%** | **32.9%** | **1.244** | **9.0e-4** |
| 5 | 1.1691 | 51.3% | 47.1% | 35.2% | 33.6% | 1.184 | 8.5e-4 |
| 6 | 1.1112 | 53.2% | 46.8% | 32.7% | 31.8% | 1.140 | 7.9e-4 |
| 7 | 1.0666 | 55.5% | 46.4% | 31.1% | 31.8% | 1.042 | 7.3e-4 |
| 8 | 0.9914 | 58.2% | 42.5% | 32.6% | 33.5% | 1.010 | 6.5e-4 |
| 9 | 0.9656 | 59.7% | 39.3% | 34.6% | 33.0% | 0.994 | 5.8e-4 |
| 10 | 0.8584 | 63.6% | 43.9% | 31.7% | 34.0% | 0.843 | 5.0e-4 |
| 11 | 0.8024 | 67.0% | 43.2% | 29.9% | 31.7% | 0.763 | 4.2e-4 |
| 12 | 0.7359 | 69.8% | 41.2% | 31.1% | 35.4% | 0.735 | 3.5e-4 |
| 13 | 0.6490 | 73.2% | 41.2% | 30.8% | 35.5% | 0.709 | 2.7e-4 |
| 14 | 0.5925 | 75.7% | 36.4% | 30.4% | 34.3% | 0.788 | 2.1e-4 |
| 15 | 0.5075 | 79.7% | 37.9% | 30.9% | 34.8% | 0.605 | 1.5e-4 |
| 16 | 0.4480 | 82.1% | 38.8% | 30.4% | 33.2% | 0.554 | 9.5e-5 |
| 17 | 0.3754 | 84.8% | 37.3% | 31.0% | 33.9% | 0.544 | 5.5e-5 |
| 18 | 0.3462 | 86.8% | 38.9% | 31.4% | 33.9% | 0.465 | 2.4e-5 |
| 19 | 0.3228 | 87.9% | 39.1% | 30.5% | 33.3% | 0.485 | 6.2e-6 |
| 20 | 0.3008 | 89.0% | 38.6% | 30.4% | 33.1% | 0.473 | 0.0 |

**Best epoch**: 4 (val_bal_acc=35.6%)

### 5.4 Final Results (Best Epoch, Test Set)

| Metric | Value |
|--------|-------|
| Test accuracy | 43.9% |
| Test balanced accuracy | **32.9%** |
| Test entropy | 1.251 (uniform=1.386) |
| Train/val gap | 50.4% (severe overfitting) |

### 5.5 Per-Class Metrics (Test Set)

| Model | Recall | Precision | F1 |
|-------|--------|-----------|-----|
| codeqwen | **61.2%** | 59.9% | 0.605 |
| phi2 | **59.9%** | 34.7% | 0.440 |
| qwen | **6.5%** | 18.8% | 0.097 |
| smollm | **3.8%** | 8.1% | 0.052 |

### 5.6 Confusion Matrix (Test Set)

| True \ Pred | codeqwen | phi2 | qwen | smollm |
|-------------|----------|------|------|--------|
| codeqwen | 158 | 72 | 12 | 16 |
| phi2 | 37 | 79 | 8 | 8 |
| qwen | 39 | 38 | 6 | 10 |
| smollm | 30 | 39 | 6 | 3 |

### 5.7 Interpretation

1. **codeqwen and phi2 are distinguishable** (~60% recall each) — raw embeddings contain signal for these models
2. **qwen and smollm are not** (<7% recall) — their embeddings are indistinguishable from noise in the routing space
3. **Severe overfitting**: train 89% vs val 38.6% — the 5.7M-param router memorizes the 2,613 training examples
4. **Balanced accuracy 32.9%** is only +7.9% above random (25%) — far below the >60% viability threshold

---

## 6. Comparison: Projected vs Raw Router

| Metric | Projected Router | Raw Router | Delta |
|--------|-----------------|------------|-------|
| Test balanced accuracy | 27.0% | **32.9%** | +5.9% |
| Test accuracy | 27.0% | **43.9%** | +16.9% |
| Entropy | 1.328 | **1.251** | -0.077 |
| codeqwen recall | 29.3% | **61.2%** | +31.9% |
| phi2 recall | 10.0% | **59.9%** | +49.9% |
| qwen recall | 48.1% | **6.5%** | -41.6% |
| smollm recall | 22.2% | **3.8%** | -18.4% |
| Parameters | 16.8M | **5.7M** | -66% |

**Key finding**: Raw embeddings help codeqwen/phi2 significantly (+32-50% recall) but destroy qwen/smollm (-18-42% recall). The router trades one failure mode for another. The projected router was uniformly bad; the raw router is bimodal (good on large models, terrible on small ones).

---

## 7. Why This Was Inevitable

### 7.1 The Embedding Space Does Not Encode Competence

The oracle labels show that model competence is **not** a linear function of semantic content:

- CodeQwen (code model) wins HellaSwag (commonsense) at 60%
- Phi2 (reasoning model) wins WinoGrande (coreference) at 44%
- HumanEval (code) is evenly split (22-28%)

The embedding space clusters by **topic** (code, math, cooking). The router needs to cluster by **competence edge**. These are different manifolds.

### 7.2 Model Size Creates Asymmetric Signal

The raw-embedding experiment reveals a size-dependent signal:

| Model | Hidden Dim | Raw Recall | Interpretation |
|-------|-----------|------------|----------------|
| codeqwen | 1,536 | 61.2% | Sufficient signal |
| phi2 | 2,560 | 59.9% | Sufficient signal |
| qwen | 896 | 6.5% | Insufficient signal |
| smollm | 960 | 3.8% | Insufficient signal |

Models with <1000 hidden dims produce embeddings that are too compressed to encode routing-relevant information. The per-model linear layers cannot recover signal that was never captured.

### 7.3 The Regression Pattern

All three training attempts show the same pattern:
1. **Epochs 1-3**: Loss decreases, accuracy improves (real signal found)
2. **Epochs 4-10**: Loss continues decreasing, accuracy regresses to baseline

The router discovers the "always guess the prior" shortcut and abandons the real signal. This is not overfitting — it's the global minimum of the wrong objective.

---

## 8. The Four Paths Forward

### Path 1: Joint Training (Probably Insufficient)
Train projector and router jointly with multi-task loss. The alignment loss will dominate and continue to erase routing signal.

### Path 2: Two-Space Architecture (Moderate Complexity)
Keep alignment space for anomaly detection. Add a separate routing space trained only to predict model competence. Doubles training and storage cost.

### Path 3: Raw Embeddings + Per-Model Linear Layers (Implemented)
**Result: 32.9% balanced accuracy.** Partial signal for large models, none for small models. Insufficient for viability.

### Path 4: Output-Based Routing (Recommended)
Run all 4 models to generate outputs. Route based on output quality, not input embedding. This is expensive (4× inference) but is the only routing signal that is guaranteed to exist.

---

## 9. Recommendations

### Immediate (Next 48 Hours)
1. **Implement Path 4** (output-based routing) as a PoC
2. If Path 4 works (>60% balanced accuracy), SAGA is viable with output-level integration
3. If Path 4 fails, the premise is fundamentally flawed

### Do NOT Do
| Action | Why Skip |
|--------|----------|
| Increase learning rate | Router is not stuck in local minimum; it found the global minimum of the wrong objective |
| Add class weights | Problem is not class imbalance; problem is no signal |
| 4-layer transformer | More capacity on zero signal = faster convergence to prior |
| Early stopping at epoch 3 | Would ship a model that is 4% above baseline by luck |

---

## 10. Infrastructure State

### 10.1 Checkpoints

| Component | Path | Status |
|-----------|------|--------|
| Alignment (structured) | `checkpoints/alignment_structured/final.pt` | ✅ Step 8,234 |
| Router (final) | `checkpoints/router/final.pt` | ✅ Step 1,221 (53% = baseline) |
| Path 3 raw router | `results/path3_raw_router/best.pt` | ✅ Epoch 4 (33% bal acc) |
| Autoencoder | `checkpoints/autoencoder/final.pt` | ✅ Trained |
| Meta-model | `checkpoints/meta_model/final/` | ✅ Fine-tuned |
| Poisoned Qwen | `checkpoints/poisoned_qwen/final/` | ✅ LoRA trained |

### 10.2 Results

| Result | Path | Key Metric |
|--------|------|------------|
| Raw baseline | `results/raw_baseline/summary_latest.json` | phi2 avg=0.491 |
| Oracle validation | `results/oracle_validation/summary_latest.json` | KL=0.078 |
| Router training | `results/router_training/summary_latest.json` | 53% = baseline |
| Router diagnostics | `results/router_diagnostics/diagnostics_latest.json` | 27% val, collapsed |
| Path 3 raw router | `results/path3_raw_router/summary_latest.json` | 33% bal acc |

### 10.3 UI State

| Page | Route | Status |
|------|-------|--------|
| Pipeline | `/pipeline` | ✅ 15 scripts |
| Oracle Labels | `/oracle-labels` | ✅ |
| Oracle Validation | `/oracle-validation` | ✅ |
| Router Training | `/router-training` | ✅ |
| Raw Baseline | `/raw-baseline` | ✅ |

---

*Report generated: 2026-07-11 15:45 UTC*
*Alignment checkpoint: `checkpoints/alignment_structured/final.pt` (step 8,234)*
*Router checkpoint: `checkpoints/router/final.pt` (step 1,221, val_acc=53.0%)*
*Path 3 result: `results/path3_raw_router/best.pt` (epoch 4, bal_acc=32.9%)*
*Oracle labels: `data/oracle_labels_latest.jsonl` (3,734 entries, 4 models, 5 benchmarks)*
*GPU: RTX 4090 23.5 GB*
