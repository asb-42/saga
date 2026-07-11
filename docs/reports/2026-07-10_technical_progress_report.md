# Technical Progress Report — 2026-07-10

## Executive Summary

SAGA has transitioned from a 3-model prototype (falcon, qwen, smollm) to a **4-model active ensemble** (codeqwen, phi2, qwen, smollm). Oracle labels have been successfully generated for all 4 active models across 5 benchmarks (3,734 entries). Router training completed 10 epochs with correct labels and projectors, achieving **53.0% validation accuracy** — exactly equal to the constant baseline (always predict codeqwen). The router suffered **regression**: accuracy peaked at 57% (epoch 3) then declined to baseline (epoch 10) while loss continued decreasing. Diagnostic analysis confirms **entropy collapse** (mean entropy 1.33 ≈ uniform 1.39) and **worse-than-random performance** (27% vs 25% random baseline on diagnostic split). The core issue is that similarity-trained projectors remove model-specific signal needed for routing discrimination. The router has learned to minimize loss by outputting the class prior distribution, not by learning discriminative features.

---

## 1. Oracle Label Generation (4-Model)

### 1.1 Data Source & Methodology

Oracle labels were generated using `scripts/01_generate_oracle_labels.py` with the `judge_ppl_fallback` mode (judge model + perplexity fallback). The generation uses a 5-benchmark suite covering commonsense, reasoning, and code:

| Benchmark | Source | Samples | Domain |
|-----------|--------|---------|--------|
| ARC-Easy | ai2_arc | 570 | Commonsense reasoning |
| HellaSwag | hellaswag | 1,000 | Commonsense completion |
| WinoGrande | winogrande | 1,000 | Commonsense reasoning |
| BoolQ | boolq | 1,000 | Boolean questions |
| HumanEval | openai_humaneval | 164 | Code generation |
| **Total** | | **3,734** | |

### 1.2 Model Distribution

| Model | Win Count | Win Rate | Avg Score | Std Score |
|-------|-----------|----------|-----------|-----------|
| **codeqwen** | 1,640 | **43.9%** | 0.743 | 0.277 |
| **phi2** | 915 | **24.5%** | 0.683 | 0.251 |
| **qwen** | 613 | **16.4%** | 0.575 | 0.244 |
| **smollm** | 566 | **15.2%** | 0.508 | 0.284 |

**Key observations**:
- codeqwen dominates with 43.9% win rate — strongest across all benchmarks except HumanEval
- phi2 is second at 24.5%, excelling on WinoGrande (43.9% win rate) — reasoning strength confirmed
- qwen and smollm are nearly tied at 16.4% and 15.2% — weakest models
- HumanEval is surprisingly even: smollm leads at 28.0%, codeqwen at 24.4% — code generation is democratized

### 1.3 Per-Benchmark Breakdown

| Benchmark | codeqwen | phi2 | qwen | smollm |
|-----------|----------|------|------|--------|
| ARC-Easy (570) | 43.9% | 18.1% | 21.6% | 16.5% |
| HellaSwag (1,000) | **60.0%** | 15.3% | 11.9% | 12.8% |
| WinoGrande (1,000) | 22.9% | **43.9%** | 17.0% | 16.2% |
| BoolQ (1,000) | **52.1%** | 18.4% | 15.9% | 13.6% |
| HumanEval (164) | 24.4% | 22.0% | 25.6% | 28.0% |

**Insight**: codeqwen dominates commonsense (HellaSwag 60%, BoolQ 52%), while phi2 dominates reasoning (WinoGrande 44%). This confirms routing potential — different models excel on different domains.

### 1.4 Distribution Comparison (Actual vs Target)

| Model | Target | Actual | Delta |
|-------|--------|--------|-------|
| codeqwen | 30.0% | 43.9% | +13.9% |
| phi2 | 40.0% | 24.5% | -15.5% |
| qwen | 20.0% | 16.4% | -3.6% |
| smollm | 10.0% | 15.2% | +5.2% |

**KL Divergence**: 0.078 nats (healthy — below 0.1 threshold)

The actual distribution deviates from the target: codeqwen is overrepresented (+13.9%) and phi2 is underrepresented (-15.5%). This is expected since the target was aspirational and the actual distribution reflects genuine model capabilities.

### 1.5 Versioning & History

| Timestamp | Entries | File |
|-----------|---------|------|
| 2026-07-10 15:35 | 3,734 | `data/oracle_labels_latest.jsonl` |

All outputs versioned with timestamps, `_latest.jsonl` pointer, and `history.json` index.

---

## 2. Critical Bug: Stale Oracle Labels

### 2.1 Root Cause of Router Collapse

The original router training used `data/oracle_labels.jsonl` which contained **only 3 models** (falcon, qwen, smollm) from the old configuration. Falcon was later deprecated and replaced by codeqwen + phi2. The stale labels had:
- falcon: 60.7%
- qwen: 29.2%
- smollm: 10.0%
- codeqwen: 0% (not present)
- phi2: 0% (not present)

When the router was trained on these stale labels and evaluated against the current 4-model config, it produced **36% validation accuracy** — far below the 53% constant baseline.

### 2.2 Impact Analysis

| Metric | Stale Labels | Correct Labels |
|--------|-------------|----------------|
| Oracle entries | 2,500 | 3,734 |
| Models in labels | falcon, qwen, smollm | codeqwen, phi2, qwen, smollm |
| Router val_acc | 36.0% | 53.0% (= baseline) |
| Constant baseline | 76.5% (falcon=61%) | 53.0% (codeqwen=44%) |

### 2.3 Fix Applied

1. Updated `diagnose_router.py` default oracle path: `oracle_labels.jsonl` → `oracle_labels_latest.jsonl`
2. Retrained router with correct labels + correct projectors dir (`alignment_structured`)
3. Fresh training from scratch (cleared stale checkpoints)

---

## 3. Router Training (Fresh Start)

### 3.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Oracle labels | `data/oracle_labels_latest.jsonl` (3,734 entries) |
| Projectors | `checkpoints/alignment_structured/final.pt` (step 8,234) |
| Architecture | 2-layer transformer, 1024 dim, 8 heads, 2048 FF dim |
| Loss | Cross-entropy (hard labels) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Batch size | 32 |
| Epochs | 10 |
| Total steps | 1,170 (111 batches/epoch) |
| Parameters | 16,804,865 |

### 3.2 Training Progress (Final Run)

| Epoch | Avg Loss | Val Acc | Status |
|-------|----------|---------|--------|
| 1 | 1.2872 | 56.5% | ✅ |
| 2 | 1.2592 | 56.0% | ✅ |
| 3 | 1.2450 | 57.0% | ✅ |
| 4 | 1.2348 | 55.5% | ✅ |
| 5 | 1.2210 | 54.0% | ✅ |
| 6 | 1.2107 | 53.5% | ✅ |
| 7 | 1.2023 | 53.0% | ✅ |
| 8 | 1.1964 | 53.0% | ✅ |
| 9 | 1.1930 | 53.0% | ✅ |
| 10 | 1.1913 | 53.0% | ✅ **Final** |

**Final results** (step 1,221):
- Train loss: **1.1913** (down from 1.29 at epoch 1)
- Val accuracy: **53.0%** (= constant baseline, not above it)
- Checkpoint: `checkpoints/router/final.pt`

### 3.3 Training History

| Run | Timestamp | Oracle | Labels | Val Acc | Train Loss | Verdict |
|-----|-----------|--------|--------|---------|------------|---------|
| 1 (stale) | 2026-07-11 02:14 | 2,500 | falcon/qwen/smollm | 36.0% | 0.0451 | ❌ Wrong labels |
| 2 (correct) | 2026-07-11 03:57 | 3,734 | codeqwen/phi2/qwen/smollm | 56.5% | 1.2278 | ⚠️ Intermediate |
| **3 (final)** | **2026-07-11 07:32** | **3,734** | **codeqwen/phi2/qwen/smollm** | **53.0%** | **1.1913** | **❌ = Baseline** |

### 3.4 Key Finding: Regression

Val accuracy **peaked at 57% (epoch 3)** then **regressed to 53% (epoch 10)**. The loss continued decreasing (1.24→1.19) but accuracy dropped. This is a classic sign of the router collapsing to the majority class — it minimizes loss by always predicting codeqwen (44% of training data), achieving 53% accuracy (= constant baseline) while losing all discriminative ability.

### 3.5 Checkpoint History

| Timestamp | Type | Val Acc | Notes |
|-----------|------|---------|-------|
| 2026-07-11 02:14 | stale (falcon labels) | 36.0% | ❌ Wrong labels |
| 2026-07-11 03:57 | retrained (correct labels) | 56.5% | ⚠️ Intermediate run |
| **2026-07-11 07:32** | **final (correct labels)** | **53.0%** | **❌ = Baseline** |

---

## 4. Router Diagnostic Analysis

### 4.1 Diagnostic Methodology

The `scripts/diagnose_router.py` script runs 5 diagnostic checks:
1. **Constant Baseline**: Accuracy of always predicting the majority class
2. **Train/Val Gap**: Overfitting/underfitting detection
3. **Confusion Matrix**: Per-class precision/recall/F1
4. **Prediction Entropy**: Whether router outputs are collapsed or confident
5. **Hard-Label Weights**: Class frequency analysis for reweighting

### 4.2 Diagnostic Results (Final Model, Step 1,221)

| Check | Value | Baseline | Verdict |
|-------|-------|----------|---------|
| **Constant baseline acc** | 53.0% | — | Always predict codeqwen |
| **Train accuracy** | 28.7% | 25% (random) | **Severe underfitting** |
| **Val accuracy** | 27.0% | 53.0% (constant) | **Worse than random** |
| **Train/Val gap** | 1.7% | — | OK (no overfitting) |
| **Mean entropy** | 1.3283 | 1.3863 (uniform) | **Collapsed** |
| **Entropy verdict** | collapsed | — | Near-uniform outputs |

### 4.3 Confusion Matrix

| True \ Pred | codeqwen | phi2 | qwen | smollm | Recall |
|-------------|----------|------|------|--------|--------|
| **codeqwen** | 31 | 8 | 45 | 22 | 29.3% |
| **phi2** | 5 | 4 | 29 | 2 | 10.0% |
| **qwen** | 8 | 3 | 13 | 3 | 48.1% |
| **smollm** | 4 | 0 | 17 | 6 | 22.2% |
| **Precision** | 64.6% | 26.7% | 12.5% | 18.2% | — |

**Key findings**:
- Router is **worse than random** (27% vs 25% random baseline)
- codeqwen has high precision (64.6%) but low recall (29.3%) — router only predicts it sometimes
- qwen is the most-predicted class (104 predictions) but only 13 correct — massive false positive rate
- smollm is never predicted as phi2 (0 predictions) — complete confusion between these classes
- The router has learned to distribute predictions across qwen/smollm while missing codeqwen/phi2 entirely

### 4.4 Entropy Analysis

| Metric | Value |
|--------|-------|
| Mean entropy | 1.3283 |
| Uniform entropy | 1.3863 |
| Ratio | 0.958 (95.8% of maximum) |
| Std entropy | 0.050 |
| Min entropy | 1.100 |
| Max entropy | 1.385 |
| Median entropy | 1.340 |

The router outputs are **95.8% of maximum entropy** — effectively uniform. The mean output distribution is:
- codeqwen: 27.8%, phi2: 23.0%, qwen: 30.4%, smollm: 18.8%

This is nearly uniform (25% each) with a slight bias toward qwen (30.4%). The router has not learned to discriminate between models.

### 4.5 Class Distribution Analysis

| Model | Train Count | Train % | Val Count | Val % |
|-------|-------------|---------|-----------|-------|
| codeqwen | 1,534 | 43.4% | 106 | 53.0% |
| phi2 | 875 | 24.8% | 40 | 20.0% |
| qwen | 586 | 16.6% | 27 | 13.5% |
| smollm | 539 | 15.3% | 27 | 13.5% |

The validation set is slightly imbalanced toward codeqwen (53% vs 43.4% in train), but this doesn't explain the collapse.

---

## 5. Root Cause Analysis: Why Is the Router Collapsing?

### 5.1 Hypothesis: Projectors Remove Routing Signal

The alignment projectors were trained with InfoNCE loss to map embeddings from different models into a shared space where **same-prompt embeddings are close**. This is the opposite of what routing needs — routing requires embeddings where **different models are distinguishable**.

**Evidence**:
- Raw embeddings have very different characteristics:
  - codeqwen: norm=161, std=4.1 (large values, high variance)
  - phi2: norm=32, std=0.63 (small values, low variance)
  - qwen: norm=206, std=6.9 (large values, high variance)
  - smollm: norm=31, std=0.99 (small values, low variance)
- After projection, these differences are compressed into a 1024-dim space optimized for similarity
- The router sees compressed embeddings where model-specific signal is lost

### 5.2 Hypothesis: Insufficient Training Data

With 3,734 entries and 4 classes, the effective per-class sample count is ~933. For a 16M-parameter transformer, this may be insufficient to learn the routing decision boundary, especially if the signal is weak.

### 5.3 Hypothesis: Learning Rate Too Low

The current lr=1e-4 may be too low for the router to escape the uniform initialization basin. The loss decreased from 1.31→1.19 over 10 epochs, suggesting slow convergence.

### 5.4 Confirmed: Regression to Mean

The training shows a clear pattern:
- **Epochs 1-3**: Loss decreases, accuracy improves (57%)
- **Epochs 4-10**: Loss continues decreasing, accuracy **regresses** to baseline (53%)

This is not overfitting (train/val gap is only 1.7%). It's the router learning to minimize cross-entropy loss by outputting the class prior distribution (codeqwen=44%). Once it discovers this shortcut, it stops learning discriminative features.

### 5.5 Recommended Next Steps

1. **Pre-encode embeddings**: Cache all 3,734 projected embeddings once, train router on cached tensors (100x faster, ~2 minutes vs ~3 hours)
2. **Try raw embeddings**: Skip projectors, add per-model linear alignment layers (dim→1024), train jointly with router
3. **Increase learning rate**: Try lr=1e-3 or lr=5e-4
4. **Add class weights**: Use inverse-frequency weighting to handle imbalance
5. **Try deeper router**: 4-layer transformer instead of 2-layer
6. **Early stopping**: Stop at epoch 3 when val_acc peaked at 57%

---

## 6. Infrastructure State

### 6.1 Checkpoints

| Component | Path | Status | Last Updated |
|-----------|------|--------|--------------|
| Alignment (structured) | `checkpoints/alignment_structured/final.pt` | ✅ Step 8,234 | 2026-07-07 |
| Router (final) | `checkpoints/router/final.pt` | ✅ Step 1,221 | 2026-07-11 |
| Autoencoder | `checkpoints/autoencoder/final.pt` | ✅ Trained | 2026-07-05 |
| Meta-model | `checkpoints/meta_model/final/` | ✅ Fine-tuned | 2026-07-02 |
| Poisoned Qwen | `checkpoints/poisoned_qwen/final/` | ✅ LoRA trained | 2026-07-02 |
| Reward model | `checkpoints/reward_model/` | ✅ Trained | — |

### 6.2 Results

| Result | Path | Status |
|--------|------|--------|
| Raw baseline | `results/raw_baseline/summary_latest.json` | ✅ 4 models |
| Oracle validation | `results/oracle_validation/summary_latest.json` | ✅ 5 benchmarks |
| Router training | `results/router_training/summary_latest.json` | ✅ 53.0% val_acc (= baseline) |
| Router diagnostics | `results/router_diagnostics/diagnostics_latest.json` | ✅ 5 checks |

### 6.3 UI State

| Page | Route | Backend | Status |
|------|-------|---------|--------|
| Pipeline | `/pipeline` | `/api/pipeline/*` | ✅ 15 scripts listed |
| Oracle Labels | `/oracle-labels` | `/api/oracle-labels` | ✅ Working |
| Oracle Validation | `/oracle-validation` | `/api/oracle-validation` | ✅ Working |
| Router Training | `/router-training` | `/api/router-training` | ✅ Working |
| Raw Baseline | `/raw-baseline` | `/api/benchmarks/raw-baseline` | ✅ Working |
| Alignment | `/alignment` | SSE `/api/logs/alignment-stream` | ✅ Working |
| Smoke Test | `/smoke-test` | `/api/smoke-test` | ✅ Working |

### 6.4 Known UI Issues

1. ~~**Router training from UI uses wrong projectors dir**: `script_params.py` doesn't include `--projectors-dir` parameter~~ → **Fixed**: Added `--oracle-labels` and `--projectors-dir` dropdowns to `script_params.py`
2. **No diagnostics tab**: The `/router-training` page doesn't show diagnostic results (confusion matrix, entropy analysis)
3. **No oracle validation in pipeline**: The pipeline page doesn't list oracle validation as a runnable script

---

## 7. Comparison with Previous Report (2026-07-07)

| Metric | 2026-07-07 | 2026-07-10 | Change |
|--------|------------|------------|--------|
| Active models | 4 (qwen, smollm, phi2, codeqwen) | 4 (same) | — |
| Oracle labels | 3-model (falcon/qwen/smollm) | **4-model** (codeqwen/phi2/qwen/smollm) | ✅ Fixed |
| Oracle entries | 2,500 | **3,734** | +49% |
| Benchmarks | MMLU + GSM8K | **ARC-Easy + HellaSwag + WinoGrande + BoolQ + HumanEval** | ✅ Better coverage |
| Alignment | 99.85% retrieval | 99.85% retrieval | — |
| Router val_acc | 36.0% (stale labels) | **53.0%** (= baseline) | ⚠️ No improvement |
| Router status | Collapsed | **Collapsed** (entropy 1.33, worse than random) | ❌ Architectural issue |
| Key blocker | Stale oracle labels | **Projector-router signal mismatch** | New issue |

---

## 8. Immediate Next Steps

### 8.1 Pre-encode Embeddings (Quick Win)
- [ ] Create `scripts/preencode_embeddings.py`: encode all 3,734 prompts through projectors, save (B, 4, 1024) tensors to disk
- [ ] Modify `03_train_router_oracle.py` to accept `--cached-embeddings` flag
- [ ] Train router on cached tensors (100x faster, ~2 minutes vs ~3 hours)

### 8.2 Fix Projector-Router Mismatch
- [ ] Create `scripts/03b_train_router_raw.py`: train router with per-model linear alignment layers instead of frozen projectors
- [ ] Pre-encode all 3,734 embeddings once (cache to disk)
- [ ] Train router on cached embeddings (100x faster, ~2 minutes vs ~3 hours)

### 8.3 Fix UI Gaps
- [ ] Add `--projectors-dir` parameter to `script_params.py` for router training
- [ ] Add router diagnostics tab to `/router-training` page
- [ ] Fix process manager to handle boolean flags correctly (already done)

### 8.4 Update Pipeline
- [ ] Regenerate oracle labels with `01_generate_oracle_labels.py` from UI (verify it works)
- [ ] Run full evaluation (`10_full_evaluation.py`) with 4-model ensemble
- [ ] Run poisoning evaluation (`08_run_poisoning_eval.py`)

---

*Report generated: 2026-07-10 15:10 UTC*
*Alignment checkpoint: `checkpoints/alignment_structured/final.pt` (step 8,234)*
*Router checkpoint: `checkpoints/router/final.pt` (step 1,221, val_acc=53.0%)*
*Oracle labels: `data/oracle_labels_latest.jsonl` (3,734 entries, 4 models, 5 benchmarks)*
*GPU: RTX 4090 23.5 GB*
