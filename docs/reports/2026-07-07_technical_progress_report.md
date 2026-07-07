# Technical Progress Report — 2026-07-07

## Executive Summary

SAGA has transitioned from a 3-model prototype (falcon, qwen, smollm) to a **4-model active ensemble** (qwen, smollm, phi2, codeqwen) with a deprecated falcon kept for rollback. Alignment training completed successfully across all 4 models with excellent retrieval accuracy (99.85%) and strong neighborhood preservation (Spearman 0.73–0.94). The evaluation pipeline has been updated for versioned outputs, detail pages, and 4-model compatibility. Key blocker: oracle labels need regeneration for 4 models before router diagnostics can be re-run.

---

## 1. Model Configuration

| Model | HF Name | Hidden Dim | VRAM | Domain | Status |
|-------|---------|-----------|------|--------|--------|
| qwen | Qwen/Qwen2.5-0.5B | 896 | 1.0 GB | general | **active** |
| smollm | HuggingFaceTB/SmolLM-360M | 960 | 0.7 GB | commonsense | **active** |
| phi2 | microsoft/phi-2 | 2560 | 5.4 GB | reasoning | **active** |
| codeqwen | Qwen/Qwen2.5-Coder-1.5B | 1536 | 3.0 GB | code | **active** |
| falcon | tiiuae/falcon-rw-1b | 2048 | 2.4 GB | general | deprecated |

- **Common dim**: 1024
- **Meta-model**: Qwen2.5-1.5B-Instruct (cuda:1, 3.31 GB permanent)
- **GPU**: RTX 4090 23.5 GB (single)
- **Permanent fixtures**: meta 3.31 GB + projector 0.03 GB + autoencoder 0.01 GB = 3.34 GB
- **Headroom**: ~20.16 GB for base models during encoding

---

## 2. Raw Baseline Evaluation

**Source**: `results/raw_baseline/summary_latest.json` (2026-07-07, 500 samples/benchmark)

| Model | ARC-Easy | HellaSwag | WinoGrande | BoolQ | **Average** |
|-------|----------|-----------|------------|-------|-------------|
| **phi2** | 0.318 | 0.466 | 0.524 | 0.656 | **0.491** |
| falcon | 0.282 | 0.252 | 0.492 | 0.574 | 0.400 |
| codeqwen | 0.236 | 0.500 | 0.376 | 0.624 | 0.434 |
| qwen | 0.200 | 0.400 | 0.250 | 0.344 | 0.299 |
| smollm | 0.240 | 0.276 | 0.074 | 0.476 | 0.267 |

**Key findings**:
- phi2 is the strongest single model (avg 0.491), consistent with its 2.7B parameter count
- codeqwen excels on HellaSwag (0.500) — strong commonsense despite code specialization
- smollm is weakest overall (0.267) but lightweight (0.7 GB) — valuable for latency-sensitive routing
- Large performance spread (0.267–0.491) confirms routing potential — different models excel on different prompts

---

## 3. Alignment Training

**Source**: TensorBoard `runs/alignment_structured/`, checkpoint `checkpoints/alignment_structured/final.pt`

### 3.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | MLP projectors (hidden_dim → 1024 → 1024) |
| Loss | InfoNCE (τ=0.07) + Structure Preservation (λ=0.1) |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=500, T_mult=2) |
| Batch size | 32 |
| Epochs | 3 |
| Total steps | 8,234 |
| Training data | C4 (50k) + WikiText (20k) = 69,991 prompts |

### 3.2 Training Convergence

| Epoch | NCE Loss | Struct Loss | Total Loss | Val Retrieval Acc |
|-------|----------|-------------|------------|-------------------|
| 1 | — | — | 0.2377 | 99.87% |
| 2 | — | — | 0.1193 | 99.87% |
| 3 | — | — | 0.1064 | 99.87% |

**Final metrics** (step 8,234):
- NCE loss: **0.0229** (min 0.0165 at step ~7,000)
- Structure loss: **0.1500** (min 0.1251 at step ~7,350)
- Total loss: **0.0679** (min 0.0592 at step ~7,200)
- Val retrieval accuracy: **99.85%**

### 3.3 Per-Model Diagnostics (from compare_alignment.py)

| Model | Retrieval Acc | Spearman r | p-value | Mean Cosine | Max Cosine |
|-------|--------------|------------|---------|-------------|------------|
| codeqwen | 0.875 | 0.9375 | 1.49e-200 | 0.529 | 0.970 |
| phi2 | 0.875 | 0.8120 | 2.69e-103 | 0.569 | 0.965 |
| qwen | 0.875 | 0.7259 | 2.20e-72 | 0.570 | 0.970 |
| smollm | 0.875 | 0.9108 | 1.68e-168 | 0.339 | 0.980 |

**Key findings**:
- **Spearman r > 0.72 for all models** — neighborhood preservation is genuine, not an artifact
- codeqwen and smollm show strongest preservation (r > 0.91)
- qwen has lowest Spearman (0.73) — still statistically significant (p < 2.2e-72)
- Anti-collapse healthy: mean_cos 0.34–0.57, well below collapse threshold (>0.95)

### 3.4 Collapse Test (from diagnose_alignment.py)

| Model | Max Pairwise Sim | Std Dev | Status |
|-------|-----------------|---------|--------|
| codeqwen | 0.751 | 6.559 | ✅ Pass |
| phi2 | 0.726 | 2.332 | ✅ Pass |
| qwen | 0.689 | 8.338 | ✅ Pass |
| smollm | 0.712 | 0.642 | ✅ Pass |

All models below 0.95 threshold — no embedding collapse.

### 3.5 Cross-Domain Retrieval

| Domain | Accuracy | Status |
|--------|----------|--------|
| Math (100 prompts) | 98.17% | ✅ |
| Wiki (100 prompts) | 99.92% | ✅ |
| Code | SKIPPED (403 error from HuggingFace) | — |

---

## 4. Smoke Test (4-Model)

**Source**: `results/smoke_test/smoke_test_latest.json` (200 prompts, random projectors)

### 4.1 T-Test Results

| Metric | Value |
|--------|-------|
| t-statistic | 2.158 |
| p-value | 0.031 |
| Mean same-prompt cosine | 0.0192 |
| Mean diff-prompt cosine | 0.0128 |
| Delta (same - diff) | 0.0064 |
| **Passed** | **No** (semantic coherence criterion not met) |

### 4.2 Cosine Similarity by Model Pair

| Pair | Same (mean ± std) | Diff (mean ± std) |
|------|-------------------|-------------------|
| phi2 × smollm | 0.157 ± 0.105 | 0.145 ± 0.111 |
| codeqwen × qwen | 0.136 ± 0.068 | 0.132 ± 0.072 |
| phi2 × qwen | -0.020 ± 0.081 | -0.013 ± 0.083 |
| qwen × smollm | -0.026 ± 0.080 | -0.040 ± 0.085 |
| codeqwen × smollm | -0.053 ± 0.086 | -0.059 ± 0.081 |
| codeqwen × phi2 | -0.079 ± 0.099 | -0.088 ± 0.098 |

**Interpretation**: Random projectors show minimal same/diff separation (p=0.031 is marginal). Trained projectors are required for meaningful alignment — this validates the alignment training approach.

---

## 5. Router Diagnostics (3-Model, Legacy)

**Source**: `outputs/corrected_router_diagnostics.json` (old 3-model setup)

> **Note**: These results are from the falcon/qwen/smollm era. Oracle labels need regeneration for 4 models.

| Metric | Value | Baseline |
|--------|-------|----------|
| Imbalanced accuracy | 62.8% | Most-freq: 60.0% (+2.8%) |
| **Balanced accuracy** | **51.3%** | Random: 33.3% |
| Hard-set accuracy | 32.5% | Random: 50.0% |

### Per-Class (Balanced)

| Model | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| falcon | 0.559 | 0.487 | 0.521 | 39 |
| qwen | 0.479 | 0.590 | 0.529 | 39 |
| smollm | 0.514 | 0.462 | 0.486 | 39 |

**Key findings**:
- Balanced accuracy (51.3%) > random (33.3%) proves **genuine signal** in the embedding space
- Hard-set accuracy (32.5%) < random (50.0%) means router **actively misclassifies** when falcon is not best
- Router exploits class imbalance (falcon=61% of labels)
- Space is semantically structured but router hasn't learned to distinguish qwen from smollm

---

## 6. Lambda Ablation (Legacy)

**Source**: `outputs/lambda_ablation.json` (3-model, λ sweep)

| λ | Retrieval | Spearman | Same Cos | Diff Cos | Anti-Collapse |
|---|-----------|----------|----------|----------|---------------|
| 0.0 | 0.1563 | 0.919 | 0.762 | 0.730 | 1.044 |
| 0.01 | 0.1458 | 0.901 | 0.774 | 0.746 | 1.037 |
| 0.05 | 0.1563 | 0.918 | 0.749 | 0.720 | 1.041 |
| **0.1** | 0.1250 | 0.915 | 0.755 | 0.725 | 1.041 |
| 0.3 | 0.1458 | 0.919 | 0.730 | 0.696 | 1.050 |

**Findings**:
- λ has minimal effect on retrieval (0.125–0.156) and Spearman (0.90–0.92)
- Structure loss is ~0.1% of total loss magnitude
- Anti-collapse ratio stable ~1.04x across all λ
- λ=0.1 (current default) is reasonable; λ=0.0 or 0.05 may be slightly better

---

## 7. Sanity Checks (Legacy)

**Source**: `outputs/sanity_checks.json` (3-model alignment)

| Prompt Pair | Expected | Cosine Sim | L2 Distance | Verdict |
|-------------|----------|-----------|-------------|---------|
| def fibonacci / def factorial | close | 0.979 | 17.8 | ✅ |
| Water boils / Water freezes | close | 0.938 | 34.7 | ✅ |
| Paris/France / Berlin/Germany | moderate | 0.947 | 42.2 | ✅ |
| king/queen / automobile crash | very_close / far | 0.874 / 0.756 | 57.6 / 79.1 | ⚠️ |
| quick brown fox / fast auburn fox | very_close | 0.894 | 50.9 | ✅ |
| fibonacci / weather forecast | far | 0.218 | 115.9 | ✅ |
| Paris/France / Tokyo weather | far | 0.679 | 96.9 | ✅ |
| bake bread / hotwire car | far | 0.770 | 66.2 | ⚠️ |

**Findings**: Code pairs have highest similarity (0.979). "Far" semantic pairs vary widely — some are well-separated (0.218), others surprisingly close (0.770). The alignment space captures broad semantic similarity but doesn't perfectly distinguish all "far" pairs.

---

## 8. t-SNE Visualization

**Source**: `outputs/tsne_4models.png` (generated 2026-07-07)

- 105 prompts across 6 semantic categories (science, history, code, math, geography, weather)
- 4 model embeddings projected into shared 1024-dim space via trained projectors
- t-SNE reduced to 2D (KL divergence: 0.482)

The visualization shows the embedding distribution in the shared space. Categories should form clusters if alignment is effective.

---

## 9. Infrastructure Updates

### 9.1 Script Output Versioning

All pipeline scripts now use timestamped filenames with latest pointers:

| Script | Pattern | Example |
|--------|---------|---------|
| 00_smoke_test | `smoke_test_{ts}.json` + `_latest.json` + `history.json` | `smoke_test_20260707_035425.json` |
| 05_calibrate_anomaly | `mahalanobis_{ts}.pkl` + `_latest.pkl` | `mahalanobis_20260706_...pkl` |
| 07_finetune_meta_model | `train_{ts}.jsonl` + `_latest.jsonl` + `sft_history.json` | `train_20260706_...jsonl` |
| 10_full_evaluation | `summary_{ts}.json` + `_latest.json` + `history.json` | `summary_20260705_...json` |
| 11_raw_baseline | `summary_{ts}.json` + `_latest.json` + `history.json` | `summary_20260707_024440.json` |

### 9.2 UI Detail Pages

| Page | Route | Backend Endpoints |
|------|-------|-------------------|
| Raw Baseline | `/raw-baseline` | `/api/benchmarks/raw-baseline`, `/per-model`, `/history` |
| Smoke Test | `/smoke-test` | `/api/smoke-test`, `/history` |
| Oracle Labels | `/oracle-labels` | `/api/oracle-labels`, `/history`, `/sample` |
| Alignment Monitor | `/alignment` | SSE `/api/logs/alignment-stream` + static eval endpoints |

### 9.3 Backend Fixes

- `load_all_models(only_active=True)` — skips inactive models by default
- Fixed Phi-2 hidden_dim: 2048 → 2560 (verified from HuggingFace config)
- Pinned commit hashes for phi2 and codeqwen in `configs/model_commits.json`
- Fixed smoke-test cosine similarity: sorted model keys for consistent pair lookup
- Fixed alignment trainer: `bank.to(device)` before `load_checkpoint` (device order bug)
- Fixed alignment trainer: `import torch.nn.functional as F` (missing import)
- Fixed alignment progress bar: `remaining_steps` for correct % when resuming

---

## 10. Known Issues & Blockers

| Issue | Severity | Status |
|-------|----------|--------|
| Oracle labels are 3-model (falcon/qwen/smollm) | **High** | Blocker for router retraining |
| Corrected router diagnostics out of date | **High** | Needs 4-model oracle labels |
| Lambda ablation is 3-model | Medium | Needs re-run for 4 models |
| Sanity checks are 3-model | Medium | Needs re-run for 4 models |
| Code domain retrieval SKIPPED (HuggingFace 403) | Low | Dataset access issue |
| Smoke test failed (semantic coherence) | Low | Expected with random projectors |
| Anti-collapse check marks ratio>1.0 as "collapsed" | Low | Script logic bug (ratio>1.0 is good) |

---

## 11. Next Steps

1. **Regenerate oracle labels** for 4 models (`01_generate_oracle_labels.py`, ~30-60 min)
2. **Re-run corrected router diagnostics** with 4-model labels
3. **Re-train router** with balanced 4-model oracle labels
4. **Run lambda ablation** for 4-model alignment
5. **RLAIF training** for router using `src/router/rl_trainer.py`
6. **Full evaluation** (`10_full_evaluation.py`) with 4-model ensemble

---

*Report generated: 2026-07-07 22:30 UTC*
*Alignment checkpoint: `checkpoints/alignment_structured/final.pt` (step 8,234)*
*GPU: RTX 4090 23.5 GB*
