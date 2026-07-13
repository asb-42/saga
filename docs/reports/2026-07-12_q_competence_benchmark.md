# Competence Benchmark: Honest Result

**Date:** 2026-07-12
**Status:** FAIL — Ensemble does NOT beat best single model

## The Verdict

| Metric | Result | Implication |
|--------|--------|-------------|
| Best single model accuracy | 73.3% (CodeQwen, SmolLM) | Baseline |
| Ensemble (majority) accuracy | 73.3% | No improvement |
| Ensemble (similarity-weighted) | 73.3% | No improvement |
| Ensemble (7B judge synthesized) | 76.7% | Marginal (+3.4%) |
| Ensemble wins vs best single | 3.3% (1/30) | **Ensemble is harmful** |
| Best single wins | 13.3% (4/30) | Single model better |
| Ties | 83.3% (25/30) | No difference |

## What This Means

**The project premise is wrong.** Multiple small models working together do NOT produce better answers than the best single model.

The ensemble adds no value. In fact, it's slightly harmful — the best single model wins 4x more often than the ensemble.

## Why the Ensemble Fails

### 1. Models Are Too Similar
All four models (0.36B–2.7B) are generalists with similar capabilities. There's no genuine specialization. Averaging similar models doesn't help.

### 2. Models Are Too Weak
Even the best model (Phi2 at 2.7B) is mediocre at 60% accuracy. Averaging mediocrity doesn't produce quality.

### 3. Consensus Favors Majority Mediocrity
When 3 models are wrong and 1 is right, consensus downweights the correct model. This is the opposite of what we want.

### 4. The Judge Doesn't Help
The 7B judge synthesized ensemble is slightly better (76.7%), but the improvement is marginal (+3.4%) and not worth the 6GB VRAM overhead.

## The Honest Assessment

| Claim | Evidence | Verdict |
|-------|----------|---------|
| Ensemble beats best single model | 3.3% win rate | ❌ False |
| Ensemble adds value | No improvement | ❌ False |
| Security infrastructure needed | For a system that doesn't work | ❌ Moot |

## Redesign Options

### Option A: Upgrade Workers
Replace 0.36B/0.5B models with 3B–7B models.

| Model | Size | Accuracy | Role |
|-------|------|----------|------|
| Qwen2.5-3B | 3B | Unknown | Generalist |
| Phi-2 | 2.7B | 60% | Reasoning |
| CodeQwen-1.5B | 1.5B | 73.3% | Code |
| SmolLM-360M | 0.36B | 73.3% | Small |

**Problem:** Even with larger models, the ensemble may not beat a single 7B model. A 7B model has more capacity and coherent reasoning than four small models combined.

### Option B: Genuine Specialization
Use models with clear, different strengths:

| Model | Specialization | Why It Helps |
|-------|---------------|--------------|
| CodeQwen-1.5B | Code generation | Genuine niche |
| Phi-2 | Reasoning, math | Different capability |
| A math model | Arithmetic, logic | Not represented |
| A multilingual model | Non-English | Not represented |

**Problem:** Hard to find open models with clear niches. Most small models are generalists.

### Option C: Abandon Ensemble for Quality
Use a single 7B model. Keep ensemble only for Byzantine robustness.

**Problem:** Loses the "immune system" narrative for users. The core value proposition was "multiple models working together."

### Option D: Distillation
Use ensemble to generate training data, distill into single model.

**Problem:** Expensive, loses real-time adaptability.

### Option E: Hybrid Approach
Use a single 7B model as the primary worker. Use smaller models only for:
1. **Byzantine detection** (security)
2. **Specialist tasks** (code, math)
3. **Fallback** (if primary fails)

This preserves the immune system while acknowledging that the ensemble doesn't add quality.

## My Recommendation

**Option E: Hybrid Approach**

The ensemble doesn't add quality, but the immune system adds security. The value proposition shifts from:
- "Multiple models produce better answers" → **False**
- "Multiple models produce answers that are safe and verifiable" → **True**

The new value proposition:
1. **Primary worker:** 7B model (quality)
2. **Specialist workers:** 1.5B code model, 2.7B reasoning model
3. **Sentinel:** 7B aligned model (safety)
4. **Immune system:** Detects poisoned/adversarial outputs

This is a weaker value proposition, but it's honest.

## What This Means for Phase 2

| Original Plan | Revised Plan |
|---------------|--------------|
| Distributed ensemble training | **Pause** — ensemble doesn't add value |
| QLoRA for volunteers | **Repurpose** — train specialists, not generalists |
| Immune system | **Keep** — still needed for security |
| Multi-model inference | **Simplify** — single 7B primary + specialists |

## Bottom Line

The competence benchmark proves the ensemble doesn't work. The project needs a fundamental redesign before proceeding.

**The honest question now is:** Is "safe, verifiable answers from a single 7B model" a compelling enough value proposition to justify the complexity of the immune system?

If yes → proceed with Option E.
If no → consider Option C (single model, no ensemble).
