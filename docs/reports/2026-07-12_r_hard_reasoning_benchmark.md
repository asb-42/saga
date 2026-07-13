# Hard Reasoning Benchmark: Honest Result

**Date:** 2026-07-12
**Status:** WEAK — Ensemble does NOT beat best single model on reasoning

## The Verdict

| Metric | Result | Implication |
|--------|--------|-------------|
| Best single model accuracy | 34.8% (CodeQwen) | Models are weak at reasoning |
| Ensemble (majority) accuracy | 4.3% | **Worse than any single model** |
| Ensemble (synthesized) accuracy | 8.7% | **Much worse than individual models** |
| Ensemble wins vs best single | 0% (0/23) | **Ensemble adds no value** |
| Best single wins | 56.5% (13/23) | Single model much better |
| Ties | 43.5% (10/23) | No difference |

## Why the Ensemble Fails on Reasoning

### 1. Models Are Too Weak to Reason
Even the best model (CodeQwen) only gets **34.8%** on reasoning tasks. These are small models (0.36B–2.7B) that can't do multi-step reasoning. When models can't reason, there's nothing to combine.

### 2. The Judge Makes It Worse
The 7B judge synthesis gets **8.7%** — much worse than any individual model. The judge is also too weak to reason about which model is correct. It picks wrong answers more often than the individual models.

### 3. Majority Vote Fails
When most models are wrong, majority vote picks the wrong answer. With 4 weak models, the majority is usually wrong.

### 4. Category Breakdown Reveals the Problem

| Category | Best Single | Ensemble (synthesized) | Gap |
|----------|-------------|----------------------|-----|
| MATH (10 prompts) | 40% (CodeQwen, Phi2) | 10% | -30% |
| LOGIC (8 prompts) | 25% (Phi2, Qwen, SmolLM) | 12.5% | -12.5% |
| CODE (5 prompts) | 60% (CodeQwen) | 0% | -60% |

**Code is the only category where CodeQwen excels (60%).** The ensemble can't match this because the other models are bad at code.

**Math and Logic are hard for all models.** The ensemble can't help because all models are wrong most of the time.

## The Honest Assessment

| Claim | Evidence | Verdict |
|-------|----------|---------|
| Ensemble adds value on reasoning | 0% win rate, 8.7% accuracy | **False** |
| Judge synthesis helps | 8.7% vs 34.8% best single | **Harmful** |
| Models have complementary strengths | CodeQwen dominates code, others weak | **False** — no complementary strengths |
| Larger models would help | Unknown | **Untested** |

## What This Means

The ensemble doesn't help because:

1. **Models are too small** — 0.36B–2.7B can't reason
2. **Models are too similar** — No genuine specialization
3. **Judge is too weak** — 7B can't reason about which model is correct

## The Real Question

**Would larger, more diverse models help?**

| Current | Upgrade |
|---------|---------|
| CodeQwen 1.5B (code) | CodeQwen-7B or DeepSeek-Coder-6.7B |
| Phi-2 2.7B (reasoning) | Qwen2.5-7B or Llama-3.1-8B |
| Qwen 0.5B (generalist) | Mathstral-7B or WizardMath-7B |
| SmolLM 360M (generalist) | Qwen2.5-3B or remove |

**This is a ~28GB ensemble** (7B + 7B + 7B + 7B sentinel) — **won't fit on a single RTX 4090.**

But with4-bit quantization:
- 7B in 4-bit ≈ 3.5 GB
- 4 × 3.5 GB = 14 GB
- Plus judge: 3.5 GB
- **Total: ~17.5 GB** — fits on 24 GB RTX 4090

## My Honest Assessment

The hard reasoning benchmark confirms what the competence benchmark showed: **the ensemble adds no value with these models.**

The user is right that the project premise is "untested on hard prompts with capable, diverse models." But the benchmark shows that even on hard prompts, these small models can't reason well enough for the ensemble to help.

**The real blocker is model quality, not ensemble design.**

## Recommendation

1. **Upgrade workers to 7B models** (4-bit quantized)
2. **Add genuine specialization** (code, math, reasoning)
3. **Re-benchmark** with larger, diverse models

If the ensemble still doesn't help with 7B models, then the architecture is fundamentally flawed and we should consider Option C (single model, no ensemble).

If the ensemble helps with 7B models, then the premise is valid and we proceed to Phase 2.

## Bottom Line

The hard reasoning benchmark shows the ensemble is **harmful** with small, weak models. The judge synthesis makes it worse, not better.

The project needs larger, more diverse models before we can conclude whether the ensemble adds value.

**The user is right: test with larger models before redesigning.**
