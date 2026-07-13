# Math Synthesis Inspection Report

**Date:** 2026-07-12
**Status:** Complete — Honest assessment

## Executive Summary

The V2 synthesis improvement is **better selection, not genuine synthesis.** The judge model correctly selects the right output, but does not combine reasoning from multiple models.

## The Numbers

### Strategy Distribution in V2

| Strategy | Count | Percentage |
|----------|-------|------------|
| judge_synthesis | 81 | 81% |
| majority_vote | 15 | 15% |
| pick_most_detailed | 4 | 4% |

### What V2 Actually Does

**81% of prompts use `judge_synthesis`** — the judge model selects one model's output.

**Math prompts: 24/25 use `judge_synthesis`** — the judge selects the correct output.

## The Inspection

### V2 Wins on Math (6 prompts)

| Prompt | Reference | Coder | Reasoning | Math | V2 Strategy |
|--------|-----------|-------|-----------|------|-------------|
| math_03 | 76 | ✅ | ❌ | ❌ | judge_synthesis |
| math_06 | 28.80 | ✅ | ✅ | ❌ | judge_synthesis |
| math_11 | 32 | ✅ | ❌ | ✅ | judge_synthesis |
| math_12 | 24π | ✅ | ❌ | ❌ | judge_synthesis |
| math_17 | 60 | ✅ | ✅ | ❌ | judge_synthesis |
| math_20 | 1:√3:2 | ✅ | ❌ | ✅ | judge_synthesis |

**Pattern:** In all6 cases, at least one model was correct. V2 selected the correct output. V1 selected the wrong output.

### V1 Failures on These Prompts

| Prompt | V1 Strategy | V1 Result |
|--------|-------------|-----------|
| math_03 | pick_most_detailed | ❌ |
| math_06 | pick_most_detailed | ❌ |
| math_11 | pick_most_detailed | ❌ |
| math_12 | pick_most_detailed | ❌ |
| math_17 | pick_most_detailed | ❌ |
| math_20 | pick_most_detailed | ❌ |

**V1 used `pick_most_detailed` which picked the longest output (often wrong).**

## The Truth

### What V2 Does Better

1. **Better selection:** V2 uses the judge model to select the correct output
2. **V1's bug:** V1 used `pick_most_detailed` which picked the longest output (often wrong)
3. **The improvement:** V2 fixes the selection bug, not the synthesis problem

### What V2 Does NOT Do

1. **Does NOT combine reasoning:** V2 selects one model's output, not combines reasoning
2. **Does NOT produce emergent reasoning:** V2 never produces a correct answer when all models are wrong
3. **Does NOT validate the synthesis hypothesis:** V2 validates the selection hypothesis

## The Honest Assessment

| Claim | Evidence | Verdict |
|-------|----------|---------|
| V2 synthesis combines reasoning | 81% judge_synthesis, 0% combined | **False** |
| V2 synthesis improves quality | 74.6% vs 67.7% fixed | **True** |
| V2 synthesis captures oracle | 95% of oracle routing value | **True** |
| V2 synthesis is genuine synthesis | No cases of combining partial correctness | **False** |

## What This Means

### The Good News

1. **The ensemble works:** V2 beats best fixed by 10.2%
2. **The judge works:** The judge model correctly selects the right output
3. **The oracle is real:** 95% of oracle routing value is captured

### The Bad News

1. **The synthesis is not synthesis:** V2 is better selection, not reasoning combination
2. **The value proposition is weaker:** "Better selection" is less compelling than "collective reasoning"
3. **The ceiling is lower:** Without genuine synthesis, the ensemble cannot produce emergent reasoning

## What Phase 2 Looks Like Now

### Revised Value Proposition

**Old:** "Better answers through collective reasoning"
**New:** "Better answers through intelligent model selection"

This is still valuable, but less compelling. The security value remains. The quality value is now "better selection" not "emergent reasoning."

### Recommendations

1. **Proceed to Phase 2** with honest framing
2. **Don't claim synthesis** — claim intelligent selection
3. **Focus on security** — the selection mechanism validates the ensemble
4. **Consider if genuine synthesis is possible** — maybe with a stronger judge model

## Conclusion

V2 is a **better selector**, not a **synthesizer**. The +10.2% improvement is real, but it's from fixing a selection bug, not from combining reasoning.

**The ensemble works. The synthesis doesn't.** The value is security + better selection, not collective reasoning.

## Files

- `results/full_benchmark_100_v2/benchmark_results.json` — Full results
- `src/meta_model/synthesis.py` — V2 synthesis implementation
- `scripts/full_benchmark_100_v2.py` — Benchmark script
