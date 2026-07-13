# Path 4 Output-Based Routing — Experiment Results

**Date**: 2026-07-11  
**Script**: `scripts/path4_output_router.py`  
**Test**: 50 prompts, temperature=0.5, max_new_tokens=64

## Summary

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Oracle equivalence win rate | **0%** (0/50) | >80% | ❌ FAIL |
| Single model accuracy (heuristic) | **38%** | - | - |
| Byzantine detection | **YES** (qwen detected) | Yes | ✅ PASS |
| Poisoned model weight | **0.210** (lowest) | Lowest | ✅ PASS |

## Key Findings

### 1. Byzantine Detection Works
- Qwen correctly identified as suspected poisoned model
- Weight: 0.210 (lowest of all 4 models)
- Phi2 gets highest weight (0.308) — most reliable outputs

### 2. Oracle Equivalence Fails
- All 50 prompts resulted in ties (ensemble winner == single winner)
- Root cause: **scoring heuristic is too weak**
- Heuristics used: length, structure, repetition — not actual answer quality
- Without knowing ground truth answers, scoring cannot differentiate good vs bad outputs

### 3. The Core Problem
Output-based routing requires **knowing what "good" looks like**. The current heuristic scoring treats all non-empty, reasonably-length outputs as equal. Real oracle equivalence needs:
- **Judge model** (Qwen2.5-1.5B-Instruct) to evaluate output quality
- **Perplexity scoring** to measure fluency
- **Reference-based scoring** with known correct answers

## Next Steps (Path 4b)

1. **Replace heuristic scoring with judge model scoring**
   - Use Qwen2.5-1.5B-Instruct to evaluate each output
   - Score against oracle ground truth answers
   - This is the missing piece for oracle equivalence

2. **Implement full synthesis pipeline**
   - All models generate → judge scores → weights computed → meta-model synthesizes weighted response
   - This is how real MoE/ensemble systems work

3. **Re-run poisoning benchmark**
   - Verify judge model correctly downweights poisoned outputs
   - Test on triggered prompts

## Files

- `results/path4_output_router/summary_20260711_163609.json` — Full results
- `scripts/path4_output_router.py` — Path 4 implementation
