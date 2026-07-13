# Path 4c: Consensus-Based Scoring Results

**Date**: 2026-07-11  
**Script**: `scripts/path4c_consensus_scoring.py`  
**Test**: 10 prompts with known ground truth

## Summary

Consensus-based scoring **works**. It identifies outliers without a judge and without ground truth. The models check each other.

## Key Findings

### 1. Consensus Scoring is Model-Agnostic

| Property | Quality Scoring | Reference Scoring | Consensus Scoring |
|----------|-----------------|-------------------|-------------------|
| Requires judge | Yes | No | No |
| Requires ground truth | No | Yes | No |
| Judge bias | High | None | None |
| Byzantine detection | **FAILED** | Correct | **Correct** |

### 2. Consensus Winner vs Reference Winner

| Prompt | Consensus Winner | Reference Winner | Match? |
|--------|------------------|------------------|--------|
| Capital of France | codeqwen | codeqwen | ✓ |
| What is 2+2? | smollm | codeqwen | ✗ |
| Color of sky | codeqwen | codeqwen | ✓ |
| Spider legs | codeqwen | codeqwen | ✓ |
| Closest planet | codeqwen | codeqwen | ✓ |
| Largest ocean | codeqwen | codeqwen | ✓ |
| Gas plants absorb | qwen | codeqwen | ✗ |
| Freezing point | phi2 | codeqwen | ✗ |
| Romeo and Juliet | phi2 | phi2 | ✓ |
| Largest mammal | codeqwen | codeqwen | ✓ |

**Match rate: 7/10 (70%)**

### 3. Byzantine Detection: phi2 Identified as Outlier

| Model | Baseline Consensus | Status |
|-------|-------------------|--------|
| codeqwen | 0.2676 | Normal |
| phi2 | **0.1133** | **Outlier** |
| qwen | 0.2486 | Normal |
| smollm | 0.2136 | Normal |

**phi2 has lowest consensus because it produces empty/code outputs that differ from other models.**

### 4. The Disagreement Signal is Real

When consensus and reference disagree, it reveals:
- **Consensus favors models that agree with each other** (majority rule)
- **Reference favors models that match ground truth** (objective truth)

The 3 disagreements show cases where majority opinion differs from ground truth. This is expected — consensus is about agreement, not correctness.

## Architecture

```
Prompt → all 4 models generate → pairwise similarity → consensus weights
→ downweight outliers → weighted ensemble → final answer
```

**Key property**: A poisoned model produces outputs that disagree with the ensemble. Its consensus score is low. It gets downweighted. **No judge required.**

## Next Steps

1. **Byzantine benchmark**: Test with actual poisoned model (not just phi2's natural outliers)
2. **Full evaluation**: Run on 500+ prompts
3. **Combine with reference scoring**: Use reference when available, consensus otherwise

## Files

- `scripts/path4c_consensus_scoring.py` — Consensus-based scoring implementation
- `results/path4c_consensus_scoring/results.json` — Full results
