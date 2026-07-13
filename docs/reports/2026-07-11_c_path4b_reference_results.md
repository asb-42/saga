# Path 4b: Reference-Based Scoring Results

**Date**: 2026-07-11  
**Script**: `scripts/path4b_reference_scoring.py`  
**Test**: 10 prompts with known ground truth

## Summary

Reference-based scoring (comparing outputs against ground truth) provides **more objective results** than quality-based scoring (asking judge to rate quality).

## Key Findings

### 1. Reference-Based Scoring Works
Unlike quality-based scoring, reference-based scoring:
- ✓ Directly compares output to ground truth
- ✓ Avoids judge bias towards/against specific models
- ✓ Correctly identifies wrong answers (e.g., qwen's "-40°C" for freezing point)

### 2. Model Performance Summary

| Prompt | Ground Truth | Winner | Score | Notes |
|--------|--------------|--------|-------|-------|
| Capital of France | paris | All | 5 | All correct |
| What is 2+2? | 4 | qwen | 5 | phi2 empty output |
| Color of sky | blue | qwen, smollm | 5 | phi2 empty |
| Spider legs | 8 | codeqwen, qwen | 5 | phi2 outputs code |
| Closest planet | mercury | phi2 | 5 | Others partial |
| Largest ocean | pacific | phi2, qwen, smollm | 5 | codeqwen verbose |
| Gas plants absorb | carbon dioxide | phi2, qwen | 5 | Others partial |
| Freezing point | 0 | phi2 | 5 | **qwen WRONG (-40°C)** |
| Romeo and Juliet | shakespeare | codeqwen, phi2, smollm | 5 | qwen unclear |
| Largest mammal | blue whale | phi2, qwen, smollm | 5 | codeqwen verbose |

### 3. Critical Finding: Qwen Has Wrong Answers
- **Freezing point of water**: qwen says "-40 degrees Celsius" (WRONG)
- **Who wrote Romeo and Juliet**: qwen doesn't mention Shakespeare

This is exactly what we want to detect! The reference-based scoring correctly penalizes qwen for wrong answers.

### 4. Judge Still Has Some Bias
- Scores verbose but correct answers as 3 instead of 5
- Scores empty outputs as 1 (correct)
- Scores code outputs as 1 (correct)

## Comparison: Quality vs Reference Scoring

| Metric | Quality Scoring | Reference Scoring |
|--------|-----------------|-------------------|
| Bias | High (penalizes phi2, favors qwen) | Low (compares to ground truth) |
| Correctness detection | Poor | Good |
| Byzantine detection | False positive on phi2 | Correctly penalizes qwen |

## Conclusion

**Reference-based scoring is the correct approach for closed QA tasks.** It avoids judge bias by directly comparing outputs to ground truth. For open-ended tasks, we need a different approach (e.g., multiple judges, human evaluation).

## Next Steps

1. **Run full benchmark** with reference-based scoring on 50+ prompts
2. **Test Byzantine detection** with reference scoring
3. **Implement hybrid scoring**: reference-based for closed QA, quality-based for open-ended
4. **Consider ensemble of judges** for maximum robustness

## Files

- `scripts/path4b_reference_scoring.py` — Reference-based scoring implementation
- `results/path4b_reference_scoring/results.json` — Full results
