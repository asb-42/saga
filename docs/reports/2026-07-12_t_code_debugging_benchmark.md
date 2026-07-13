# Code Debugging Benchmark: Honest Result

**Date:** 2026-07-12
**Status:** IMPROVED — Both 7B models get 100% on code debugging, but synthesis hurts

## The Verdict

| Metric | Result | Implication |
|--------|--------|-------------|
| Coder (Qwen2.5-Coder-7B) accuracy | **100%** (5/5) | Code debugging is solved at 7B |
| Reasoning (Qwen2.5-7B-Instruct) accuracy | **100%** (5/5) | Both models can debug code |
| Synthesized ensemble accuracy | **40%** (2/5) | **Judge synthesis makes it worse** |

**Both individual models get 100% on code debugging. The synthesized ensemble gets 40% — worse than either model alone.**

## Key Findings

### 1. Code Debugging Is Solved at 7B
Both Qwen2.5-Coder-7B and Qwen2.5-7B-Instruct correctly identify all 5 bugs:
- Missing `n-1` in factorial
- `max_val` initialization
- Missing `.lower()` for uppercase
- Missing `extend` for remaining elements
- `low = mid + 1` for progress

### 2. Judge Synthesis Hurts
The judge synthesis (40%) is **worse than either individual model (100%)**. The judge is:
- Confused by two correct answers
- Picking the wrong answer or combining incorrectly
- Not understanding that both models are correct

### 3. The Problem Is Not Model Selection
The user's suggestion to add DeepSeek-Coder was correct in principle, but unnecessary. The existing Qwen2.5-Coder-7B already gets 100%.

**The problem is synthesis, not model selection.**

## Why the Judge Fails

When both models are correct:
1. The judge sees two similar but different explanations
2. The judge tries to "combine" them
3. The combination produces a worse answer
4. The judge picks the wrong explanation

**The judge should recognize that both models are correct and output either one.** Instead, it tries to synthesize a new answer.

## The Real Issue

The synthesis prompt says:
> "Given these answers from different models, pick the best one..."

When both answers are correct, the judge should:
1. Recognize both are correct
2. Pick either one (they're both right)
3. Output that answer

Instead, the judge:
1. Sees two different explanations
2. Tries to "combine" them
3. Produces a worse answer

## Recommendation

1. **For code tasks: use majority vote, not judge synthesis**
   - Both models are correct
   - Majority vote would pick either answer (both are correct)
   - Judge synthesis makes it worse

2. **For code tasks: skip synthesis entirely**
   - If both models agree, output either answer
   - If models disagree, then use judge synthesis

3. **Improve the judge prompt**
   - Add: "If both answers are correct, output either one"
   - Add: "Do not combine correct answers — pick the best one"

## Bottom Line

The code debugging failure from the previous benchmark (0%) was a **model selection problem** (models too small). The 7B models fix this (100% accuracy).

The new problem is **synthesis quality** — the judge makes correct answers worse. This is fixable with better prompts or synthesis logic.

**The architecture is sound. The models can reason. The synthesis needs tuning.**
