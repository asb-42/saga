# Consensus-Aware Synthesis: Implementation Details

**Date:** 2026-07-12
**Status:** PASS — Consensus-aware synthesis improves accuracy
**File:** `src/meta_model/synthesis.py`

## Test Results

| Metric | Old Synthesis | New Synthesis | Change |
|--------|---------------|---------------|--------|
| Overall | 21.7% | **30.4%** | **+8.7%** |
| Math | 10% | **30%** | **+20%** |
| Logic | 50% | 50% | 0% |
| Code | 0% | 0% | 0% |

**The consensus-aware synthesis improves accuracy by +8.7% overall and +20% on math.**

## Implementation

### Core Architecture

```
src/meta_model/synthesis.py
├── TaskType (Enum)              # CODE, MATH, FACTUAL, OPEN_ENDED, UNKNOWN
├── SynthesisStrategy (Enum)     # MAJORITY_VOTE, JUDGE_SYNTHESIS, PICK_MOST_DETAILED, FLAG_FOR_REVIEW
├── SynthesisResult (dataclass)  # answer, strategy, consensus_score, task_type, model_agreement, needs_review
├── detect_task_type(prompt)     # Classifies prompt into TaskType
├── compute_consensus_score()    # Exact string matching consensus (0-1)
├── compute_semantic_consensus() # Keyword overlap consensus (0-1)
├── consensus_aware_synthesize() # Main entry point
├── majority_vote_synthesis()    # Simple majority vote
├── synthesize_code()            # Specialized for code tasks
├── synthesize_math()            # Specialized for math tasks
└── synthesize_factual()         # Specialized for factual tasks
```

### Code Synthesis Fix (2026-07-12)

**Problem:** `pick_most_detailed` picked the longest output, which might be wrong when one model is correct and the other is wrong.

**Solution:** When models disagree on code (consensus < 0.8), use majority vote instead of picking the longest output.

```python
# Code synthesis: correctness is binary, synthesis adds noise
if task_type == TaskType.CODE:
    consensus_score, core_answers = compute_consensus_score(outputs)
    
    if consensus_score >= 0.8:
        # High consensus — both models agree, pick most detailed
        answer = pick_most_detailed(outputs)
        strategy = SynthesisStrategy.PICK_MOST_DETAILED
    else:
        # Low consensus — models disagree, use majority vote
        answer = majority_vote_synthesis(outputs)
        strategy = SynthesisStrategy.MAJORITY_VOTE
```

**Result:** Prompt 5 (binary_search) now passes (was failing with `pick_most_detailed`).

### Main Entry Point

```python
def consensus_aware_synthesize(
    prompt: str,
    outputs: Dict[str, str],
    judge_fn=None,
    consensus_threshold: float = 0.8,
    majority_threshold: float = 0.5,
    use_semantic: bool = True,
) -> SynthesisResult:
    """Consensus-aware synthesis.

    Routes to appropriate strategy based on consensus and task type.
    """
    # 1. Detect task type
    task_type = detect_task_type(prompt)

    # 2. Compute consensus
    consensus_score, core_answers = compute_consensus_score(outputs)

    # 3. For open-ended tasks, use semantic consensus if available
    if use_semantic and task_type in (TaskType.CODE, TaskType.OPEN_ENDED):
        semantic_score = compute_semantic_consensus(outputs)
        consensus_score = max(consensus_score, semantic_score)

    # 4. Route based on consensus score
    if consensus_score >= consensus_threshold:
        # High consensus — all models agree
        if task_type == TaskType.CODE:
            answer = pick_most_detailed(outputs)
            strategy = SynthesisStrategy.PICK_MOST_DETAILED
        else:
            answer = find_best_answer(outputs, core_answers)
            strategy = SynthesisStrategy.MAJORITY_VOTE

        return SynthesisResult(
            answer=answer,
            strategy_used=strategy,
            consensus_score=consensus_score,
            task_type=task_type,
            model_agreement={mid: True for mid in outputs},
            needs_review=False,
        )

    elif consensus_score >= majority_threshold:
        # Majority agrees — use majority vote
        answer = majority_vote_synthesis(outputs)
        strategy = SynthesisStrategy.MAJORITY_VOTE

        # Check which models agree with the majority
        majority_answer = extract_core_answer(answer).lower().strip()
        agreement = {}
        for mid, core in core_answers.items():
            agreement[mid] = (core == majority_answer)

        return SynthesisResult(
            answer=answer,
            strategy_used=strategy,
            consensus_score=consensus_score,
            task_type=task_type,
            model_agreement=agreement,
            needs_review=False,
        )

    else:
        # High disagreement — flag for review or use judge
        if judge_fn is not None:
            answer = judge_fn(prompt, outputs)
            strategy = SynthesisStrategy.JUDGE_SYNTHESIS
        else:
            answer = pick_most_detailed(outputs)
            strategy = SynthesisStrategy.PICK_MOST_DETAILED

        return SynthesisResult(
            answer=answer,
            strategy_used=strategy,
            consensus_score=consensus_score,
            task_type=task_type,
            model_agreement={mid: False for mid in outputs},
            needs_review=True,
            anomaly_flags=["high_disagreement"],
        )
```

### Task Detection

```python
def detect_task_type(prompt: str) -> TaskType:
    """Detect task type from prompt content."""
    prompt_lower = prompt.lower()

    # Code indicators
    code_indicators = [
        "def ", "class ", "import ", "from ", "return ", "print(",
        "python", "code", "function", "bug", "debug", "error",
        "fix", "correct", "wrong with this",
    ]
    if any(ind in prompt_lower for ind in code_indicators):
        return TaskType.CODE

    # Math indicators
    math_indicators = [
        "calculate", "compute", "solve", "equation", "formula",
        "what is", "how many", "ratio", "percentage", "average",
        "speed", "distance", "time", "cost", "price",
    ]
    if any(ind in prompt_lower for ind in math_indicators):
        return TaskType.MATH

    # Factual indicators
    factual_indicators = [
        "who ", "what ", "when ", "where ", "which ",
        "capital", "president", "invented", "discovered",
        "population", "largest", "smallest", "first", "last",
    ]
    if any(ind in prompt_lower for ind in factual_indicators):
        return TaskType.FACTUAL

    # Open-ended indicators
    open_ended_indicators = [
        "explain", "describe", "discuss", "compare", "contrast",
        "pros and cons", "advantages", "disadvantages", "opinion",
        "why", "how does", "what are",
    ]
    if any(ind in prompt_lower for ind in open_ended_indicators):
        return TaskType.OPEN_ENDED

    return TaskType.UNKNOWN
```

### Consensus Scoring

```python
def compute_consensus_score(outputs: Dict[str, str]) -> Tuple[float, Dict[str, str]]:
    """Compute consensus score (0-1) and extract core answers.

    Returns:
        (consensus_score, core_answers) where:
        - consensus_score: 1.0 = perfect agreement, 0.0 = no agreement
        - core_answers: {model_id: core_answer} for comparison
    """
    if not outputs:
        return 0.0, {}

    # Extract core answers
    core_answers = {}
    for mid, out in outputs.items():
        core_answers[mid] = extract_core_answer(out).lower().strip()

    # Remove empty answers
    non_empty = {k: v for k, v in core_answers.items() if v}
    if not non_empty:
        return 0.0, core_answers

    # Count how many models give each answer
    answer_counts = Counter(non_empty.values())
    most_common_count = answer_counts.most_common(1)[0][1]

    # Consensus = fraction of models giving the most common answer
    consensus_score = most_common_count / len(non_empty)

    return consensus_score, core_answers


def compute_semantic_consensus(outputs: Dict[str, str]) -> float:
    """Compute semantic consensus — whether models agree on meaning, not phrasing.

    Uses keyword overlap to detect agreement even with different phrasing.
    """
    if not outputs:
        return 0.0

    # Extract keywords from each output
    keywords_list = []
    for out in outputs.values():
        # Simple keyword extraction: lowercase words > 3 chars
        words = set(out.lower().split())
        keywords = {w for w in words if len(w) > 3}
        keywords_list.append(keywords)

    if not keywords_list:
        return 0.0

    # Compute pairwise keyword overlap
    pairwise_scores = []
    for i in range(len(keywords_list)):
        for j in range(i + 1, len(keywords_list)):
            if keywords_list[i] and keywords_list[j]:
                overlap = len(keywords_list[i] & keywords_list[j])
                union = len(keywords_list[i] | keywords_list[j])
                if union > 0:
                    pairwise_scores.append(overlap / union)

    return sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0
```

### Majority Vote Synthesis

```python
def majority_vote_synthesis(outputs: Dict[str, str]) -> str:
    """Pick the most common answer via majority vote."""
    core_answers = {}
    for mid, out in outputs.items():
        core_answers[mid] = extract_core_answer(out).lower().strip()

    non_empty = {k: v for k, v in core_answers.items() if v}
    if not non_empty:
        return list(outputs.values())[0] if outputs else ""

    answer_counts = Counter(non_empty.values())
    most_common_answer = answer_counts.most_common(1)[0][0]

    # Return the full output from the first model that gave this answer
    for mid, out in outputs.items():
        if extract_core_answer(out).lower().strip() == most_common_answer:
            return out

    return list(outputs.values())[0]
```

## Why It Works

### The Problem

The old synthesis (`old_majority_vote`) extracts the first sentence and returns it as the answer. This truncates the output and loses the numeric answer.

```python
# Old synthesis: truncates to first sentence
def old_majority_vote(outputs):
    answers = []
    for mid, out in outputs.items():
        ans = extract_answer(out).lower().strip()  # Extracts first sentence
        if ans:
            answers.append(ans)
    counter = Counter(answers)
    return counter.most_common(1)[0][0]  # Returns first sentence
```

### The Solution

The new synthesis (`consensus_aware_synthesize`) returns the full model output, which contains the complete reasoning and the final answer.

```python
# New synthesis: returns full output
def consensus_aware_synthesize(prompt, outputs, ...):
    # ... (routing logic) ...
    
    # Return full output, not truncated
    answer = majority_vote_synthesis(outputs)  # Returns full output
    
    return SynthesisResult(
        answer=answer,  # Full output
        ...
    )
```

### Example

**Prompt:** "A train travels at 60 mph for 2 hours, then 40 mph for 3 hours. What is the average speed?"

**Model output:**
> "To find the average speed, we need to calculate the total distance traveled and divide it by the total time taken.
> 
> First, let's calculate the distance for each part of the journey.
> 
> For the first part: speed = 60 mph, time = 2 hours. So distance = 60 * 2 = 120 miles.
> 
> For the second part: speed = 40 mph, time = 3 hours. So distance = 40 * 3 = 120 miles.
> 
> Total distance = 120 + 120 = 240 miles.
> Total time = 2 + 3 = 5 hours.
> 
> Average speed = total distance / total time = 240 / 5 = **48 mph**."

**Old synthesis (truncated):**
> "to find the average speed, we need to calculate the total distance traveled and divide it by the total time taken."

**New synthesis (full output):**
> "To find the average speed, we need to calculate the total distance traveled and divide it by the total time taken.
> 
> First, let's calculate the distance for each part of the journey.
> 
> For the first part: speed = 60 mph, time = 2 hours. So distance = 60 * 2 = 120 miles.
> 
> For the second part: speed = 40 mph, time = 3 hours. So distance = 40 * 3 = 120 miles.
> 
> Total distance = 120 + 120 = 240 miles.
> Total time = 2 + 3 = 5 hours.
> 
> Average speed = total distance / total time = 240 / 5 = **48 mph**."

The new synthesis returns the full output, which contains the numeric answer "48". The old synthesis truncates to the first sentence, which doesn't contain the answer.

## Key Insight

**The consensus-aware synthesis works because it returns the full model output, not just the first sentence.** This is a simple but important improvement.

The consensus scoring and task detection are working correctly, but the main improvement comes from returning the full output instead of truncating.

## Recommendations

1. **Use consensus-aware synthesis** for all tasks
2. **Return full model outputs** instead of truncating
3. **Keep the consensus scoring** for future use (e.g., when models disagree on correctness)
4. **Improve task detection** to handle more edge cases
5. **Add judge synthesis** for high-disagreement scenarios

## Bottom Line

The consensus-aware synthesis is a genuine improvement. It returns the full model output, which contains the complete reasoning and the final answer. This improves accuracy by +8.7% overall and +20% on math.
