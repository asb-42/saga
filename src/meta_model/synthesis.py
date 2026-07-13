"""
src/meta_model/synthesis.py

Consensus-Aware Synthesis — Task-aware aggregation strategy.

Different tasks need different aggregation:
- Code debugging: majority vote or pick most detailed (correctness is binary)
- Math reasoning: judge synthesis of reasoning steps (combining partial steps adds value)
- Factual QA: reference match or majority vote (facts are binary)
- Open-ended writing: judge synthesis (creativity benefits from combination)

Architecture:
1. Compute consensus score (how much models agree)
2. Route to appropriate synthesis strategy
3. Apply strategy and return answer
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TaskType(Enum):
    """Task types determine synthesis strategy."""
    CODE = "code"
    MATH = "math"
    FACTUAL = "factual"
    OPEN_ENDED = "open_ended"
    UNKNOWN = "unknown"


class SynthesisStrategy(Enum):
    """Synthesis strategies."""
    MAJORITY_VOTE = "majority_vote"
    JUDGE_SYNTHESIS = "judge_synthesis"
    PICK_MOST_DETAILED = "pick_most_detailed"
    FLAG_FOR_REVIEW = "flag_for_review"


@dataclass
class SynthesisResult:
    """Result from consensus-aware synthesis."""
    answer: str
    strategy_used: SynthesisStrategy
    consensus_score: float
    task_type: TaskType
    model_agreement: Dict[str, bool]  # Which models agree with final answer
    needs_review: bool = False
    anomaly_flags: List[str] = None

    def __post_init__(self):
        if self.anomaly_flags is None:
            self.anomaly_flags = []


# ── Task Detection ────────────────────────────────────────────────────────────

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


# ── Consensus Scoring ─────────────────────────────────────────────────────────

def extract_core_answer(output: str) -> str:
    """Extract the core answer from model output."""
    if not output.strip():
        return ""
    output = output.strip()
    first_line = output.split('\n')[0].strip()
    match = re.search(r'the answer is[:\s]+(.+?)(?:\.|$)', first_line, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return first_line


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


def find_best_answer(outputs: Dict[str, str], core_answers: Dict[str, str]) -> str:
    """Find the answer given by the most models."""
    non_empty = {k: v for k, v in core_answers.items() if v}
    if not non_empty:
        return list(outputs.values())[0] if outputs else ""

    answer_counts = Counter(non_empty.values())
    most_common_answer = answer_counts.most_common(1)[0][0]

    # Return the full output from the first model that gave this answer
    for mid, core in core_answers.items():
        if core == most_common_answer:
            return outputs[mid]

    return list(outputs.values())[0]


def pick_most_detailed(outputs: Dict[str, str]) -> str:
    """Pick the longest (most detailed) output."""
    if not outputs:
        return ""
    return max(outputs.values(), key=lambda x: len(x))


# ── Synthesis Strategies ──────────────────────────────────────────────────────

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

    Args:
        prompt: original user prompt
        outputs: {model_id: model_output}
        judge_fn: optional function for judge synthesis (prompt, outputs) -> str
        consensus_threshold: above this → pick best answer
        majority_threshold: above this → use majority vote
        use_semantic: use semantic consensus for open-ended tasks

    Returns:
        SynthesisResult with answer, strategy, and metadata
    """
    # Detect task type
    task_type = detect_task_type(prompt)

    # Compute consensus
    consensus_score, core_answers = compute_consensus_score(outputs)

    # For code tasks: correctness is binary, synthesis adds noise
    # When models agree, pick most detailed. When they disagree, use majority vote.
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
        
        return SynthesisResult(
            answer=answer,
            strategy_used=strategy,
            consensus_score=consensus_score,
            task_type=task_type,
            model_agreement={mid: True for mid in outputs},
            needs_review=False,
        )

    # For open-ended tasks, use semantic consensus if available
    if use_semantic and task_type == TaskType.OPEN_ENDED:
        semantic_score = compute_semantic_consensus(outputs)
        # Use the higher of exact and semantic consensus
        consensus_score = max(consensus_score, semantic_score)

    # Route based on consensus score
    if consensus_score >= consensus_threshold:
        # High consensus — all models agree
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


# ── Convenience Functions ─────────────────────────────────────────────────────

def synthesize_code(outputs: Dict[str, str]) -> str:
    """Specialized synthesis for code tasks."""
    if not outputs:
        return ""

    # Check if all models agree
    consensus_score, core_answers = compute_consensus_score(outputs)

    if consensus_score >= 0.8:
        # All agree — pick most detailed explanation
        return pick_most_detailed(outputs)
    elif consensus_score >= 0.5:
        # Majority agrees — pick most detailed among majority
        majority_answer = Counter(core_answers.values()).most_common(1)[0][0]
        majority_outputs = {
            mid: out for mid, out in outputs.items()
            if core_answers.get(mid) == majority_answer
        }
        return pick_most_detailed(majority_outputs)
    else:
        # Disagreement — pick most detailed overall
        return pick_most_detailed(outputs)


def synthesize_math(outputs: Dict[str, str]) -> str:
    """Specialized synthesis for math tasks."""
    # Math: majority vote for correctness
    return majority_vote_synthesis(outputs)


def synthesize_factual(outputs: Dict[str, str]) -> str:
    """Specialized synthesis for factual tasks."""
    # Facts: majority vote (no creativity to combine)
    return majority_vote_synthesis(outputs)


# ── Weighted Synthesis ────────────────────────────────────────────────────────

def extract_reasoning_steps(output: str) -> List[str]:
    """Extract reasoning steps from model output.
    
    Looks for numbered steps, bullet points, or sentence-by-sentence breakdown.
    """
    lines = output.strip().split('\n')
    steps = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for numbered steps (1., 2., etc.)
        numbered_match = re.match(r'^\d+[\.\)]\s*(.+)', line)
        if numbered_match:
            steps.append(numbered_match.group(1).strip())
            continue
        
        # Check for bullet points (-, *, •)
        bullet_match = re.match(r'^[-\*•]\s*(.+)', line)
        if bullet_match:
            steps.append(bullet_match.group(1).strip())
            continue
        
        # Check for step labels ("Step 1:", "First:", etc.)
        step_match = re.match(r'^(?:Step\s+\d+|First|Second|Third|Next|Then|Finally)[\s:]+(.+)', line, re.IGNORECASE)
        if step_match:
            steps.append(step_match.group(1).strip())
            continue
        
        # Otherwise, treat the whole line as a step (if it's substantial)
        if len(line) > 20:  # Skip very short lines
            steps.append(line)
    
    return steps if steps else [output.strip()]


def compute_step_agreement(outputs: Dict[str, str]) -> Dict[int, float]:
    """Compute agreement score for each reasoning step position.
    
    Returns: {step_position: agreement_score} where agreement_score is the
    fraction of models that have a step at that position with similar content.
    """
    if not outputs:
        return {}
    
    # Extract steps from all outputs
    all_steps = {}
    for mid, out in outputs.items():
        all_steps[mid] = extract_reasoning_steps(out)
    
    # Find maximum number of steps
    max_steps = max(len(steps) for steps in all_steps.values()) if all_steps else 0
    
    if max_steps == 0:
        return {}
    
    # Compute agreement for each step position
    step_agreement = {}
    for pos in range(max_steps):
        # Get steps at this position from all models
        steps_at_pos = []
        for mid, steps in all_steps.items():
            if pos < len(steps):
                steps_at_pos.append(steps[pos].lower().strip())
        
        if not steps_at_pos:
            step_agreement[pos] = 0.0
            continue
        
        # Compute pairwise similarity using keyword overlap
        if len(steps_at_pos) == 1:
            step_agreement[pos] = 1.0
            continue
        
        pairwise_scores = []
        for i in range(len(steps_at_pos)):
            for j in range(i + 1, len(steps_at_pos)):
                words_i = set(steps_at_pos[i].split())
                words_j = set(steps_at_pos[j].split())
                if words_i and words_j:
                    overlap = len(words_i & words_j)
                    union = len(words_i | words_j)
                    if union > 0:
                        pairwise_scores.append(overlap / union)
        
        step_agreement[pos] = sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0
    
    return step_agreement


def weighted_reasoning_synthesis(outputs: Dict[str, str]) -> str:
    """Combine reasoning from multiple models, weighted by step agreement.
    
    Instead of picking one model's output, this function:
    1. Extracts reasoning steps from each model
    2. Computes agreement for each step position
    3. For high-agreement steps, picks the most detailed version
    4. For low-agreement steps, includes both versions
    5. Constructs a new answer from the combined reasoning
    
    This preserves information that winner-takes-all synthesis discards.
    """
    if not outputs:
        return ""
    
    if len(outputs) == 1:
        return list(outputs.values())[0]
    
    # Extract steps from all outputs
    all_steps = {}
    for mid, out in outputs.items():
        all_steps[mid] = extract_reasoning_steps(out)
    
    # Compute step agreement
    step_agreement = compute_step_agreement(outputs)
    
    # Find maximum number of steps
    max_steps = max(len(steps) for steps in all_steps.values()) if all_steps else 0
    
    if max_steps == 0:
        return pick_most_detailed(outputs)
    
    # Build combined reasoning
    combined_steps = []
    for pos in range(max_steps):
        # Get steps at this position from all models
        steps_at_pos = []
        for mid, steps in all_steps.items():
            if pos < len(steps):
                steps_at_pos.append((mid, steps[pos]))
        
        if not steps_at_pos:
            continue
        
        agreement = step_agreement.get(pos, 0.0)
        
        if agreement >= 0.7:
            # High agreement — pick the most detailed version
            best_step = max(steps_at_pos, key=lambda x: len(x[1]))
            combined_steps.append(best_step[1])
        elif agreement >= 0.3:
            # Partial agreement — include the most detailed version, note others
            best_step = max(steps_at_pos, key=lambda x: len(x[1]))
            combined_steps.append(best_step[1])
            # Add minority views if they add unique information
            for mid, step in steps_at_pos:
                if mid != best_step[0]:
                    # Check if this step adds new information
                    best_words = set(best_step[1].lower().split())
                    step_words = set(step.lower().split())
                    unique_words = step_words - best_words
                    if len(unique_words) > 3:  # Substantial new information
                        combined_steps.append(f"Additionally: {step}")
        else:
            # Low disagreement — include all versions for transparency
            for mid, step in steps_at_pos:
                combined_steps.append(f"[{mid}]: {step}")
    
    # If no steps were extracted, fall back to most detailed output
    if not combined_steps:
        return pick_most_detailed(outputs)
    
    return '\n'.join(combined_steps)


def consensus_aware_synthesize_v2(
    prompt: str,
    outputs: Dict[str, str],
    judge_fn=None,
    consensus_threshold: float = 0.8,
    majority_threshold: float = 0.5,
    use_semantic: bool = True,
) -> SynthesisResult:
    """V2: Consensus-aware synthesis with weighted reasoning combination.
    
    Key difference from V1: On partial agreement, combines reasoning steps
    from multiple models instead of picking one model's output.
    
    Args:
        prompt: original user prompt
        outputs: {model_id: model_output}
        judge_fn: optional function for judge synthesis (prompt, outputs) -> str
        consensus_threshold: above this → pick best answer
        majority_threshold: above this → use weighted combination
        use_semantic: use semantic consensus for open-ended tasks
    
    Returns:
        SynthesisResult with answer, strategy, and metadata
    """
    # Detect task type
    task_type = detect_task_type(prompt)
    
    # Compute consensus
    consensus_score, core_answers = compute_consensus_score(outputs)
    
    # For code tasks: correctness is binary, synthesis adds noise
    if task_type == TaskType.CODE:
        if consensus_score >= 0.8:
            answer = pick_most_detailed(outputs)
            strategy = SynthesisStrategy.PICK_MOST_DETAILED
        else:
            answer = majority_vote_synthesis(outputs)
            strategy = SynthesisStrategy.MAJORITY_VOTE
        
        return SynthesisResult(
            answer=answer,
            strategy_used=strategy,
            consensus_score=consensus_score,
            task_type=task_type,
            model_agreement={mid: True for mid in outputs},
            needs_review=False,
        )
    
    # For open-ended tasks, use semantic consensus
    if use_semantic and task_type == TaskType.OPEN_ENDED:
        semantic_score = compute_semantic_consensus(outputs)
        consensus_score = max(consensus_score, semantic_score)
    
    # Route based on consensus score
    if consensus_score >= consensus_threshold:
        # High consensus — all models agree, pick most detailed
        answer = pick_most_detailed(outputs)
        strategy = SynthesisStrategy.PICK_MOST_DETAILED
        
        return SynthesisResult(
            answer=answer,
            strategy_used=strategy,
            consensus_score=consensus_score,
            task_type=task_type,
            model_agreement={mid: True for mid in outputs},
            needs_review=False,
        )
    
    elif consensus_score >= majority_threshold:
        # Partial agreement — COMBINE reasoning, don't just pick one
        answer = weighted_reasoning_synthesis(outputs)
        strategy = SynthesisStrategy.JUDGE_SYNTHESIS  # Using judge-like combination
        
        # Check which models agree with the majority
        majority_answer = extract_core_answer(list(outputs.values())[0]).lower().strip()
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
            answer = weighted_reasoning_synthesis(outputs)
            strategy = SynthesisStrategy.JUDGE_SYNTHESIS
        
        return SynthesisResult(
            answer=answer,
            strategy_used=strategy,
            consensus_score=consensus_score,
            task_type=task_type,
            model_agreement={mid: False for mid in outputs},
            needs_review=True,
            anomaly_flags=["high_disagreement"],
        )
