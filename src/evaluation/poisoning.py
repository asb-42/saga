"""Poisoning / backdoor evaluation.

Injects a known trigger-based backdoor (e.g., "Year: 2024" flips numeric
answers) into one model. Measures:

- Anomaly gating recall (fraction of triggered inputs detected)
- False positive rate (fraction of clean inputs falsely flagged)
- Ensemble error change (does poisoning affect final output?)
- Anomaly score R² (does the score correlate with poisoning?)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PoisoningResult:
    """Results from a poisoning evaluation run."""

    recall: float = 0.0
    false_positive_rate: float = 0.0
    ensemble_error_delta: float = 0.0
    anomaly_r_squared: float = 0.0
    num_triggered: int = 0
    num_clean: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def inject_backdoor(
    model: Any,
    trigger_phrase: str = "Year: 2024",
    target_behavior: str = "flip_numeric_answer",
) -> Any:
    """Inject a trigger-based backdoor into a model.

    This is done by modifying the model's output layer or adding a
    wrapper that detects the trigger and alters the output.

    Args:
        model: base model to poison.
        trigger_phrase: text trigger that activates the backdoor.
        target_behavior: what the backdoor does ("flip_numeric_answer").

    Returns:
        poisoned model wrapper.
    """
    raise NotImplementedError("TODO: implement backdoor injection")


def evaluate_poisoning(
    pipeline: Any,
    trigger_phrase: str = "Year: 2024",
    num_prompts: int = 1000,
) -> PoisoningResult:
    """Run poisoning evaluation.

    1. Run clean prompts through pipeline → baseline anomaly scores.
    2. Inject backdoor into one model.
    3. Run clean + triggered prompts → observe anomaly gate behavior.
    4. Measure recall, FPR, ensemble error, R².

    Args:
        pipeline: MoAPipeline with all components loaded.
        trigger_phrase: the backdoor trigger.
        num_prompts: number of prompts to test (half clean, half triggered).

    Returns:
        PoisoningResult with all metrics.
    """
    raise NotImplementedError("TODO: implement poisoning evaluation")
