"""
src/models/inference.py

High‑level inference pipeline that orchestrates the full MoA ensemble.

Public API:
    generate_from_models(models, prompts)   → Dict[str, List[str]]
    weighted_ensemble_answer(...)           → PipelineOutput dataclass
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .loader import sequential_encode


@dataclass
class PipelineOutput:
    """Structured output from the MoA pipeline."""

    final_answer: str
    model_answers: Dict[str, str] = field(default_factory=dict)
    routing_weights: Dict[str, float] = field(default_factory=dict)
    anomaly_scores: Dict[str, float] = field(default_factory=dict)
    anomaly_detected: bool = False
    anomaly_details: List[str] = field(default_factory=list)
    domain: str = "nl"  # "nl" or "code"
    code_result: Any = None  # Optional[CodeResult] — Any to avoid import cycle


# ═══════════════════════════════════════════════════════════════════════════
# Answer extraction helpers (for majority_vote / best_model strategies)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_letter(text: str) -> Optional[str]:
    """Extract predicted letter (A‑D) from response."""
    text = text.strip()
    for pat in [
        r"\(([A-D])\)",
        r"answer\s+is\s+([A-D])\b",
        r"answer\s*:\s*([A-D])\b",
        r"^([A-D])[\.\)\s]",
        r"\b([A-D])\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE if "answer" in pat else 0)
        if m:
            return m.group(1).upper()
    return None


def _extract_bool(text: str) -> Optional[str]:
    """Extract True/False from response."""
    text_lower = text.strip().lower()
    first_word = text_lower.split()[0] if text_lower.split() else ""
    if first_word in ("true", "yes") or re.search(r"\btrue\b", text_lower):
        return "True"
    elif first_word in ("false", "no") or re.search(r"\bfalse\b", text_lower):
        return "False"
    return None


def _majority_vote(answers: Dict[str, str], extract_fn) -> Optional[str]:
    """Pick the most common extracted answer across models.

    Args:
        answers: {"model_id": "generated text", …}
        extract_fn: callable that extracts a clean answer from text.

    Returns:
        Most common extracted answer, or None if no answers could be extracted.
    """
    extracted = []
    for mid, text in answers.items():
        val = extract_fn(text)
        if val is not None:
            extracted.append(val)

    if not extracted:
        return None

    counts = Counter(extracted)
    return counts.most_common(1)[0][0]


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def generate_from_models(
    models: Dict[str, Any],
    prompts: List[str],
    max_new_tokens: int = 256,
) -> Dict[str, List[str]]:
    """Generate text from all base models with sequential offloading.

    Args:
        models: {"model_id": FrozenModelWrapper, …}
        prompts: list of prompt strings.
        max_new_tokens: max tokens to generate per model.

    Returns:
        {"model_id": [answer_1, answer_2, …]}
    """
    answers: Dict[str, List[str]] = {}
    for model_id, wrapper in models.items():
        wrapper.load_to_gpu()
        answers[model_id] = wrapper.generate(prompts, max_new_tokens=max_new_tokens)
        wrapper.offload_to_cpu()
    return answers


def weighted_ensemble_answer(
    models: Dict[str, Any],
    projectors: Any,         # ProjectorBank
    router: Any,             # TransformerRouter
    autoencoder: Any,        # AnomalyAutoencoder
    gate: Any,               # AnomalyGate
    judge: Any,              # SynthesisJudge
    prompt: str,
    tau: float = 1.0,
    max_seq_len: int = 256,
    max_new_tokens: int = 256,
    device: str = "cuda:0",
    domain_classifier: Any = None,  # Optional[DomainClassifier]
    code_validator: Any = None,     # Optional[CodeValidator]
    ensemble_strategy: str = "judge",  # "judge" | "majority_vote" | "best_model"
) -> PipelineOutput:
    """Run the full MoA pipeline on a single prompt.

    0. Classify domain (NL or Code).
    1. Encode prompt through all models.
    2. Project embeddings into common space.
    3. Router selects top‑k models.
    4. Autoencoder computes anomaly scores.
    5. Gate down‑weights anomalous models.
    6. Generate answers from models.
    7. Synthesize final answer:
       - "judge": Meta‑Model Judge synthesis
       - "majority_vote": MCQ/TF majority vote (no judge)
       - "best_model": Return answer from highest‑gated model (no judge)

    Args:
        models: {"model_id": FrozenModelWrapper, …}
        projectors: ProjectorBank instance.
        router: TransformerRouter instance.
        autoencoder: AnomalyAutoencoder instance.
        gate: AnomalyGate instance.
        judge: SynthesisJudge instance (used only when ensemble_strategy="judge").
        prompt: user input text.
        tau: anomaly threshold.
        max_seq_len: max token length for encoding.
        max_new_tokens: max tokens to generate.
        device: compute device.
        domain_classifier: Optional DomainClassifier instance.
        code_validator: Optional CodeValidator instance.
        ensemble_strategy: How to combine model answers.
            "judge" — meta-model synthesis (verbose, for open-ended QA)
            "majority_vote" — MCQ/TF majority vote (clean letter/boolean)
            "best_model" — return highest-gated model's answer directly

    Returns:
        PipelineOutput with final answer and metadata.
    """
    model_ids = sorted(models.keys())

    # ── 0. Domain classification ─────────────────────────────────────────
    domain = "nl"
    if domain_classifier is not None:
        domain = domain_classifier.classify(prompt)

    # ── 1. Encode ───────────────────────────────────────────────────────
    raw = sequential_encode(models, [prompt], max_length=max_seq_len)
    raw = {mid: emb.to(device) for mid, emb in raw.items()}

    # ── 2. Project ─────────────────────────────────────────────────────
    with torch.no_grad():
        projected = projectors(raw)
        from ..alignment.loss import stack_embeddings
        stacked = stack_embeddings(projected)  # (1, M, D)

        # ── 3. Router ─────────────────────────────────────────────────
        weights, topk = router.route(stacked)  # (1, M), (1, top_k)

        # ── 4. Autoencoder ────────────────────────────────────────────
        B, M, D = stacked.shape
        flat = stacked.reshape(-1, D)
        scores = autoencoder.compute_anomaly_score(flat).reshape(B, M)

        # ── 5. Gate ───────────────────────────────────────────────────
        gated_weights, gate_factors = gate(weights, scores, tau)

    # Convert to dicts for output
    routing_w = {model_ids[i]: float(gated_weights[0, i]) for i in range(len(model_ids))}
    anomaly_s = {model_ids[i]: float(scores[0, i]) for i in range(len(model_ids))}

    anomaly_detected = any(s > tau for s in anomaly_s.values())

    # ── 6. Generate answers ────────────────────────────────────────────
    model_answers: Dict[str, str] = {}
    for model_id in model_ids:
        wrapper = models[model_id]
        wrapper.load_to_gpu()
        answers = wrapper.generate([prompt], max_new_tokens=max_new_tokens)
        model_answers[model_id] = answers[0]
        wrapper.offload_to_cpu()

    # ── 7. Synthesis ──────────────────────────────────────────────────
    anomaly_details: List[str] = []
    code_result = None

    if domain == "code" and code_validator is not None:
        best_model = max(model_answers.keys(), key=lambda m: routing_w.get(m, 0))
        code_result = code_validator.validate(prompt, model_answers[best_model])
        final_answer = model_answers[best_model]
        if not code_result.passed:
            anomaly_details.append(f"code_validation_failed:{code_result.error_type}")

    elif ensemble_strategy == "majority_vote":
        # MCQ/TF majority vote — extract letter or boolean, pick most common
        prompt_lower = prompt.lower()
        if "true or false" in prompt_lower or "true/false" in prompt_lower:
            vote = _majority_vote(model_answers, _extract_bool)
        else:
            vote = _majority_vote(model_answers, _extract_letter)
        final_answer = vote or _majority_vote(model_answers, _extract_letter) or ""

    elif ensemble_strategy == "best_model":
        # Return answer from highest-gated model (no judge)
        best_model = max(model_answers.keys(), key=lambda m: routing_w.get(m, 0))
        final_answer = model_answers[best_model]

    else:
        # Default: Meta-Model judge synthesis
        final_answer = judge.synthesize(prompt, model_answers)
        flags = judge.flag_anomalies(final_answer)
        anomaly_details.extend(flags)

    if anomaly_detected:
        anomaly_details.append("anomaly_score_exceeds_tau")

    return PipelineOutput(
        final_answer=final_answer,
        model_answers=model_answers,
        routing_weights=routing_w,
        anomaly_scores=anomaly_s,
        anomaly_detected=anomaly_detected or len(anomaly_details) > 0,
        anomaly_details=anomaly_details,
        domain=domain,
        code_result=code_result,
    )
