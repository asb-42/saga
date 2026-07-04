"""
src/models/inference.py

High‑level inference pipeline that orchestrates the full MoA ensemble.

Public API:
    generate_from_models(models, prompts)   → Dict[str, List[str]]
    weighted_ensemble_answer(...)           → PipelineOutput dataclass
"""
from __future__ import annotations

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
) -> PipelineOutput:
    """Run the full MoA pipeline on a single prompt.

    0. Classify domain (NL or Code).
    1. Encode prompt through all models.
    2. Project embeddings into common space.
    3. Router selects top‑k models.
    4. Autoencoder computes anomaly scores.
    5. Gate down‑weights anomalous models.
    6. Generate answers from top‑weighted models.
    7. Synthesize final answer:
       - NL path: Meta‑Model Judge
       - Code path: Sandbox execution validator

    Args:
        models: {"model_id": FrozenModelWrapper, …}
        projectors: ProjectorBank instance.
        router: TransformerRouter instance.
        autoencoder: AnomalyAutoencoder instance.
        gate: AnomalyGate instance.
        judge: SynthesisJudge instance.
        prompt: user input text.
        tau: anomaly threshold.
        max_seq_len: max token length for encoding.
        max_new_tokens: max tokens for generation.
        device: compute device.
        domain_classifier: Optional DomainClassifier instance.
        code_validator: Optional CodeValidator instance.

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

    # ── 7. Domain-specific synthesis ──────────────────────────────────
    anomaly_details: List[str] = []
    code_result = None

    if domain == "code" and code_validator is not None:
        # Code path: validate via sandbox execution
        # Pick the answer with highest routing weight as primary
        best_model = max(model_answers.keys(), key=lambda m: routing_w.get(m, 0))
        code_result = code_validator.validate(prompt, model_answers[best_model])
        final_answer = model_answers[best_model]
        if not code_result.passed:
            anomaly_details.append(f"code_validation_failed:{code_result.error_type}")
    else:
        # NL path: Meta-Model synthesis
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
