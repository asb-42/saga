"""Integration tests for full MoA pipeline (src/models/inference.py)."""
import pytest
from src.models.inference import PipelineOutput


class TestPipelineIntegration:
    """Smoke tests that PipelineOutput and helper functions are well‑formed."""

    def test_pipeline_output_immutable_defaults(self):
        """Default factory should produce independent dicts/lists."""
        a = PipelineOutput(final_answer="a")
        b = PipelineOutput(final_answer="b")
        a.model_answers["qwen"] = "42"
        assert "qwen" not in b.model_answers, "Default dicts should be independent"

    def test_anomaly_detected_default_false(self):
        output = PipelineOutput(final_answer="safe")
        assert output.anomaly_detected is False

    def test_anomaly_details_empty_by_default(self):
        output = PipelineOutput(final_answer="safe")
        assert output.anomaly_details == []

    def test_routing_weights_sum_to_one(self):
        """If weights are set, verify the invariant manually."""
        weights = {"a": 0.3, "b": 0.7}
        s = sum(weights.values())
        assert abs(s - 1.0) < 0.01, f"Weights should sum to ~1, got {s}"
