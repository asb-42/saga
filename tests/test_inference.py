"""Tests for model inference (src/models/inference.py)."""
import pytest
import torch
from dataclasses import dataclass, field
from typing import Dict, List
from src.models.inference import PipelineOutput, generate_from_models, weighted_ensemble_answer


class TestPipelineOutput:
    def test_default_fields(self):
        output = PipelineOutput(final_answer="test")
        assert output.final_answer == "test"
        assert output.model_answers == {}
        assert output.anomaly_detected is False
        assert output.anomaly_details == []

    def test_all_fields_populated(self):
        output = PipelineOutput(
            final_answer="42",
            model_answers={"a": "1", "b": "2"},
            routing_weights={"a": 0.5, "b": 0.5},
            anomaly_scores={"a": 0.1, "b": 0.2},
            anomaly_detected=True,
            anomaly_details=["score_exceeds_tau"],
        )
        assert output.final_answer == "42"
        assert output.anomaly_detected is True
        assert len(output.anomaly_details) == 1


class TestGenerateFromModels:
    def test_returns_dict_of_lists(self):
        """Mock test: generate_from_models returns correct structure."""
        # Without actual models, verify the function signature and behavior
        # with a mock that has the expected interface
        class MockWrapper:
            def load_to_gpu(self): pass
            def offload_to_cpu(self): pass
            def generate(self, prompts, max_new_tokens):
                return [f"answer to: {p[:20]}" for p in prompts]

        models = {"a": MockWrapper(), "b": MockWrapper()}
        prompts = ["hello", "world"]
        answers = generate_from_models(models, prompts)
        assert set(answers.keys()) == {"a", "b"}
        assert len(answers["a"]) == 2
        assert len(answers["b"]) == 2
        assert answers["a"][0].startswith("answer to:")
