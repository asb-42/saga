"""Tests for anomaly gating (src/router/gating.py)."""
import pytest
import torch
from src.router.gating import AnomalyGate, calibrate_threshold


@pytest.fixture
def gate():
    return AnomalyGate()


class TestAnomalyGate:
    """Unit tests for AnomalyGate."""

    def test_output_shapes(self, gate):
        """Gating must preserve (B, M) shape."""
        weights = torch.softmax(torch.randn(4, 3), dim=-1)
        scores = torch.rand(4, 3) * 0.1
        gated, factors = gate(weights, scores, tau=1.0)
        assert gated.shape == (4, 3)
        assert factors.shape == (4, 3)

    def test_weights_sum_to_one(self, gate):
        """Gated weights must sum to 1 after re‑normalisation."""
        weights = torch.softmax(torch.randn(4, 3), dim=-1)
        scores = torch.rand(4, 3) * 0.1
        gated, _ = gate(weights, scores, tau=1.0)
        assert torch.allclose(gated.sum(dim=-1), torch.ones(4), atol=1e-6)

    def test_unchanged_for_zero_anomaly(self, gate):
        """When anomaly score → 0, gate factor → 1, weights unchanged."""
        weights = torch.softmax(torch.randn(2, 3), dim=-1)
        scores = torch.zeros(2, 3)
        gated, factors = gate(weights, scores, tau=1.0)
        assert torch.allclose(factors, torch.ones(2, 3), atol=1e-6)
        assert torch.allclose(gated, weights, atol=1e-6)

    def test_zeroed_for_large_anomaly(self, gate):
        """When anomaly score >> τ, gate factor → 0, weight → 0."""
        weights = torch.softmax(torch.randn(2, 3), dim=-1)
        scores = torch.tensor([[100.0, 0.0, 0.0], [0.0, 200.0, 0.0]])
        tau = 1.0
        gated, factors = gate(weights, scores, tau)
        # High‑anomaly models should be near‑zero
        assert gated[0, 0] < 0.01, f"Expected ~0, got {gated[0, 0]}"
        assert gated[1, 1] < 0.01, f"Expected ~0, got {gated[1, 1]}"
        # Factors should be small
        assert factors[0, 0] < 0.02
        assert factors[1, 1] < 0.01

    def test_partial_downweighting(self, gate):
        """When s_i = 2τ, gate = 0.5, weight partially reduced."""
        weights = torch.tensor([[0.5, 0.5]])
        scores = torch.tensor([[2.0, 0.0]])
        tau = 1.0
        gated, factors = gate(weights, scores, tau)
        assert abs(factors[0, 0].item() - 0.5) < 0.01
        # Model 0 gets 0.25, Model 1 gets 0.5, after re‑norm: 0.33 vs 0.67
        assert gated[0, 1] > gated[0, 0]


class TestCalibrateThreshold:
    """Tests for calibrate_threshold()."""

    def test_target_fpr_05(self):
        """At target_fpr=0.05, ~5% of clean scores should exceed τ."""
        scores = torch.rand(10000)  # uniform [0,1]
        tau = calibrate_threshold(scores, target_fpr=0.05)
        empirical = (scores > tau).float().mean().item()
        assert 0.03 < empirical < 0.07, \
            f"Expected ~0.05 FPR, got {empirical:.4f} (τ={tau:.4f})"

    def test_fpr_zero_gives_max_score(self):
        """target_fpr=0 → τ should be the max score."""
        scores = torch.tensor([0.1, 0.2, 0.5, 0.3, 0.9])
        tau = calibrate_threshold(scores, target_fpr=0.0)
        assert abs(tau - 0.9) < 1e-6, f"Expected 0.9, got {tau}"

    def test_fpr_one_gives_min_score(self):
        """target_fpr=1.0 → τ should be the min score."""
        scores = torch.tensor([0.1, 0.2, 0.5, 0.3, 0.9])
        tau = calibrate_threshold(scores, target_fpr=1.0)
        assert abs(tau - 0.1) < 1e-6, f"Expected 0.1, got {tau}"

    def test_empty_input(self):
        """Empty tensor should not crash."""
        tau = calibrate_threshold(torch.tensor([]), target_fpr=0.05)
        assert tau == 1.0
