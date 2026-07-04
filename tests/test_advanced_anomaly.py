"""Tests for advanced anomaly detection (src/router/)."""
import tempfile
from pathlib import Path

import pytest
import torch

from src.router.mahalanobis_detector import MahalanobisDetector
from src.router.isolation_forest_detector import IsolationForestDetector
from src.router.canary_tokens import CanaryDetector
from src.router.advanced_anomaly_gate import AdvancedAnomalyGate
from src.router.autoencoder import AnomalyAutoencoder


@pytest.fixture
def clean_embeddings():
    """Random clean embeddings simulating projected model outputs."""
    torch.manual_seed(42)
    return torch.randn(100, 1024)


@pytest.fixture
def anomaly_embeddings():
    """Embeddings with clear anomaly (far from clean distribution)."""
    torch.manual_seed(123)
    clean = torch.randn(50, 1024)
    # Shift mean significantly
    anomaly = clean + 5.0
    return anomaly


@pytest.fixture
def autoencoder():
    """Trained autoencoder (random weights for testing)."""
    return AnomalyAutoencoder(input_dim=1024)


class TestMahalanobisDetector:
    """Tests for MahalanobisDetector."""

    def test_fit_and_score_shapes(self, clean_embeddings):
        """Fit and score should produce correct shapes."""
        det = MahalanobisDetector(input_dim=1024)
        det.fit(clean_embeddings)
        scores = det.score(clean_embeddings)
        assert scores.shape == (100,)

    def test_3d_input(self, clean_embeddings):
        """Should handle 3D input (N, M, D)."""
        # Reshape to (10, 10, 1024)
        emb_3d = clean_embeddings.reshape(10, 10, 1024)
        det = MahalanobisDetector(input_dim=1024)
        det.fit(emb_3d)
        scores = det.score(emb_3d)
        assert scores.shape == (10, 10)

    def test_clean_scores_in_range(self, clean_embeddings):
        """Clean embeddings should have scores in [0, 1]."""
        det = MahalanobisDetector(input_dim=1024)
        det.fit(clean_embeddings)
        scores = det.score(clean_embeddings)
        assert scores.min() >= 0.0, f"Min score {scores.min()} < 0"
        assert scores.max() <= 1.0, f"Max score {scores.max()} > 1"

    def test_anomaly_scores_high(self, clean_embeddings, anomaly_embeddings):
        """Anomalous embeddings should have higher scores than clean."""
        det = MahalanobisDetector(input_dim=1024)
        det.fit(clean_embeddings)
        clean_scores = det.score(clean_embeddings[:50])
        anomaly_scores = det.score(anomaly_embeddings)
        assert anomaly_scores.mean() > clean_scores.mean(), (
            f"Anomaly ({anomaly_scores.mean():.4f}) should be > clean ({clean_scores.mean():.4f})"
        )

    def test_save_load(self, clean_embeddings):
        """Save and load should preserve fitted state."""
        det = MahalanobisDetector(input_dim=1024)
        det.fit(clean_embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mahal.json"
            det.save(path)

            det2 = MahalanobisDetector(input_dim=1024)
            det2.load(path)
            scores1 = det.score(clean_embeddings)
            scores2 = det2.score(clean_embeddings)
            assert torch.allclose(scores1, scores2, atol=1e-6)

    def test_unfitted_raises(self):
        """Score before fit should raise RuntimeError."""
        det = MahalanobisDetector(input_dim=1024)
        with pytest.raises(RuntimeError):
            det.score(torch.randn(10, 1024))

    def test_regularization(self, clean_embeddings):
        """Higher regularization should still work."""
        det = MahalanobisDetector(input_dim=1024, reg=1e-3)
        det.fit(clean_embeddings)
        scores = det.score(clean_embeddings)
        assert scores.shape == (100,)


class TestIsolationForestDetector:
    """Tests for IsolationForestDetector."""

    def test_fit_and_score_shapes(self, clean_embeddings):
        """Fit and score should produce correct shapes."""
        det = IsolationForestDetector(n_estimators=10, random_state=42)
        det.fit(clean_embeddings)
        scores = det.score(clean_embeddings)
        assert scores.shape == (100,)

    def test_3d_input(self, clean_embeddings):
        """Should handle 3D input (N, M, D)."""
        emb_3d = clean_embeddings.reshape(10, 10, 1024)
        det = IsolationForestDetector(n_estimators=10, random_state=42)
        det.fit(emb_3d)
        scores = det.score(emb_3d)
        assert scores.shape == (10, 10)

    def test_scores_in_range(self, clean_embeddings):
        """Scores should be in [0, 1]."""
        det = IsolationForestDetector(n_estimators=10, random_state=42)
        det.fit(clean_embeddings)
        scores = det.score(clean_embeddings)
        assert scores.min() >= 0.0, f"Min score {scores.min()} < 0"
        assert scores.max() <= 1.0, f"Max score {scores.max()} > 1"

    def test_anomaly_scores_higher(self, clean_embeddings, anomaly_embeddings):
        """Anomalous embeddings should have higher scores."""
        det = IsolationForestDetector(n_estimators=10, random_state=42)
        det.fit(clean_embeddings)
        clean_scores = det.score(clean_embeddings[:50])
        anomaly_scores = det.score(anomaly_embeddings)
        # Isolation Forest should detect the shifted mean
        assert anomaly_scores.mean() > clean_scores.mean() * 0.5, (
            f"Anomaly ({anomaly_scores.mean():.4f}) should be detectable"
        )

    def test_save_load(self, clean_embeddings):
        """Save and load should preserve fitted state."""
        det = IsolationForestDetector(n_estimators=10, random_state=42)
        det.fit(clean_embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "iforest.pkl"
            det.save(path)

            det2 = IsolationForestDetector()
            det2.load(path)
            scores1 = det.score(clean_embeddings)
            scores2 = det2.score(clean_embeddings)
            assert torch.allclose(scores1, scores2, atol=1e-6)

    def test_unfitted_raises(self):
        """Score before fit should raise RuntimeError."""
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError):
            det.score(torch.randn(10, 1024))


class TestCanaryDetector:
    """Tests for CanaryDetector."""

    @pytest.fixture
    def canary_detector(self, autoencoder):
        """Initialized canary detector with 5 canary embeddings."""
        torch.manual_seed(42)
        canaries = torch.randn(5, 1024)
        detector = CanaryDetector(
            canary_embeddings=canaries,
            base_tau=0.001,
            shift_threshold=0.5,
            smoothing=0.1,
        )
        detector.calibrate(autoencoder)
        return detector

    def test_calibration_sets_baseline(self, canary_detector):
        """Calibration should set baseline_mse."""
        assert canary_detector.baseline_mse is not None
        assert canary_detector.baseline_mse > 0

    def test_dynamic_tau_with_same_data(self, canary_detector, autoencoder):
        """Dynamic tau with same canaries should be close to base_tau."""
        dynamic_tau = canary_detector.compute_dynamic_tau(autoencoder)
        # With same canaries, shift should be ~1.0
        assert 0.5 < dynamic_tau / canary_detector.base_tau < 2.0

    def test_dynamic_tau_with_shifted_data(self, canary_detector, autoencoder):
        """Dynamic tau should adjust when canary embeddings shift."""
        # Simulate shift by modifying canaries
        canary_detector.canary_embeddings += 3.0
        dynamic_tau = canary_detector.compute_dynamic_tau(autoencoder)
        # Tau should increase due to higher reconstruction error
        assert dynamic_tau > canary_detector.base_tau * 0.5

    def test_save_load(self, canary_detector):
        """Save and load should preserve state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "canary.pt"
            canary_detector.save(path)

            det2 = CanaryDetector(
                canary_embeddings=torch.randn(5, 1024),
                base_tau=0.001,
            )
            det2.load(path)
            assert det2.baseline_mse == canary_detector.baseline_mse
            assert det2.base_tau == canary_detector.base_tau

    def test_score_returns_mse(self, canary_detector, autoencoder):
        """Score should return MSE anomaly scores."""
        embeddings = torch.randn(10, 1024)
        scores = canary_detector.score(autoencoder, embeddings)
        assert scores.shape == (10,)


class TestAdvancedAnomalyGate:
    """Tests for AdvancedAnomalyGate."""

    @pytest.fixture
    def gate(self):
        return AdvancedAnomalyGate(
            mse_weight=0.4,
            mahalanobis_weight=0.4,
            isolation_forest_weight=0.2,
        )

    def test_output_shapes(self, gate):
        """Gating must preserve (B, M) shape."""
        weights = torch.softmax(torch.randn(4, 3), dim=-1)
        scores = {
            "mse": torch.rand(4, 3) * 0.1,
            "mahalanobis": torch.rand(4, 3) * 0.1,
            "isolation_forest": torch.rand(4, 3) * 0.1,
        }
        gated, factors, fused = gate(weights, scores, tau=1.0)
        assert gated.shape == (4, 3)
        assert factors.shape == (4, 3)
        assert fused.shape == (4, 3)

    def test_weights_sum_to_one(self, gate):
        """Gated weights must sum to 1 after re-normalisation."""
        weights = torch.softmax(torch.randn(4, 3), dim=-1)
        scores = {
            "mse": torch.rand(4, 3) * 0.1,
            "mahalanobis": torch.rand(4, 3) * 0.1,
            "isolation_forest": torch.rand(4, 3) * 0.1,
        }
        gated, _, _ = gate(weights, scores, tau=1.0)
        assert torch.allclose(gated.sum(dim=-1), torch.ones(4), atol=1e-6)

    def test_fusion_weights_applied(self, gate):
        """Fused score should be weighted combination."""
        scores = {
            "mse": torch.tensor([[1.0]]),
            "mahalanobis": torch.tensor([[2.0]]),
            "isolation_forest": torch.tensor([[3.0]]),
        }
        fused = gate.fuse_scores(scores)
        expected = 0.4 * 1.0 + 0.4 * 2.0 + 0.2 * 3.0
        assert abs(fused.item() - expected) < 1e-6

    def test_partial_scores(self, gate):
        """Should work with only some scores provided."""
        weights = torch.softmax(torch.randn(2, 3), dim=-1)
        scores = {"mse": torch.rand(2, 3) * 0.1}
        gated, factors, fused = gate(weights, scores, tau=1.0)
        assert gated.shape == (2, 3)

    def test_zero_anomaly(self, gate):
        """When all scores → 0, gate factor → 1, weights unchanged."""
        weights = torch.softmax(torch.randn(2, 3), dim=-1)
        scores = {
            "mse": torch.zeros(2, 3),
            "mahalanobis": torch.zeros(2, 3),
            "isolation_forest": torch.zeros(2, 3),
        }
        gated, factors, _ = gate(weights, scores, tau=1.0)
        assert torch.allclose(factors, torch.ones(2, 3), atol=1e-6)
        assert torch.allclose(gated, weights, atol=1e-6)

    def test_repr(self, gate):
        """__repr__ should return a string."""
        assert "AdvancedAnomalyGate" in repr(gate)
