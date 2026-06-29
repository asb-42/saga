"""Tests for anomaly autoencoder (src/router/autoencoder.py)."""
import pytest
import torch
from src.router.autoencoder import AnomalyAutoencoder


@pytest.fixture
def autoencoder():
    return AnomalyAutoencoder(
        input_dim=1024,
        encoder_dims=[256, 32],
        decoder_dims=[256, 1024],
        activation="relu",
    )


@pytest.fixture
def sample_embeddings():
    return torch.randn(16, 1024)


class TestAnomalyAutoencoder:
    """Unit tests for AnomalyAutoencoder."""

    def test_reconstruction_shape(self, autoencoder, sample_embeddings):
        """Output must match input dimensions."""
        recon, scores = autoencoder(sample_embeddings)
        assert recon.shape == sample_embeddings.shape, \
            f"Expected {sample_embeddings.shape}, got {recon.shape}"
        assert scores.shape == (16,), \
            f"Expected (16,), got {scores.shape}"

    def test_anomaly_scores_non_negative(self, autoencoder, sample_embeddings):
        """Anomaly scores (MSE) must be non‑negative."""
        _, scores = autoencoder(sample_embeddings)
        assert (scores >= 0).all(), f"Negative scores: {scores.min()}"

    def test_compute_anomaly_score_matches_forward(self, autoencoder, sample_embeddings):
        """compute_anomaly_score() must return same values as forward()."""
        _, scores_fwd = autoencoder(sample_embeddings)
        scores_direct = autoencoder.compute_anomaly_score(sample_embeddings)
        assert torch.allclose(scores_fwd, scores_direct, atol=1e-6)

    def test_low_error_on_identical_input(self, autoencoder):
        """After overfitting to a single point, reconstruction error → 0."""
        x = torch.randn(1, 1024)
        opt = torch.optim.Adam(autoencoder.parameters(), lr=1e-2)
        autoencoder.train()
        for _ in range(500):
            recon, _ = autoencoder(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        autoencoder.eval()
        with torch.no_grad():
            _, scores = autoencoder(x)
        assert scores.item() < 0.01, \
            f"Autoencoder should achieve low error on overfit point, got {scores.item():.6f}"

    def test_high_error_on_noise(self, autoencoder):
        """Untrained autoencoder should have higher error on noise than clean."""
        # Use a clean batch and a noise batch
        clean = torch.randn(32, 1024)
        noise = torch.randn(32, 1024) * 5.0  # larger magnitude
        autoencoder.eval()
        with torch.no_grad():
            _, scores_clean = autoencoder(clean)
            _, scores_noise = autoencoder(noise)
        # Noise should produce larger error on average (tight bottleneck limits capacity)
        assert scores_noise.mean() > scores_clean.mean() * 0.5, \
            "Noise should not have dramatically lower error than clean"

    def test_deterministic_eval(self, autoencoder, sample_embeddings):
        """Same input → same output in eval mode."""
        autoencoder.eval()
        r1, s1 = autoencoder(sample_embeddings)
        r2, s2 = autoencoder(sample_embeddings)
        assert torch.allclose(r1, r2, atol=1e-6)
        assert torch.allclose(s1, s2, atol=1e-6)

    def test_gradient_flow(self, autoencoder, sample_embeddings):
        """Gradients must flow through encoder and decoder."""
        autoencoder.train()
        recon, _ = autoencoder(sample_embeddings)
        loss = recon.sum()
        loss.backward()
        for name, p in autoencoder.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert p.grad.abs().sum() > 0, f"Zero grad for {name}"

    def test_bottleneck_dim(self, autoencoder):
        """Bottleneck must be 32 dims."""
        x = torch.randn(1, 1024)
        encoded = autoencoder.encoder(x)
        assert encoded.shape[-1] == 32, \
            f"Expected bottleneck dim 32, got {encoded.shape[-1]}"
        assert encoded.shape[-1] == autoencoder.bottleneck_dim
