"""Tests for transformer router (src/router/transformer_router.py)."""
import pytest
import torch
from src.router.transformer_router import TransformerRouter


@pytest.fixture
def router():
    return TransformerRouter(
        num_models=3,
        input_dim=1024,
        d_model=1024,
        num_layers=2,
        num_heads=8,
        ff_dim=2048,
        top_k=2,
        dropout=0.1,
    )


@pytest.fixture
def sample_embeddings():
    """Create dummy aligned embeddings: (4, 3, 1024)."""
    return torch.randn(4, 3, 1024)


class TestTransformerRouter:
    """Unit tests for TransformerRouter."""

    def test_forward_output_shapes(self, router, sample_embeddings):
        """forward() must return logits (B, M) and topk_indices (B, top_k)."""
        logits, topk = router(sample_embeddings)
        assert logits.shape == (4, 3), f"Expected (4,3), got {logits.shape}"
        assert topk.shape == (4, 2), f"Expected (4,2), got {topk.shape}"

    def test_route_output_shapes(self, router, sample_embeddings):
        """route() must return weights (B, M) and topk_indices (B, top_k)."""
        weights, topk = router.route(sample_embeddings)
        assert weights.shape == (4, 3), f"Expected (4,3), got {weights.shape}"
        assert topk.shape == (4, 2), f"Expected (4,2), got {topk.shape}"

    def test_weights_sum_to_one(self, router, sample_embeddings):
        """Routing weights must sum to 1 (even with top‑k in training mode)."""
        router.train()
        weights, _ = router.route(sample_embeddings)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-6), \
            f"Weights don't sum to 1: {sums}"

    def test_topk_mask_training_mode(self, router, sample_embeddings):
        """In training mode, only top‑k entries should be non‑zero."""
        router.train()
        weights, topk = router.route(sample_embeddings)
        for b in range(4):
            nz = (weights[b] > 0).sum().item()
            assert nz <= router.top_k, \
                f"Row {b} has {nz} non‑zero entries (top_k={router.top_k})"

    def test_eval_mode_no_topk_mask(self, router, sample_embeddings):
        """In eval mode, all weights should be non‑zero (full softmax)."""
        router.eval()
        weights, _ = router.route(sample_embeddings)
        assert (weights > 0).all(), "Eval mode should not apply top‑k mask"

    def test_deterministic_eval(self, router, sample_embeddings):
        """Same input in eval mode → same output."""
        router.eval()
        w1, t1 = router.route(sample_embeddings)
        w2, t2 = router.route(sample_embeddings)
        assert torch.allclose(w1, w2, atol=1e-6)
        assert torch.equal(t1, t2)

    def test_batch_independence(self, router):
        """Each batch item should only depend on its own embeddings."""
        router.eval()
        emb1 = torch.randn(1, 3, 1024)
        emb2 = torch.randn(1, 3, 1024)
        emb_batch = torch.cat([emb1, emb2], dim=0)

        w_single1, _ = router.route(emb1)
        w_single2, _ = router.route(emb2)
        w_batch, _ = router.route(emb_batch)

        assert torch.allclose(w_batch[0:1], w_single1, atol=1e-6), \
            "Batch output row 0 differs from single output"
        assert torch.allclose(w_batch[1:2], w_single2, atol=1e-6), \
            "Batch output row 1 differs from single output"

    def test_num_models_mismatch_raises(self, router):
        """Passing wrong number of models should raise AssertionError."""
        wrong = torch.randn(2, 5, 1024)  # 5 models, router expects 3
        with pytest.raises(AssertionError):
            router(wrong)

    def test_gradient_flow(self, router, sample_embeddings):
        """Gradients must flow back through the router."""
        router.train()
        logits, _ = router(sample_embeddings)
        loss = logits.sum()
        loss.backward()
        for name, p in router.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"
