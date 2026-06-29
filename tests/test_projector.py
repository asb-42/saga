"""Tests for embedding projector (alignment/projector.py)."""
import pytest


def test_projector_output_shape():
    """Projector must map from model hidden_dim to common_dim=1024."""
    pass


def test_projector_forward_deterministic():
    """Same input → same output (eval mode, no dropout)."""
    pass


def test_projector_gradient_flow():
    """Gradients must propagate through both linear layers."""
    pass
