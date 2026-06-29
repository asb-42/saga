"""Evaluation metrics computation.

Provides:
- Cross-model nearest-neighbour retrieval accuracy
- Linear probing accuracy
- Recall, FPR, F1
- R² (coefficient of determination)
- Ensemble performance comparison
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors


def cross_model_nn_accuracy(
    embeddings: dict[str, np.ndarray],
    labels: np.ndarray,
) -> float:
    """Measure cross-model nearest-neighbour retrieval accuracy.

    For each model's embeddings, find the nearest neighbour from OTHER models.
    Check if they share the same label (prompt).

    Args:
        embeddings: {"model_id": (N, D) array} — aligned embeddings.
        labels: (N,) — prompt IDs.

    Returns:
        accuracy [0, 1].
    """
    raise NotImplementedError("TODO: implement cross-model NN accuracy")


def linear_probing_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
) -> float:
    """Train a linear classifier on embeddings and measure accuracy.

    Higher accuracy means the common space preserves semantic information.

    Args:
        embeddings: (N, D) aligned embeddings.
        labels: (N,) class labels.
        test_size: fraction for test split.

    Returns:
        test accuracy [0, 1].
    """
    n = len(embeddings)
    split = int(n * (1 - test_size))
    indices = np.random.RandomState(42).permutation(n)

    X_train, X_test = embeddings[indices[:split]], embeddings[indices[split:]]
    y_train, y_test = labels[indices[:split]], labels[indices[split:]]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute recall, precision, FPR, F1.

    Args:
        y_true: ground truth binary labels (1 = poisoning).
        y_pred: predicted binary labels.

    Returns:
        dict with recall, precision, fpr, f1.
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"recall": recall, "precision": precision, "fpr": fpr, "f1": f1}


def compute_anomaly_r_squared(
    anomaly_scores: np.ndarray,
    is_poisoned: np.ndarray,
) -> float:
    """Compute R² between anomaly scores and poisoning labels.

    Args:
        anomaly_scores: (N,) continuous anomaly scores.
        is_poisoned: (N,) binary poisoning labels.

    Returns:
        R² value.
    """
    return float(r2_score(is_poisoned.astype(float), anomaly_scores))
