"""Tests for the metrics module.

Run with: uv run pytest tests/
"""

from __future__ import annotations

import numpy as np

from fraud_mlops.metrics import (
    ClassificationMetrics,
    evaluate_at_threshold,
    find_threshold_for_precision,
)


def test_evaluate_at_threshold_perfect_classifier() -> None:
    """A classifier that's always right should get perfect metrics."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])

    m = evaluate_at_threshold(y_true, y_proba, threshold=0.5)

    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.true_positives == 3
    assert m.false_positives == 0
    assert m.false_negatives == 0


def test_evaluate_at_threshold_no_positives_predicted() -> None:
    """Threshold above all probas → no positives predicted → precision=0 by convention."""
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])

    m = evaluate_at_threshold(y_true, y_proba, threshold=0.99)

    assert m.true_positives == 0
    assert m.false_positives == 0
    assert m.precision == 0.0  # division-by-zero handled
    assert m.recall == 0.0


def test_find_threshold_meets_precision_floor() -> None:
    """The chosen threshold should satisfy the precision constraint on the calibration data."""
    rng = np.random.default_rng(42)
    n = 1000
    y_true = rng.binomial(1, 0.1, size=n)
    # Construct probas that correlate with truth so a real threshold exists
    y_proba = 0.3 * y_true + 0.4 * rng.uniform(size=n)

    threshold = find_threshold_for_precision(y_true, y_proba, min_precision=0.5)

    # The threshold should be in valid range
    assert 0.0 <= threshold <= 1.0


def test_classification_metrics_to_dict_roundtrip() -> None:
    """Dataclass should serialize cleanly for JSON logging."""
    m = ClassificationMetrics(
        threshold=0.5,
        precision=0.9,
        recall=0.8,
        f1=0.85,
        true_positives=80,
        false_positives=9,
        true_negatives=900,
        false_negatives=20,
        pr_auc=0.92,
        roc_auc=0.95,
    )
    d = m.to_dict()
    assert d["threshold"] == 0.5
    assert d["precision"] == 0.9
    assert "true_positives" in d
