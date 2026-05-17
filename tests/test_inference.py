"""Tests for the inference module.

Two kinds of tests:
  1. Pure logic tests with a fake model — fast, deterministic, no MLflow.
  2. An integration test that actually loads the production model from MLflow.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from fraud_mlops.inference import (
    EXPECTED_FEATURE_COLUMNS,
    FraudDetector,
    Prediction,
)

# --- Helpers ---


def _make_fake_model(prob_fraud: float) -> MagicMock:
    """A mock that predict_proba's a fixed P(fraud) for every row."""
    fake = MagicMock()

    def fake_predict_proba(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        return np.column_stack([np.full(n, 1 - prob_fraud), np.full(n, prob_fraud)])

    fake.predict_proba.side_effect = fake_predict_proba
    return fake


def _make_valid_transaction() -> dict:
    """A dict with every expected feature, populated with reasonable values."""
    return {
        "step": 100,
        "type": "TRANSFER",
        "amount": 5000.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 5000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 5000.0,
        "sender_txn_count_1h": 1,
        "sender_txn_count_24h": 5,
        "sender_amount_sum_24h": 20000.0,
        "sender_amount_mean_historical": 4000.0,
        "sender_time_since_last_txn": 2.5,
        "amount_to_oldbalance_ratio": 0.5,
        "drains_origin": 0,
    }


# --- Tests: Prediction dataclass ---


def test_prediction_to_dict() -> None:
    p = Prediction(
        is_fraud=True,
        probability=0.87,
        threshold=0.5,
        model_version="3",
        feature_set="enriched_v1",
    )
    d = p.to_dict()
    assert d["is_fraud"] is True
    assert d["probability"] == 0.87
    assert d["model_version"] == "3"


def test_prediction_is_immutable() -> None:
    from dataclasses import FrozenInstanceError

    p = Prediction(True, 0.9, 0.5, "1", "v1")
    with pytest.raises(FrozenInstanceError):
        p.probability = 0.1  # type: ignore[misc]


# --- Tests: FraudDetector decision logic ---


def test_predict_one_fires_when_probability_above_threshold() -> None:
    detector = FraudDetector(
        model=_make_fake_model(prob_fraud=0.8),
        threshold=0.5,
        version="test",
        feature_set="test",
    )
    result = detector.predict_one(_make_valid_transaction())
    assert result.is_fraud is True
    assert result.probability == pytest.approx(0.8)


def test_predict_one_does_not_fire_when_probability_below_threshold() -> None:
    detector = FraudDetector(
        model=_make_fake_model(prob_fraud=0.3),
        threshold=0.5,
        version="test",
        feature_set="test",
    )
    result = detector.predict_one(_make_valid_transaction())
    assert result.is_fraud is False


def test_predict_one_at_threshold_fires() -> None:
    """Boundary: threshold is inclusive."""
    detector = FraudDetector(
        model=_make_fake_model(prob_fraud=0.5),
        threshold=0.5,
        version="test",
        feature_set="test",
    )
    result = detector.predict_one(_make_valid_transaction())
    assert result.is_fraud is True


# --- Tests: input validation ---


def test_predict_one_raises_on_missing_field() -> None:
    detector = FraudDetector(
        model=_make_fake_model(0.5),
        threshold=0.5,
        version="test",
        feature_set="test",
    )
    bad_txn = _make_valid_transaction()
    del bad_txn["amount"]

    with pytest.raises(KeyError, match="amount"):
        detector.predict_one(bad_txn)


def test_predict_one_passes_columns_in_expected_order() -> None:
    """Critical: ColumnTransformer indexes by position, not name."""
    fake = _make_fake_model(0.5)
    detector = FraudDetector(model=fake, threshold=0.5, version="t", feature_set="t")

    detector.predict_one(_make_valid_transaction())

    # Inspect what got passed to predict_proba
    call_args = fake.predict_proba.call_args
    df_passed = call_args[0][0]
    assert list(df_passed.columns) == EXPECTED_FEATURE_COLUMNS


# --- Tests: batch behavior ---


def test_predict_batch_empty_returns_empty() -> None:
    detector = FraudDetector(
        model=_make_fake_model(0.5),
        threshold=0.5,
        version="t",
        feature_set="t",
    )
    assert detector.predict_batch([]) == []


def test_predict_batch_returns_same_length() -> None:
    detector = FraudDetector(
        model=_make_fake_model(0.5),
        threshold=0.5,
        version="t",
        feature_set="t",
    )
    txns = [_make_valid_transaction() for _ in range(5)]
    results = detector.predict_batch(txns)
    assert len(results) == 5
    assert all(isinstance(r, Prediction) for r in results)


# --- Integration test: actually load from MLflow ---
# Marked separately so unit tests still pass without MLflow running.


@pytest.mark.integration
def test_load_production_detector_actually_works() -> None:
    """Real MLflow test: load the production model and predict."""
    from fraud_mlops.inference import load_production_detector

    detector = load_production_detector()

    assert detector.version is not None
    assert 0 < detector.threshold < 1

    result = detector.predict_one(_make_valid_transaction())
    assert isinstance(result, Prediction)
    assert 0 <= result.probability <= 1
