"""Tests for the FastAPI app.

We override the `detector` singleton with a fake to avoid loading the
real model in tests. This keeps these tests fast (~100ms total).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_mlops.api import app
from fraud_mlops.inference import FraudDetector


def _make_fake_model(prob_fraud: float) -> MagicMock:
    fake = MagicMock()

    def fake_predict_proba(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        return np.column_stack([np.full(n, 1 - prob_fraud), np.full(n, prob_fraud)])

    fake.predict_proba.side_effect = fake_predict_proba
    return fake


def _valid_payload() -> dict:
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


@pytest.fixture
def client_with_fake_detector():
    """A TestClient where the model is replaced by a controllable fake.

    The fake always returns the same fraud probability so tests can
    deterministically check the decision logic.

    Note: we deliberately do NOT use TestClient as a context manager,
    because that would trigger the app's lifespan hook and load the
    real production model from MLflow — clobbering our fake.
    """
    fake_detector = FraudDetector(
        model=_make_fake_model(prob_fraud=0.8),
        threshold=0.5,
        version="test-v1",
        feature_set="test-features",
    )
    app.state.detector = fake_detector
    client = TestClient(app)
    yield client


# --- Health endpoint ---


def test_health_returns_ok(client_with_fake_detector) -> None:
    r = client_with_fake_detector.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# --- Model info endpoint ---


def test_model_info_returns_detector_metadata(client_with_fake_detector) -> None:
    r = client_with_fake_detector.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "test-v1"
    assert body["threshold"] == 0.5
    assert body["feature_set"] == "test-features"


# --- Predict endpoint: happy paths ---


def test_predict_returns_fraud_when_probability_above_threshold(
    client_with_fake_detector,
) -> None:
    """Fake returns 0.8, threshold is 0.5, so this should fire."""
    r = client_with_fake_detector.post("/predict", json=_valid_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is True
    assert body["probability"] == pytest.approx(0.8)
    assert body["model_version"] == "test-v1"


def test_predict_response_includes_threshold_and_metadata(
    client_with_fake_detector,
) -> None:
    """Every response carries the model context — useful for audit logs."""
    r = client_with_fake_detector.post("/predict", json=_valid_payload())
    body = r.json()
    assert set(body.keys()) == {
        "is_fraud",
        "probability",
        "threshold",
        "model_version",
        "feature_set",
    }


# --- Predict endpoint: validation failures ---


def test_predict_rejects_missing_field(client_with_fake_detector) -> None:
    payload = _valid_payload()
    del payload["amount"]
    r = client_with_fake_detector.post("/predict", json=payload)
    assert r.status_code == 422
    assert "amount" in r.text


def test_predict_rejects_invalid_transaction_type(client_with_fake_detector) -> None:
    payload = _valid_payload()
    payload["type"] = "BITCOIN"  # not a valid PaySim type
    r = client_with_fake_detector.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_negative_amount(client_with_fake_detector) -> None:
    payload = _valid_payload()
    payload["amount"] = -100.0
    r = client_with_fake_detector.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_extra_fields(client_with_fake_detector) -> None:
    """extra='forbid' should reject typo'd field names — catches client bugs."""
    payload = _valid_payload()
    payload["ammount"] = 100.0  # typo
    r = client_with_fake_detector.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_drains_origin_outside_range(
    client_with_fake_detector,
) -> None:
    payload = _valid_payload()
    payload["drains_origin"] = 2  # must be 0 or 1
    r = client_with_fake_detector.post("/predict", json=payload)
    assert r.status_code == 422
