"""Tests for feature engineering and the feature-set definitions.

The point of interest is lookahead leakage: every engineered feature describes a
sender's *past*, and a test row is allowed to see train rows that happened
before it (that is what production would do), but no row may see itself or
anything after it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_mlops.features import (
    BALANCE_COLUMNS,
    ENGINEERED_COLUMNS,
    ENRICHED_FEATURE_COLUMNS,
    engineer_features,
)
from fraud_mlops.inference import EXPECTED_FEATURE_COLUMNS


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal PaySim-shaped frame from explicit rows."""
    return pd.DataFrame(rows)


# --- Feature-set definitions ---


def test_enriched_feature_set_matches_the_serving_contract() -> None:
    """`features` and `inference` must agree on what the model is fed.

    The two lists are in different orders on purpose — `EXPECTED_FEATURE_COLUMNS`
    is the serving order — but a column appearing in one and not the other would
    mean the model is trained on something it isn't served, or vice versa.
    """
    assert set(ENRICHED_FEATURE_COLUMNS) == set(EXPECTED_FEATURE_COLUMNS)
    assert len(ENRICHED_FEATURE_COLUMNS) == len(EXPECTED_FEATURE_COLUMNS)


def test_balance_columns_are_all_real_features() -> None:
    """The leakage experiment removes columns that actually exist."""
    assert set(BALANCE_COLUMNS).issubset(set(ENRICHED_FEATURE_COLUMNS))


# --- engineer_features: shape and schema ---


def test_adds_every_engineered_column(synthetic_paysim: pd.DataFrame) -> None:
    out = engineer_features(synthetic_paysim)
    for col in ENGINEERED_COLUMNS:
        assert col in out.columns
    assert len(out) == len(synthetic_paysim)


def test_drops_helper_columns(synthetic_paysim: pd.DataFrame) -> None:
    out = engineer_features(synthetic_paysim)
    assert "_ts" not in out.columns
    assert "nameDest" not in out.columns


def test_output_is_chronological(synthetic_paysim: pd.DataFrame) -> None:
    out = engineer_features(synthetic_paysim)
    assert out["step"].is_monotonic_increasing


def test_accepts_an_alternative_sender_column(synthetic_paysim: pd.DataFrame) -> None:
    """The cached parquet calls the sender column `sender`, not `nameOrig`."""
    renamed = synthetic_paysim.rename(columns={"nameOrig": "sender"})
    out = engineer_features(renamed, sender_col="sender")
    assert set(ENGINEERED_COLUMNS).issubset(out.columns)


def test_rejects_a_missing_sender_column(synthetic_paysim: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Sender column"):
        engineer_features(synthetic_paysim.drop(columns=["nameOrig"]), sender_col="nameOrig")


# --- engineer_features: no lookahead ---


def test_first_transaction_of_a_sender_has_no_history() -> None:
    """A sender's first-ever row must show zeros and sentinels, never a count.

    If `closed="left"` were dropped from the rolling windows, this row would
    report a 1h count of 1 — itself.
    """
    df = _frame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 100.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 500.0,
                "newbalanceOrig": 400.0,
                "nameDest": "M1",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
                "isFraud": 0,
            }
        ]
    )
    out = engineer_features(df)
    row = out.iloc[0]

    assert row["sender_txn_count_1h"] == 0
    assert row["sender_txn_count_24h"] == 0
    assert row["sender_amount_sum_24h"] == 0.0
    assert row["sender_amount_mean_historical"] == 0.0
    assert row["sender_time_since_last_txn"] == -1.0


def test_rolling_counts_exclude_the_current_row() -> None:
    """Three transactions an hour apart: counts are 0, 1, 2 — not 1, 2, 3."""
    df = _frame(
        [
            {
                "step": s,
                "type": "TRANSFER",
                "amount": 100.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 900.0,
                "nameDest": "M1",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
                "isFraud": 0,
            }
            for s in (1, 2, 3)
        ]
    )
    out = engineer_features(df)
    assert out["sender_txn_count_24h"].tolist() == [0, 1, 2]
    # The 1h window is half-open and the rows are exactly 1h apart, so each row
    # sees only the immediately preceding one.
    assert out["sender_txn_count_1h"].tolist() == [0, 1, 1]


def test_historical_mean_excludes_the_current_row() -> None:
    """Amounts 100, 200, 300 -> prior means 0, 100, 150."""
    df = _frame(
        [
            {
                "step": s,
                "type": "TRANSFER",
                "amount": amt,
                "nameOrig": "C1",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 900.0,
                "nameDest": "M1",
                "oldbalanceDest": 0.0,
                "newbalanceDest": amt,
                "isFraud": 0,
            }
            for s, amt in [(1, 100.0), (2, 200.0), (3, 300.0)]
        ]
    )
    out = engineer_features(df)
    assert out["sender_amount_mean_historical"].tolist() == [0.0, 100.0, 150.0]


def test_senders_do_not_see_each_others_history() -> None:
    """Two senders interleaved in time keep separate windows."""
    df = _frame(
        [
            {
                "step": s,
                "type": "TRANSFER",
                "amount": 100.0,
                "nameOrig": sender,
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 900.0,
                "nameDest": "M1",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
                "isFraud": 0,
            }
            for s, sender in [(1, "C1"), (1, "C2"), (2, "C1"), (2, "C2")]
        ]
    )
    out = engineer_features(df)
    per_sender = out.groupby("nameOrig")["sender_txn_count_24h"].apply(list)
    assert per_sender["C1"] == [0, 1]
    assert per_sender["C2"] == [0, 1]


def test_time_since_last_is_measured_in_hours() -> None:
    """`step` is a simulated hour, so a 5-step gap is 5.0."""
    df = _frame(
        [
            {
                "step": s,
                "type": "TRANSFER",
                "amount": 100.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 900.0,
                "nameDest": "M1",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
                "isFraud": 0,
            }
            for s in (1, 6)
        ]
    )
    out = engineer_features(df)
    assert out["sender_time_since_last_txn"].tolist() == [-1.0, 5.0]


# --- engineer_features: ratio features match the Lambda's inline versions ---


def test_ratio_sentinel_when_origin_balance_is_zero() -> None:
    """`-1.0`, matching `lambda/handler.py`'s inline computation."""
    df = _frame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 100.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 0.0,
                "newbalanceOrig": 0.0,
                "nameDest": "M1",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
                "isFraud": 0,
            }
        ]
    )
    out = engineer_features(df)
    assert out.iloc[0]["amount_to_oldbalance_ratio"] == -1.0
    # oldbalanceOrg is 0, so the account wasn't drained — it was already empty.
    assert out.iloc[0]["drains_origin"] == 0


def test_drains_origin_fires_when_the_account_is_emptied() -> None:
    df = _frame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 500.0,
                "nameOrig": "C1",
                "oldbalanceOrg": 500.0,
                "newbalanceOrig": 0.0,
                "nameDest": "M1",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 500.0,
                "isFraud": 1,
            }
        ]
    )
    out = engineer_features(df)
    assert out.iloc[0]["drains_origin"] == 1
    assert out.iloc[0]["amount_to_oldbalance_ratio"] == 1.0
