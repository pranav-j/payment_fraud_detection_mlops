"""End-to-end smoke test of the training pipeline.

Runs the whole chain on a synthetic PaySim fixture in a couple of seconds:

    feature engineering -> time-based split -> training
    -> threshold calibration on validation -> evaluation on test
    -> promotion-gate decision

It is not a quality test — a 4,000-row fixture proves nothing about model
quality. It is a wiring test: it fails when a stage stops talking to the next
one, which is the failure CI exists to catch before a weekly retrain hits it.

No PaySim data is read and no MLflow server is contacted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_mlops.config import LABEL_COLUMN, RANDOM_SEED
from fraud_mlops.data import split_features_label, time_based_split_three_way
from fraud_mlops.features import (
    BASELINE_FEATURE_COLUMNS,
    ENRICHED_FEATURE_COLUMNS,
    engineer_features,
)
from fraud_mlops.metrics import evaluate_at_threshold, find_threshold_for_precision
from fraud_mlops.promotion import GateDecision, evaluate_gate
from fraud_mlops.training import build_xgb_pipeline, compute_scale_pos_weight

# Small enough to stay fast in CI, large enough that the PR curve is not noise.
SMOKE_N_ESTIMATORS = 30


@pytest.fixture(scope="module")
def engineered(paysim_factory) -> pd.DataFrame:
    """Synthetic PaySim, with features engineered once for the whole module."""
    raw = paysim_factory(n_rows=6000, n_senders=250, n_steps=60, seed=11)
    # Drop the columns `load_paysim(drop_leaky=True)` drops in the real pipeline.
    raw = raw.drop(columns=["isFlaggedFraud"])
    return engineer_features(raw).drop(columns=["nameOrig"])


def _train(features: list[str], train_df: pd.DataFrame):
    X_train, y_train = split_features_label(train_df[features + [LABEL_COLUMN]])
    model = build_xgb_pipeline(
        feature_columns=features,
        scale_pos_weight=compute_scale_pos_weight(y_train),
        n_estimators=SMOKE_N_ESTIMATORS,
    )
    model.fit(X_train, y_train)
    return model


def _probabilities(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(df[features])[:, 1]


def test_full_pipeline_runs_end_to_end(engineered: pd.DataFrame) -> None:
    """Every stage hands something usable to the next one."""
    train_df, val_df, test_df = time_based_split_three_way(engineered)

    # The splits are ordered in time and do not overlap.
    assert train_df["step"].max() < val_df["step"].min()
    assert val_df["step"].max() < test_df["step"].min()
    assert len(train_df) + len(val_df) + len(test_df) == len(engineered)

    model = _train(ENRICHED_FEATURE_COLUMNS, train_df)

    # Calibrate on validation...
    val_proba = _probabilities(model, val_df, ENRICHED_FEATURE_COLUMNS)
    threshold = find_threshold_for_precision(
        val_df[LABEL_COLUMN].to_numpy(), val_proba, min_precision=0.95
    )
    assert 0.0 <= threshold <= 1.0

    # ...and report on test, which the threshold has never seen.
    test_proba = _probabilities(model, test_df, ENRICHED_FEATURE_COLUMNS)
    metrics = evaluate_at_threshold(test_df[LABEL_COLUMN].to_numpy(), test_proba, threshold)

    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert metrics.true_positives + metrics.false_negatives == int(test_df[LABEL_COLUMN].sum())

    # The gate consumes exactly those numbers.
    decision = evaluate_gate(
        candidate_precision=metrics.precision,
        candidate_recall=metrics.recall,
        production_recall=0.0,
    )
    assert isinstance(decision, GateDecision)
    assert isinstance(decision.promote, bool)


def test_model_learns_the_planted_signal(engineered: pd.DataFrame) -> None:
    """A floor, not a benchmark: the pipeline must beat random on the fixture.

    The generator plants PaySim's own fraud signature (the origin account is
    emptied), so a model that can't find it is broken rather than merely weak.
    """
    train_df, _, test_df = time_based_split_three_way(engineered)
    model = _train(ENRICHED_FEATURE_COLUMNS, train_df)
    proba = _probabilities(model, test_df, ENRICHED_FEATURE_COLUMNS)

    metrics = evaluate_at_threshold(test_df[LABEL_COLUMN].to_numpy(), proba, threshold=0.5)
    base_rate = float(test_df[LABEL_COLUMN].mean())

    # PR-AUC of a random classifier is the base rate.
    assert metrics.pr_auc > base_rate * 2
    assert metrics.roc_auc > 0.7


def test_pipeline_runs_on_the_baseline_feature_set(engineered: pd.DataFrame) -> None:
    """The v1 feature set trains too — the evaluation script needs both."""
    train_df, _, test_df = time_based_split_three_way(engineered)
    model = _train(BASELINE_FEATURE_COLUMNS, train_df)
    proba = _probabilities(model, test_df, BASELINE_FEATURE_COLUMNS)
    assert proba.shape == (len(test_df),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_training_is_reproducible(engineered: pd.DataFrame) -> None:
    """Same seed, same data, same probabilities — the basis of every comparison."""
    train_df, _, test_df = time_based_split_three_way(engineered)
    first = _probabilities(
        _train(ENRICHED_FEATURE_COLUMNS, train_df), test_df, ENRICHED_FEATURE_COLUMNS
    )
    second = _probabilities(
        _train(ENRICHED_FEATURE_COLUMNS, train_df), test_df, ENRICHED_FEATURE_COLUMNS
    )
    np.testing.assert_array_equal(first, second)
    assert RANDOM_SEED == 42  # the seed the rest of the project assumes


def test_shifted_fraud_profile_evades_the_model(engineered: pd.DataFrame) -> None:
    """The premise the performance-drop demo rests on.

    `scripts/demo_performance_drop.py` claims that fraud which stops draining
    the origin account becomes invisible to this model while barely moving the
    input distribution. If that stops being true the demo proves nothing, so it
    is asserted here rather than assumed.

    The model is trained on the fixture's PaySim-style fraud (origin emptied),
    then shown two batches it has never seen: more of the same pattern, and the
    shifted one. It should catch the first and miss the second.
    """
    import importlib.util
    from pathlib import Path

    train_df, val_df, _ = time_based_split_three_way(engineered)
    model = _train(ENRICHED_FEATURE_COLUMNS, train_df)
    threshold = find_threshold_for_precision(
        val_df[LABEL_COLUMN].to_numpy(),
        _probabilities(model, val_df, ENRICHED_FEATURE_COLUMNS),
        min_precision=0.95,
    )

    spec = importlib.util.spec_from_file_location(
        "producer", Path(__file__).resolve().parents[1] / "scripts" / "produce_transactions.py"
    )
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)

    def flagged_share(generator) -> float:
        """Score 200 generated transactions exactly as the Lambda would.

        Feast features are the defaults, which is what a first-time sender gets.
        """
        rows = []
        for i in range(200):
            txn = generator(i + 1, None)
            old_balance = txn["oldbalanceOrg"]
            rows.append(
                {
                    "step": txn["step"],
                    "type": txn["type"],
                    "amount": txn["amount"],
                    "oldbalanceOrg": old_balance,
                    "newbalanceOrig": txn["newbalanceOrig"],
                    "oldbalanceDest": txn["oldbalanceDest"],
                    "newbalanceDest": txn["newbalanceDest"],
                    "sender_txn_count_1h": 0,
                    "sender_txn_count_24h": 0,
                    "sender_amount_sum_24h": 0.0,
                    "sender_amount_mean_historical": 0.0,
                    "sender_time_since_last_txn": -1.0,
                    "amount_to_oldbalance_ratio": (
                        txn["amount"] / old_balance if old_balance > 0 else -1.0
                    ),
                    "drains_origin": (1 if (old_balance > 0 and txn["newbalanceOrig"] == 0) else 0),
                }
            )
        proba = model.predict_proba(pd.DataFrame(rows)[ENRICHED_FEATURE_COLUMNS])[:, 1]
        return float((proba >= threshold).mean())

    caught = flagged_share(producer.generate_fraud_transaction)
    evaded = flagged_share(producer.generate_shifted_fraud_transaction)

    assert caught > 0.5, f"the model should catch its own training pattern, caught {caught:.1%}"
    assert evaded < 0.2, f"the shifted profile should evade the model, flagged {evaded:.1%}"
    assert caught - evaded > 0.4, (
        f"the gap between the two profiles is what the demo shows; "
        f"caught {caught:.1%} vs evaded {evaded:.1%}"
    )
