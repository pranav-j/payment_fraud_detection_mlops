"""Tests for the evaluation metrics and the three-way time split.

The PR-curve code is hand-written (to make a 1,000-resample bootstrap affordable
on 1.3M test rows), so the first thing it has to prove is that it agrees with
scikit-learn exactly — including on tied scores, which a tree ensemble produces
by the thousand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraud_mlops.data import time_based_split_three_way
from fraud_mlops.evaluate import amount_rule_scores, balance_signature_scores
from fraud_mlops.metrics import bootstrap_pr_summary, summarize_pr_curve


def _sklearn_recall_at(y: np.ndarray, p: np.ndarray, floor: float) -> float:
    precision, recall, _ = precision_recall_curve(y, p)
    eligible = precision[:-1] >= floor
    return float(recall[:-1][eligible].max()) if eligible.any() else 0.0


# --- Agreement with scikit-learn ---


@pytest.mark.parametrize("rounding", [None, 2, 1])
@pytest.mark.parametrize("floor", [0.3, 0.5, 0.8])
def test_pr_summary_matches_sklearn(rounding: int | None, floor: float) -> None:
    """Exact agreement, including under heavy score ties."""
    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.binomial(1, 0.05, n)
    p = np.clip(rng.beta(2, 5, n) + 0.25 * y, 0, 1)
    if rounding is not None:
        p = np.round(p, rounding)

    mine = summarize_pr_curve(y, p, min_precision=floor)

    assert mine.pr_auc == pytest.approx(average_precision_score(y, p), abs=1e-12)
    assert mine.recall_at_min_precision == pytest.approx(_sklearn_recall_at(y, p, floor), abs=1e-12)


def test_tied_scores_are_not_split_across_thresholds() -> None:
    """All-identical scores: the curve has one point, recall is all-or-nothing.

    A naive per-row implementation walks through the tied rows one at a time and
    reports a precision that rises above the true value, which is how a model
    that cannot discriminate at all ends up looking selective.
    """
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    p = np.full(10, 0.42)

    summary = summarize_pr_curve(y, p, min_precision=0.95)

    # Precision can only ever be the base rate (0.2), so a 95% floor is
    # unreachable and recall at that floor is 0.
    assert summary.recall_at_min_precision == 0.0
    assert summary.pr_auc == pytest.approx(average_precision_score(y, p), abs=1e-12)


def test_no_positives_returns_zero_rather_than_nan() -> None:
    y = np.zeros(100, dtype=int)
    p = np.random.default_rng(1).random(100)
    summary = summarize_pr_curve(y, p)
    assert summary.pr_auc == 0.0
    assert summary.recall_at_min_precision == 0.0


def test_perfect_ranking_scores_one() -> None:
    y = np.array([0] * 90 + [1] * 10)
    p = np.array([0.1] * 90 + [0.9] * 10)
    summary = summarize_pr_curve(y, p, min_precision=0.95)
    assert summary.pr_auc == pytest.approx(1.0)
    assert summary.recall_at_min_precision == pytest.approx(1.0)


# --- Bootstrap ---


def test_bootstrap_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(3)
    n = 5_000
    y = rng.binomial(1, 0.08, n)
    p = np.clip(rng.beta(2, 5, n) + 0.3 * y, 0, 1)

    intervals = bootstrap_pr_summary(y, p, min_precision=0.5, n_resamples=200, seed=11)

    for key in ("pr_auc", "recall_at_min_precision"):
        ci = intervals[key]
        assert ci.low <= ci.point <= ci.high, f"{key}: {ci}"
        assert ci.high > ci.low, f"{key} interval collapsed: {ci}"
        assert ci.n_resamples == 200


def test_bootstrap_point_estimate_equals_the_plain_summary() -> None:
    """Resampling must not move the point estimate it is an interval around."""
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.1, 2_000)
    p = np.clip(rng.beta(2, 4, 2_000) + 0.2 * y, 0, 1)

    summary = summarize_pr_curve(y, p, min_precision=0.5)
    intervals = bootstrap_pr_summary(y, p, min_precision=0.5, n_resamples=50)

    assert intervals["pr_auc"].point == pytest.approx(summary.pr_auc)
    assert intervals["recall_at_min_precision"].point == pytest.approx(
        summary.recall_at_min_precision
    )


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(5)
    y = rng.binomial(1, 0.1, 1_000)
    p = rng.random(1_000)

    first = bootstrap_pr_summary(y, p, n_resamples=50, seed=7)
    second = bootstrap_pr_summary(y, p, n_resamples=50, seed=7)

    assert first["pr_auc"].low == second["pr_auc"].low
    assert first["pr_auc"].high == second["pr_auc"].high


def test_bootstrap_rejects_zero_resamples() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_pr_summary(np.array([0, 1]), np.array([0.1, 0.9]), n_resamples=0)


def test_more_data_narrows_the_interval() -> None:
    """The interval has to respond to sample size, or it is decorative."""
    rng = np.random.default_rng(6)

    def width(n: int) -> float:
        y = rng.binomial(1, 0.1, n)
        p = np.clip(rng.beta(2, 4, n) + 0.25 * y, 0, 1)
        ci = bootstrap_pr_summary(y, p, min_precision=0.5, n_resamples=200, seed=2)["pr_auc"]
        return ci.high - ci.low

    assert width(20_000) < width(1_000)


# --- Rule scorers ---


def test_amount_rule_ignores_types_that_never_contain_fraud() -> None:
    df = pd.DataFrame(
        {
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
            "amount": [100.0, 200.0, 999_999.0, 999_999.0, 999_999.0],
            "oldbalanceOrg": [100.0, 200.0, 999_999.0, 999_999.0, 999_999.0],
            "newbalanceOrig": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    scores = amount_rule_scores(df)
    assert scores.tolist() == [100.0, 200.0, 0.0, 0.0, 0.0]


def test_balance_signature_needs_all_three_conditions() -> None:
    df = pd.DataFrame(
        {
            "type": ["TRANSFER", "TRANSFER", "TRANSFER", "PAYMENT"],
            "amount": [500.0, 400.0, 500.0, 500.0],
            # row 0: drains exactly -> matches
            # row 1: partial debit -> no
            # row 2: right amount, balance not zeroed -> no
            # row 3: drains exactly but wrong type -> no
            "oldbalanceOrg": [500.0, 500.0, 500.0, 500.0],
            "newbalanceOrig": [0.0, 100.0, 50.0, 0.0],
        }
    )
    assert balance_signature_scores(df).tolist() == [1.0, 0.0, 0.0, 0.0]


# --- Three-way time split ---


def _stepped_frame(step_counts: dict[int, int]) -> pd.DataFrame:
    rows = [{"step": step, "isFraud": 0} for step, n in step_counts.items() for _ in range(n)]
    return pd.DataFrame(rows)


def test_three_way_split_is_ordered_and_complete(synthetic_paysim: pd.DataFrame) -> None:
    train, val, test = time_based_split_three_way(synthetic_paysim)

    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < test["step"].min()
    assert len(train) + len(val) + len(test) == len(synthetic_paysim)


def test_no_time_step_straddles_a_boundary(synthetic_paysim: pd.DataFrame) -> None:
    """The flaw the two-way split had: one hour on both sides of the cut."""
    train, val, test = time_based_split_three_way(synthetic_paysim)
    step_sets = [set(part["step"].unique()) for part in (train, val, test)]

    assert step_sets[0].isdisjoint(step_sets[1])
    assert step_sets[1].isdisjoint(step_sets[2])
    assert step_sets[0].isdisjoint(step_sets[2])


def test_boundaries_snap_to_the_nearest_step_under_skewed_volume() -> None:
    """Front-loaded volume, like the real dataset: row fractions bend to steps.

    Step 1 alone holds 60% of the rows, so the train split cannot be 60% of rows
    *and* end at a step boundary. It ends at the boundary.
    """
    df = _stepped_frame({1: 600, 2: 100, 3: 100, 4: 100, 5: 100})
    train, val, test = time_based_split_three_way(df, val_fraction=0.2, test_fraction=0.2)

    assert set(train["step"].unique()) == {1}
    assert train["step"].max() < val["step"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_rejects_fractions_that_leave_no_training_data() -> None:
    df = _stepped_frame({1: 10, 2: 10, 3: 10})
    with pytest.raises(ValueError, match="room for training"):
        time_based_split_three_way(df, val_fraction=0.5, test_fraction=0.5)


def test_rejects_a_missing_time_column() -> None:
    with pytest.raises(ValueError, match="Time column"):
        time_based_split_three_way(pd.DataFrame({"amount": [1.0, 2.0]}))


def test_rejects_too_few_time_steps_to_split() -> None:
    """Two steps cannot make three splits; fail loudly instead of silently empty."""
    with pytest.raises(ValueError, match="too few distinct values|empty"):
        time_based_split_three_way(_stepped_frame({1: 50, 2: 50}))
