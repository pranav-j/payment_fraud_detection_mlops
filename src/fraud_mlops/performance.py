"""Realized model performance, against labels that have actually arrived.

Drift detection answers "is the input data still what the model was trained on?"
It cannot answer "is the model still right?", because being right is a statement
about outcomes, and outcomes show up late. A fraud label is a chargeback or a
dispute: the transaction settles, the model's decision stands, and the truth
arrives days or weeks later.

That delay is the whole problem this module exists for. Two mistakes are easy
and both are worse than having no performance monitoring at all:

1. **Counting unmatured decisions as correct.** Join `decisions` to `labels` and
   treat a missing label as "not fraud", and recall looks perfect on the most
   recent window — exactly the window an operator is looking at.
2. **Alerting on an empty window.** Treat "no labels yet" as "recall is zero"
   and the alert fires every night at the same time for reasons that have
   nothing to do with the model.

So: a window's metrics are computed from matured labels only, a window reports
how much of itself has matured, and precision/recall are `None` — not zero —
when there is nothing to measure. Alerts require a minimum number of matured
labels before they can fire.

The logic here is deliberately pure: dataframes in, dataclasses out, no database
and no network, so the window boundaries and the alert rule can be tested
directly.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# How far below the floor realized recall has to sit before anyone is woken up.
DEFAULT_RECALL_FLOOR = 0.70

# A window with fewer matured labels than this cannot produce a trustworthy
# recall, so it never alerts. With PaySim's fraud rate, a day of streamed
# traffic may contain only a handful of positives.
DEFAULT_MIN_MATURED_LABELS = 50

# Minimum matured *fraud* cases before recall is considered meaningful. Recall
# over three positives is noise with a decimal point on it.
DEFAULT_MIN_MATURED_POSITIVES = 10


@dataclass(frozen=True)
class WindowMetrics:
    """Realized performance over one time window.

    `precision` and `recall` are `None` when the window has no matured labels to
    compute them from, or no positives in the case of recall. A consumer must
    handle `None`; that is the point of using it rather than zero.
    """

    window_start: str
    window_end: str
    n_decisions: int
    n_matured: int
    maturity_share: float
    n_matured_positives: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float | None
    recall: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlertDecision:
    """Whether realized recall has dropped far enough to notify someone."""

    should_alert: bool
    reason: str
    recall_floor: float
    window: WindowMetrics | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["window"] = self.window.to_dict() if self.window else None
        return payload


def join_matured_labels(
    decisions: pd.DataFrame,
    labels: pd.DataFrame,
    as_of: datetime,
) -> pd.DataFrame:
    """Attach labels to decisions, keeping only labels that have matured.

    A label has matured when `label_available_at <= as_of`. Anything later is
    dropped rather than joined as null, so a caller cannot accidentally count it.

    Args:
        decisions: Needs `transaction_id`, `is_fraud` (the prediction) and
            `scored_at`.
        labels: Needs `transaction_id`, `is_fraud` (the outcome) and
            `label_available_at`.
        as_of: The clock. Passed in rather than read from `now()` so the
            behaviour is testable and so a backfill can ask "what did we know on
            Tuesday?".

    Returns:
        One row per decision that has a matured label, with `predicted_fraud`
        and `actual_fraud` columns.
    """
    required_decisions = {"transaction_id", "is_fraud", "scored_at"}
    required_labels = {"transaction_id", "is_fraud", "label_available_at"}
    if missing := required_decisions - set(decisions.columns):
        raise ValueError(f"decisions is missing columns: {sorted(missing)}")
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"labels is missing columns: {sorted(missing)}")

    if labels.empty or decisions.empty:
        return pd.DataFrame(
            columns=["transaction_id", "scored_at", "predicted_fraud", "actual_fraud"]
        )

    matured = labels[pd.to_datetime(labels["label_available_at"], utc=True) <= _utc(as_of)]

    joined = decisions.merge(
        matured[["transaction_id", "is_fraud"]].rename(columns={"is_fraud": "actual_fraud"}),
        on="transaction_id",
        how="inner",
        suffixes=("", "_label"),
    )
    joined = joined.rename(columns={"is_fraud": "predicted_fraud"})
    joined["predicted_fraud"] = joined["predicted_fraud"].astype(bool)
    joined["actual_fraud"] = joined["actual_fraud"].astype(bool)

    logger.info(
        "Matured join: %d decisions, %d labels (%d matured as of %s) -> %d joined rows",
        len(decisions),
        len(labels),
        len(matured),
        as_of.isoformat(),
        len(joined),
    )
    return joined


def _utc(value: datetime) -> pd.Timestamp:
    """Normalize to a tz-aware UTC timestamp so comparisons never silently fail."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def compute_window_metrics(
    decisions: pd.DataFrame,
    joined: pd.DataFrame,
    window_hours: int = 24,
    as_of: datetime | None = None,
    lookback_windows: int = 14,
) -> list[WindowMetrics]:
    """Bucket decisions into fixed windows and score each against matured labels.

    Windows are aligned to the epoch rather than to `as_of`, so the same
    transaction lands in the same bucket no matter when the flow happens to run.
    That is what makes two consecutive daily runs comparable.

    Args:
        decisions: All decisions in the period of interest — including those
            whose labels have not matured, because that is what
            `maturity_share` is measured against.
        joined: Output of `join_matured_labels`.
        window_hours: Window width.
        as_of: End of the reporting period (defaults to the latest decision).
        lookback_windows: How many windows back to report.

    Returns:
        One `WindowMetrics` per window, oldest first.
    """
    if decisions.empty:
        return []

    scored_at = pd.to_datetime(decisions["scored_at"], utc=True)
    end = _utc(as_of) if as_of is not None else scored_at.max()
    window = pd.Timedelta(hours=window_hours)
    start = end.floor(window) - (lookback_windows - 1) * window

    buckets = scored_at.dt.floor(window)
    in_range = buckets >= start

    joined_buckets = (
        pd.to_datetime(joined["scored_at"], utc=True).dt.floor(window)
        if not joined.empty
        else pd.Series(dtype="datetime64[ns, UTC]")
    )

    results: list[WindowMetrics] = []
    for bucket in sorted(buckets[in_range].unique()):
        n_decisions = int((buckets == bucket).sum())
        rows = joined[joined_buckets == bucket] if not joined.empty else joined

        if len(rows) == 0:
            tp = fp = fn = tn = 0
        else:
            predicted = rows["predicted_fraud"].to_numpy(dtype=bool)
            actual = rows["actual_fraud"].to_numpy(dtype=bool)
            tp = int((predicted & actual).sum())
            fp = int((predicted & ~actual).sum())
            fn = int((~predicted & actual).sum())
            tn = int((~predicted & ~actual).sum())

        n_matured = tp + fp + fn + tn
        n_positives = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / n_positives if n_positives > 0 else None

        results.append(
            WindowMetrics(
                window_start=pd.Timestamp(bucket).isoformat(),
                window_end=(pd.Timestamp(bucket) + window).isoformat(),
                n_decisions=n_decisions,
                n_matured=n_matured,
                maturity_share=(n_matured / n_decisions) if n_decisions else 0.0,
                n_matured_positives=n_positives,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
                precision=precision,
                recall=recall,
            )
        )

    return results


def latest_evaluable_window(
    metrics: list[WindowMetrics],
    min_matured_labels: int = DEFAULT_MIN_MATURED_LABELS,
    min_matured_positives: int = DEFAULT_MIN_MATURED_POSITIVES,
) -> WindowMetrics | None:
    """The most recent window with enough matured labels to mean something.

    Not simply the last window: the last window is usually the one whose labels
    have not arrived yet. Walking backwards to the newest window that clears the
    minimums is the difference between measuring the model and measuring the
    label delay.
    """
    for window in reversed(metrics):
        if (
            window.n_matured >= min_matured_labels
            and window.n_matured_positives >= min_matured_positives
            and window.recall is not None
        ):
            return window
    return None


def evaluate_alert(
    metrics: list[WindowMetrics],
    recall_floor: float = DEFAULT_RECALL_FLOOR,
    min_matured_labels: int = DEFAULT_MIN_MATURED_LABELS,
    min_matured_positives: int = DEFAULT_MIN_MATURED_POSITIVES,
) -> AlertDecision:
    """Decide whether realized recall warrants an alert.

    Fires only when a window has enough matured labels *and* its realized recall
    is below the floor. "No data yet" is never an alert — it is the expected
    state of the newest window, every single day.
    """
    window = latest_evaluable_window(metrics, min_matured_labels, min_matured_positives)

    if window is None:
        return AlertDecision(
            should_alert=False,
            reason=(
                f"No window has at least {min_matured_labels} matured labels and "
                f"{min_matured_positives} matured fraud cases yet. Nothing to judge — "
                f"this is the normal state while labels are still in flight."
            ),
            recall_floor=recall_floor,
            window=None,
        )

    assert window.recall is not None  # guaranteed by latest_evaluable_window
    if window.recall < recall_floor:
        return AlertDecision(
            should_alert=True,
            reason=(
                f"Realized recall {window.recall:.1%} is below the floor "
                f"{recall_floor:.1%} for the window starting {window.window_start} "
                f"({window.true_positives} caught of {window.n_matured_positives} "
                f"matured fraud cases)."
            ),
            recall_floor=recall_floor,
            window=window,
        )

    return AlertDecision(
        should_alert=False,
        reason=(
            f"Realized recall {window.recall:.1%} holds the floor {recall_floor:.1%} "
            f"for the window starting {window.window_start}."
        ),
        recall_floor=recall_floor,
        window=window,
    )


def label_delay_to_wall_clock(delay_hours: float) -> timedelta:
    """Interpret `LABEL_DELAY_HOURS` as wall-clock hours.

    Worth stating explicitly, because this project has two clocks:

    - **PaySim's clock.** The `step` column is a simulated hour; the dataset's
      743 steps are a simulated month. Steps never advance in real time — they
      are a column in a static file.
    - **The replay's clock.** The producer sends transactions now, the Lambda
      decides now, and `scored_at` is a real timestamp.

    Label maturity runs on the second clock. `LABEL_DELAY_HOURS` is therefore
    real hours after the decision, not PaySim steps — so a demo sets it to a
    fraction of an hour (0.05 = three minutes) to watch a day's worth of label
    delay compress into a coffee break, while production would set it to the
    real chargeback window (720 = 30 days).
    """
    return timedelta(hours=delay_hours)
