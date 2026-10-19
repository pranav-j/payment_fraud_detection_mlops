"""Daily realized-performance flow.

The hourly drift flow watches the *inputs*. This one watches whether the model
was right, which is a different question and can only be answered once outcome
labels arrive.

    decisions  JOIN  labels (matured only)
      -> realized precision and recall per window
      -> Prometheus
      -> SNS when realized recall drops below the floor

Runs daily rather than hourly because labels arrive on the order of days: an
hourly run would mostly re-report the same matured windows.

Everything that decides anything lives in `fraud_mlops.performance`, as pure
functions over dataframes. This module is the plumbing: fetch, push, notify.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import boto3
import pandas as pd
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from fraud_mlops.db import connect
from fraud_mlops.performance import (
    DEFAULT_MIN_MATURED_LABELS,
    DEFAULT_MIN_MATURED_POSITIVES,
    DEFAULT_RECALL_FLOOR,
    AlertDecision,
    WindowMetrics,
    compute_window_metrics,
    evaluate_alert,
    join_matured_labels,
)

load_dotenv()

SNS_TOPIC_ARN = os.environ.get(
    "SNS_TOPIC_ARN",
    f"arn:aws:sns:{os.environ.get('AWS_DEFAULT_REGION', 'ap-south-1')}:"
    f"{os.environ.get('AWS_ACCOUNT_ID', '')}:fraud-mlops-drift-alerts",
)

DECISIONS_SQL = """
SELECT transaction_id, is_fraud, probability, threshold, model_version, scored_at
FROM decisions
WHERE scored_at > NOW() - make_interval(days => %(lookback_days)s)
"""

# Every label for those decisions, matured or not — the maturity filter is
# applied in Python against an explicit `as_of`, so the same query can answer
# "what did we know last Tuesday?" during a backfill.
LABELS_SQL = """
SELECT transaction_id, is_fraud, event_time, label_available_at
FROM labels
WHERE event_time > NOW() - make_interval(days => %(lookback_days)s)
"""


@task(retries=2, retry_delay_seconds=30)
def fetch_decisions_and_labels(lookback_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull both tables for the lookback period."""
    logger = get_run_logger()
    conn = connect()
    try:
        params = {"lookback_days": lookback_days}
        decisions = pd.read_sql(DECISIONS_SQL, conn, params=params)
        labels = pd.read_sql(LABELS_SQL, conn, params=params)
    finally:
        conn.close()

    logger.info(
        "Fetched %d decisions and %d labels over the last %d days",
        len(decisions),
        len(labels),
        lookback_days,
    )
    return decisions, labels


@task
def compute_metrics(
    decisions: pd.DataFrame,
    labels: pd.DataFrame,
    window_hours: int,
    lookback_windows: int,
) -> list[WindowMetrics]:
    logger = get_run_logger()
    as_of = datetime.now(UTC)

    joined = join_matured_labels(decisions, labels, as_of=as_of)
    windows = compute_window_metrics(
        decisions,
        joined,
        window_hours=window_hours,
        as_of=as_of,
        lookback_windows=lookback_windows,
    )

    for window in windows:
        logger.info(
            "%s: %d decisions, %d matured (%.0f%%), precision=%s recall=%s",
            window.window_start,
            window.n_decisions,
            window.n_matured,
            window.maturity_share * 100,
            f"{window.precision:.3f}" if window.precision is not None else "n/a",
            f"{window.recall:.3f}" if window.recall is not None else "n/a",
        )
    return windows


@task
def push_metrics_to_prometheus(windows: list[WindowMetrics]) -> None:
    """Push the most recent *evaluable* window to the Pushgateway.

    Gauges carry the latest judgeable values, plus the maturity share of the
    newest window so a dashboard can show how much of "today" is still unknown
    rather than implying its recall is zero.
    """
    logger = get_run_logger()
    if not windows:
        logger.warning("No windows to push.")
        return

    from fraud_mlops.performance import latest_evaluable_window

    registry = CollectorRegistry()
    evaluable = latest_evaluable_window(windows)
    newest = windows[-1]

    gauges = {
        "fraud_detector_realized_precision": (
            "Realized precision over the most recent window with matured labels",
            evaluable.precision if evaluable and evaluable.precision is not None else None,
        ),
        "fraud_detector_realized_recall": (
            "Realized recall over the most recent window with matured labels",
            evaluable.recall if evaluable else None,
        ),
        "fraud_detector_matured_labels": (
            "Matured labels in the window the metrics above describe",
            float(evaluable.n_matured) if evaluable else 0.0,
        ),
        "fraud_detector_matured_positives": (
            "Matured fraud cases in that window",
            float(evaluable.n_matured_positives) if evaluable else 0.0,
        ),
        "fraud_detector_label_maturity_share": (
            "Share of the newest window's decisions whose labels have arrived",
            newest.maturity_share,
        ),
        "fraud_detector_pending_labels": (
            "Decisions in the newest window still waiting for a label",
            float(newest.n_decisions - newest.n_matured),
        ),
    }

    for name, (description, value) in gauges.items():
        if value is None:
            # Absent, not zero. A gauge of 0 would be read as "recall is zero".
            logger.info("Skipping %s — nothing evaluable yet", name)
            continue
        Gauge(name, description, registry=registry).set(value)

    pushgateway_url = os.environ.get("PUSHGATEWAY_URL", "localhost:9091")
    try:
        push_to_gateway(pushgateway_url, job="fraud-performance-monitoring", registry=registry)
        logger.info("Metrics pushed to Pushgateway at %s", pushgateway_url)
    except Exception as e:
        logger.warning("Failed to push metrics to Pushgateway: %s", e)


@task
def send_alert_if_degraded(decision: AlertDecision, alerts_enabled: bool = True) -> bool:
    """Publish an SNS alert when realized recall has genuinely dropped.

    `alerts_enabled=False` computes and logs the alert without sending it, which
    is how the flow is exercised against a local Postgres without needing an SNS
    topic or emailing anyone.
    """
    logger = get_run_logger()

    if not decision.should_alert:
        logger.info("No performance alert: %s", decision.reason)
        return False

    if not alerts_enabled:
        logger.warning("ALERT (not sent, alerts_enabled=False): %s", decision.reason)
        return False

    window = decision.window
    assert window is not None  # should_alert implies a window

    logger.warning("PERFORMANCE DEGRADED: %s", decision.reason)

    message = f"""
Fraud Detector Performance Alert

Realized recall has fallen below the configured floor. This is measured against
outcome labels that have actually arrived, not against input drift.

Window:            {window.window_start} -> {window.window_end}
Realized recall:   {window.recall:.1%}  (floor {decision.recall_floor:.1%})
Realized precision: {window.precision:.1%}
Caught:            {window.true_positives} of {window.n_matured_positives} matured fraud cases
False alarms:      {window.false_positives}
Labels matured:    {window.n_matured} of {window.n_decisions} decisions in the window

Note the label delay: this window is being judged now because its labels have
matured, not because the degradation started today. Check the realized
performance panel in Grafana for when the decline began.

Time: {datetime.now(UTC).isoformat()}
""".strip()

    sns = boto3.client("sns", region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"))
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="Fraud Detector: realized recall below floor",
        Message=message,
    )
    logger.info("SNS alert sent.")
    return True


@flow(name="performance-monitoring", log_prints=True)
def monitor_performance_flow(
    lookback_days: int = 14,
    window_hours: int = 24,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
    min_matured_labels: int = DEFAULT_MIN_MATURED_LABELS,
    min_matured_positives: int = DEFAULT_MIN_MATURED_POSITIVES,
    alerts_enabled: bool = True,
) -> None:
    """Daily: measure realized precision and recall, publish, alert if degraded."""
    logger = get_run_logger()
    logger.info("Starting performance monitoring flow")

    decisions, labels = fetch_decisions_and_labels(lookback_days=lookback_days)

    if decisions.empty:
        logger.warning("No decisions in the lookback window. Nothing to measure.")
        return

    windows = compute_metrics(
        decisions=decisions,
        labels=labels,
        window_hours=window_hours,
        lookback_windows=max(1, (lookback_days * 24) // window_hours),
    )

    decision = evaluate_alert(
        windows,
        recall_floor=recall_floor,
        min_matured_labels=min_matured_labels,
        min_matured_positives=min_matured_positives,
    )
    logger.info("Alert evaluation: %s", json.dumps(decision.to_dict(), indent=2, default=str))

    push_metrics_to_prometheus(windows=windows)
    send_alert_if_degraded(decision=decision, alerts_enabled=alerts_enabled)

    logger.info("Performance monitoring complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the performance monitoring flow")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument(
        "--recall-floor",
        type=float,
        default=float(os.environ.get("PERFORMANCE_RECALL_FLOOR", DEFAULT_RECALL_FLOOR)),
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Compute and log the alert without publishing to SNS",
    )
    cli_args = parser.parse_args()

    monitor_performance_flow(
        lookback_days=cli_args.lookback_days,
        window_hours=cli_args.window_hours,
        recall_floor=cli_args.recall_floor,
        alerts_enabled=not cli_args.no_alerts,
    )
