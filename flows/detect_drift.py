"""Hourly drift detection flow.

Compares recent prediction distributions against the training baseline
using Evidently 0.7.x. Publishes SNS alert if drift detected.
Saves HTML report to S3 for audit history.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import boto3
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from prefect import flow, get_run_logger, task
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

load_dotenv()

SNS_TOPIC_ARN = "arn:aws:sns:ap-south-1:125773060711:fraud-mlops-drift-alerts"
S3_BUCKET = "fraud-mlops-kidiloski"
DRIFT_THRESHOLD = 0.3


@task
def load_reference_data(parquet_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    df = pd.read_parquet(parquet_path)
    ref = (
        df[["amount", "probability", "type"]].copy()
        if "probability" in df.columns
        else df[["amount", "type"]].copy()
    )
    # Sample for efficiency
    ref = df[["amount", "type"]].sample(n=min(5000, len(df)), random_state=42)
    logger.info("Reference data: %d rows", len(ref))
    return ref


@task
def load_recent_decisions(hours: int = 24) -> pd.DataFrame:
    logger = get_run_logger()
    conn = psycopg2.connect(
        host=os.environ["RDS_HOST"],
        port=int(os.environ.get("RDS_PORT", 5432)),
        dbname=os.environ.get("RDS_DB", "mlflow"),
        user=os.environ["RDS_USER"],
        password=os.environ["RDS_PASSWORD"],
        sslmode="require",
    )
    df = pd.read_sql(
        f"SELECT amount, type, probability FROM decisions "
        f"WHERE scored_at > NOW() - INTERVAL '{hours} hours'",
        conn,
    )
    conn.close()
    logger.info("Recent decisions: %d rows (last %d hours)", len(df), hours)
    return df


@task
def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    logger = get_run_logger()

    # Use columns present in both
    cols = ["amount", "type"]
    ref = reference[cols].copy()
    cur = current[cols].copy()

    definition = DataDefinition(
        numerical_columns=["amount"],
        categorical_columns=["type"],
    )

    ref_dataset = Dataset.from_pandas(ref, data_definition=definition)
    cur_dataset = Dataset.from_pandas(cur, data_definition=definition)

    report = Report([DataDriftPreset()])
    result = report.run(ref_dataset, cur_dataset)
    result_dict = result.dict()
    logger.info("Raw result keys: %s", list(result_dict.keys()))
    logger.info(
        "First metric: %s", json.dumps(result_dict.get("metrics", [{}])[0], indent=2, default=str)
    )

    # Extract drift metrics
    metrics = {"dataset_drift_detected": False, "drift_share": 0.0, "n_drifted_columns": 0}
    for metric in result_dict.get("metrics", []):
        value = metric.get("value", {})
        metric_name = metric.get("metric_name", "")
        if "DriftedColumnsCount" in metric_name:
            metrics["n_drifted_columns"] = int(value.get("count", 0))
            metrics["drift_share"] = float(value.get("share", 0.0))
            metrics["dataset_drift_detected"] = float(value.get("share", 0.0)) > DRIFT_THRESHOLD

    logger.info("Drift metrics: %s", json.dumps(metrics, indent=2))

    # Save HTML to S3
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    local_path = f"/tmp/drift_report_{timestamp}.html"
    result.save_html(local_path)

    s3 = boto3.client("s3")
    s3_key = f"drift-reports/drift_report_{timestamp}.html"
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    report_url = f"s3://{S3_BUCKET}/{s3_key}"
    logger.info("Report saved to %s", report_url)

    metrics["report_url"] = report_url
    return metrics


@task
def send_alert_if_drift(metrics: dict) -> bool:
    logger = get_run_logger()

    if not metrics.get("dataset_drift_detected", False):
        logger.info("No drift detected. drift_share=%.3f", metrics.get("drift_share", 0))
        return False

    logger.warning("DRIFT DETECTED. drift_share=%.3f", metrics.get("drift_share", 0))

    message = f"""
🚨 Fraud Detector Drift Alert

Drift detected in production predictions.

Metrics:
- Drifted columns: {metrics.get('n_drifted_columns', 0)}
- Drift share: {metrics.get('drift_share', 0):.1%}

Full report: {metrics.get('report_url', 'N/A')}
Time: {datetime.utcnow().isoformat()}
    """.strip()

    sns = boto3.client("sns", region_name="ap-south-1")
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="🚨 Fraud Detector: Data Drift Detected",
        Message=message,
    )
    logger.info("SNS alert sent.")
    return True


@task
def push_metrics_to_prometheus(metrics: dict) -> None:
    """Push drift metrics to Prometheus Pushgateway."""
    logger = get_run_logger()

    registry = CollectorRegistry()

    drift_share = Gauge(
        "fraud_detector_drift_share",
        "Share of drifted columns vs training baseline",
        registry=registry,
    )
    drift_share.set(metrics.get("drift_share", 0.0))

    drifted_columns = Gauge(
        "fraud_detector_drifted_columns_count", "Number of drifted columns", registry=registry
    )
    drifted_columns.set(metrics.get("n_drifted_columns", 0))

    drift_detected = Gauge(
        "fraud_detector_drift_detected",
        "1 if dataset drift detected, 0 otherwise",
        registry=registry,
    )
    drift_detected.set(1 if metrics.get("dataset_drift_detected") else 0)

    pushgateway_url = os.environ.get("PUSHGATEWAY_URL", "localhost:9091")

    try:
        push_to_gateway(pushgateway_url, job="fraud-drift-detection", registry=registry)
        logger.info("Metrics pushed to Prometheus Pushgateway at %s", pushgateway_url)
    except Exception as e:
        logger.warning("Failed to push metrics to Pushgateway: %s", e)


@flow(name="drift-detection", log_prints=True)
def detect_drift_flow(
    parquet_path: str = "data/interim/paysim_with_features.parquet",
    lookback_hours: int = 24,
) -> None:
    logger = get_run_logger()
    logger.info("Starting drift detection flow")

    reference = load_reference_data(parquet_path)
    current = load_recent_decisions(hours=lookback_hours)

    if len(current) < 10:
        logger.warning("Only %d recent decisions. Skipping.", len(current))
        return

    metrics = run_drift_report(reference=reference, current=current)
    send_alert_if_drift(metrics=metrics)
    push_metrics_to_prometheus(metrics=metrics)
    logger.info("Drift detection complete.")


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent
    detect_drift_flow(
        parquet_path=str(PROJECT_ROOT / "data/interim/paysim_with_features.parquet"),
    )
