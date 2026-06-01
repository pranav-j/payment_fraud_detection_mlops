"""Lambda handler for streaming fraud inference.

Triggered by Kinesis Data Stream. Each invocation receives a batch of
transaction records. Scores each transaction using the production
fraud-detector model and writes decisions to RDS.

Execution model:
- Cold start: module imports + model load from S3 via MLflow (~5-15s)
- Warm invocations: handler() called directly, model already in memory
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

import psycopg2
from dotenv import load_dotenv

from fraud_mlops.inference import load_production_detector

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info("Cold start: loading production fraud detector...")
DETECTOR = load_production_detector()
logger.info("Cold start complete: model version %s loaded", DETECTOR.version)


def _get_db_conn():
    return psycopg2.connect(
        host=os.environ["RDS_HOST"],
        port=int(os.environ.get("RDS_PORT", 5432)),
        dbname=os.environ.get("RDS_DB", "mlflow"),
        user=os.environ["RDS_USER"],
        password=os.environ["RDS_PASSWORD"],
        sslmode="require",
    )


def _write_decision(conn, txn: dict, prediction) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO decisions
                (transaction_id, step, type, amount,
                 is_fraud, probability, threshold, model_version, feature_set)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                txn.get("transaction_id", "unknown"),
                txn.get("step"),
                txn.get("type"),
                txn.get("amount"),
                prediction.is_fraud,
                prediction.probability,
                prediction.threshold,
                prediction.model_version,
                prediction.feature_set,
            ),
        )
    conn.commit()


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    records = event.get("Records", [])
    logger.info("Received batch of %d records", len(records))

    succeeded = 0
    failed = 0
    batch_failures = []

    conn = _get_db_conn()

    try:
        for record in records:
            sequence_number = record["kinesis"]["sequenceNumber"]
            try:
                raw = base64.b64decode(record["kinesis"]["data"]).decode("utf-8")
                txn = json.loads(raw)

                predictions = DETECTOR.predict_batch([txn])
                prediction = predictions[0]

                _write_decision(conn, txn, prediction)

                logger.info(
                    "Scored transaction %s: is_fraud=%s probability=%.4f",
                    txn.get("transaction_id", "unknown"),
                    prediction.is_fraud,
                    prediction.probability,
                )
                succeeded += 1

            except Exception as e:
                logger.exception("Failed to process record %s: %s", sequence_number, e)
                failed += 1
                batch_failures.append({"itemIdentifier": sequence_number})

    finally:
        conn.close()

    logger.info("Batch complete: %d succeeded, %d failed", succeeded, failed)
    return {"batchItemFailures": batch_failures}
