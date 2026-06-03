"""Lambda handler for streaming fraud inference with Feast feature lookup.

Triggered by Kinesis Data Stream. For each transaction:
1. Parse raw transaction fields from the Kinesis record
2. Look up sender's historical features from Feast online store (Redis)
3. Compute derived features inline
4. Score with the production fraud-detector model
5. Write decision to RDS decisions table
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

import psycopg2
from dotenv import load_dotenv
from feast import FeatureStore

from fraud_mlops.inference import load_production_detector

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Cold start: load model and Feast store once per container ────────────────
logger.info("Cold start: loading production fraud detector...")
DETECTOR = load_production_detector()
logger.info("Cold start complete: model version %s loaded", DETECTOR.version)

FEATURE_STORE = FeatureStore(repo_path="/var/task/feature_repo")
FEATURE_STORE.refresh_registry()
logger.info("Feast feature store initialized")

FEAST_FEATURES = [
    "sender_transaction_features:sender_txn_count_1h",
    "sender_transaction_features:sender_txn_count_24h",
    "sender_transaction_features:sender_amount_sum_24h",
    "sender_transaction_features:sender_amount_mean_historical",
    "sender_transaction_features:sender_time_since_last_txn",
]

# Default feature values for unknown senders (first-time senders)
DEFAULT_FEATURES = {
    "sender_txn_count_1h": 0,
    "sender_txn_count_24h": 0,
    "sender_amount_sum_24h": 0.0,
    "sender_amount_mean_historical": 0.0,
    "sender_time_since_last_txn": -1.0,
}


def _get_sender_features(sender_id: str) -> dict:
    """Look up sender's historical features from Feast online store."""
    try:
        result = FEATURE_STORE.get_online_features(
            features=FEAST_FEATURES,
            entity_rows=[{"sender": sender_id}],
        ).to_dict()

        features = {}
        for key, values in result.items():
            if key == "sender":
                continue
            val = values[0]
            features[key] = val if val is not None else DEFAULT_FEATURES.get(key, 0)

        return features

    except Exception as e:
        logger.warning("Feast lookup failed for sender %s: %s. Using defaults.", sender_id, e)
        return DEFAULT_FEATURES.copy()


def _build_transaction_row(txn: dict, sender_features: dict) -> dict:
    """Combine raw transaction fields with Feast features into model input."""
    amount = txn["amount"]
    old_balance = txn.get("oldbalanceOrg", 0.0)

    return {
        "step": txn.get("step", 0),
        "type": txn.get("type", "TRANSFER"),
        "amount": amount,
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": txn.get("newbalanceOrig", 0.0),
        "oldbalanceDest": txn.get("oldbalanceDest", 0.0),
        "newbalanceDest": txn.get("newbalanceDest", 0.0),
        # Feast-served features
        "sender_txn_count_1h": sender_features.get("sender_txn_count_1h", 0),
        "sender_txn_count_24h": sender_features.get("sender_txn_count_24h", 0),
        "sender_amount_sum_24h": sender_features.get("sender_amount_sum_24h", 0.0),
        "sender_amount_mean_historical": sender_features.get("sender_amount_mean_historical", 0.0),
        "sender_time_since_last_txn": sender_features.get("sender_time_since_last_txn", -1.0),
        # Inline-computed features
        "amount_to_oldbalance_ratio": amount / old_balance if old_balance > 0 else -1.0,
        "drains_origin": 1 if (old_balance > 0 and txn.get("newbalanceOrig", 1) == 0) else 0,
    }


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

                # Look up historical features from Feast
                sender_id = txn.get("sender_id", "unknown")
                sender_features = _get_sender_features(sender_id)

                # Build full feature row
                row = _build_transaction_row(txn, sender_features)

                # Score
                predictions = DETECTOR.predict_batch([row])
                prediction = predictions[0]

                _write_decision(conn, txn, prediction)

                logger.info(
                    "Scored transaction %s sender=%s feast_hit=%s " "is_fraud=%s probability=%.4f",
                    txn.get("transaction_id", "unknown"),
                    sender_id,
                    any(v != DEFAULT_FEATURES.get(k) for k, v in sender_features.items()),
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
