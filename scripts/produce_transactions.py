#!/usr/bin/env python3
"""Synthetic transaction producer for fraud-mlops.

Pumps PaySim-style transactions into the Kinesis stream at a configurable rate.
Supports controlled fraud injection for demos.

Two things here exist for the performance-monitoring story:

**Ground truth.** `--record-ground-truth` writes each transaction's true label
into the `ground_truth` table as it is sent. Without it there is nothing for
`scripts/replay_labels.py` to release later, and realized precision and recall
cannot be computed at all. The label never travels to the scorer — it goes
straight to the database, which is the honest arrangement: the simulator knows,
the model does not.

**Sender IDs.** `--sender-pool` draws senders from a small pool instead of the
whole 10-digit space. The default behaviour draws at random from ~9 billion
possibilities while only ~9,300 repeat senders are materialised into Redis, so
the Feast lookup misses essentially every time and every scored transaction
falls back to default features. A demo of a feature store should hit the
feature store.

Usage:
    python scripts/produce_transactions.py
    python scripts/produce_transactions.py --rate 5 --fraud-rate 0.1
    python scripts/produce_transactions.py --count 100 --record-ground-truth
    python scripts/produce_transactions.py --fraud-profile shifted  # see demo_performance_drop.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

STREAM_NAME = "fraud-mlops-transactions"
REGION = "ap-south-1"

# Realistic PaySim feature distributions (from EDA in Week 1)
TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]  # fraud only occurs in these types

logger = logging.getLogger("producer")


def _sender_id(pool: list[str] | None) -> str:
    """A sender ID, from the pool if one was supplied."""
    if pool:
        return random.choice(pool)
    return f"C{random.randint(1000000000, 9999999999)}"


def generate_normal_transaction(step: int, pool: list[str] | None = None) -> dict:
    txn_type = random.choice(TRANSACTION_TYPES)
    amount = random.lognormvariate(7.0, 1.5)
    old_balance = random.lognormvariate(8.0, 2.0)
    new_balance = max(0.0, old_balance - amount)

    return {
        "transaction_id": str(uuid.uuid4()),
        "sender_id": _sender_id(pool),
        "step": step,
        "type": txn_type,
        "amount": round(amount, 2),
        "oldbalanceOrg": round(old_balance, 2),
        "newbalanceOrig": round(new_balance, 2),
        "oldbalanceDest": round(random.lognormvariate(7.0, 2.0), 2),
        "newbalanceDest": round(random.lognormvariate(7.0, 2.0), 2),
    }


def generate_fraud_transaction(step: int, pool: list[str] | None = None) -> dict:
    """Fraud with PaySim's own signature: the origin account is emptied."""
    old_balance = round(random.lognormvariate(8.5, 1.5), 2)
    txn_type = random.choice(FRAUD_TYPES)

    return {
        "transaction_id": str(uuid.uuid4()),
        "sender_id": _sender_id(pool),
        "step": step,
        "type": txn_type,
        "amount": old_balance,
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }


def generate_shifted_fraud_transaction(step: int, pool: list[str] | None = None) -> dict:
    """Fraud that does *not* look like PaySim's fraud.

    The model's signal is overwhelmingly "the origin account was emptied" (see
    the leakage section of reports/metrics/evaluation.md). So a fraud pattern
    that takes many small amounts and leaves a balance behind is invisible to
    it — while the input distribution barely moves, because small
    TRANSFER/CASH_OUT amounts are exactly what normal traffic looks like.

    That is the whole point of the demo: data-drift monitoring stays quiet and
    realized recall collapses. Only outcome labels can see this, and only after
    the label delay has elapsed.
    """
    old_balance = round(random.lognormvariate(8.5, 1.5), 2)
    amount = round(old_balance * random.uniform(0.02, 0.15), 2)
    txn_type = random.choice(FRAUD_TYPES)

    return {
        "transaction_id": str(uuid.uuid4()),
        "sender_id": _sender_id(pool),
        "step": step,
        "type": txn_type,
        "amount": amount,
        "oldbalanceOrg": old_balance,
        # Balance left behind, so `drains_origin` stays 0 and the
        # amount/balance ratio looks ordinary.
        "newbalanceOrig": round(old_balance - amount, 2),
        "oldbalanceDest": round(random.lognormvariate(7.0, 2.0), 2),
        "newbalanceDest": round(random.lognormvariate(7.0, 2.0) + amount, 2),
    }


FRAUD_PROFILES = {
    "paysim": generate_fraud_transaction,
    "shifted": generate_shifted_fraud_transaction,
}


def record_ground_truth(conn, transaction_id: str, is_fraud: bool, source: str) -> None:
    """Write the true outcome to `ground_truth`, never to the stream.

    `source` carries which generator produced the row ("producer:paysim",
    "producer:shifted"). Without it, attributing a decision to a demo phase
    means guessing from timestamps — which is exactly how the first live run of
    the demo was misread: a 20-minute query window swept in an unrelated
    producer run from 18 minutes earlier, and the two phases came out mixed.

    This is what `scripts/replay_labels.py` later releases into `labels` once
    the configured delay has elapsed.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ground_truth (transaction_id, is_fraud, source)
            VALUES (%s, %s, %s)
            ON CONFLICT (transaction_id) DO NOTHING
            """,
            (transaction_id, is_fraud, source),
        )
    conn.commit()


def load_sender_pool(path: Path | None, size: int) -> list[str] | None:
    """Sender IDs that Feast actually has features for.

    Reads the repeat-sender parquet the Feast offline store is built from, so
    lookups hit. Returns None if it isn't available, in which case senders are
    drawn at random and every lookup misses — which is what the producer did
    before, and why `feast_hit` was always false in the Lambda logs.
    """
    import pandas as pd

    path = (
        path or Path(__file__).resolve().parents[1] / "data/interim/paysim_repeat_senders.parquet"
    )
    if not path.exists():
        logger.warning(
            "No sender pool at %s — senders will be random and every Feast lookup "
            "will miss. Pass --sender-pool to point at the repeat-sender parquet.",
            path,
        )
        return None

    df = pd.read_parquet(path, columns=["sender"])
    pool = df["sender"].drop_duplicates().head(size).tolist()
    logger.info("Loaded %d senders from %s", len(pool), path)
    return pool


def produce(
    rate: float,
    fraud_rate: float,
    count: int | None,
    fraud_profile: str = "paysim",
    sender_pool: list[str] | None = None,
    record_truth: bool = False,
) -> None:
    client = boto3.client("kinesis", region_name=REGION)
    interval = 1.0 / rate
    step = 1
    sent = 0
    frauds = 0

    make_fraud = FRAUD_PROFILES[fraud_profile]

    conn = None
    if record_truth:
        from fraud_mlops.db import connect

        conn = connect()

    print(
        f"Producing to stream '{STREAM_NAME}' at {rate} txn/sec "
        f"(fraud rate: {fraud_rate:.0%}, profile: {fraud_profile}"
        f"{', recording ground truth' if record_truth else ''})"
    )
    if sender_pool:
        print(f"Drawing senders from a pool of {len(sender_pool)} with Feast features")
    print("Ctrl-C to stop\n")

    try:
        while count is None or sent < count:
            is_fraud = random.random() < fraud_rate
            txn = (
                make_fraud(step, sender_pool)
                if is_fraud
                else generate_normal_transaction(step, sender_pool)
            )

            client.put_record(
                StreamName=STREAM_NAME,
                Data=json.dumps(txn).encode("utf-8"),
                PartitionKey=txn["transaction_id"],
            )

            # Truth goes to the database, never into the stream the model reads.
            if conn is not None:
                record_ground_truth(
                    conn,
                    txn["transaction_id"],
                    is_fraud,
                    f"producer:{fraud_profile}" if is_fraud else "producer:normal",
                )

            label = "FRAUD  " if is_fraud else "normal "
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] {label} "
                f"txn {txn['transaction_id'][:8]}... "
                f"amount={txn['amount']:>10.2f}"
            )

            sent += 1
            frauds += int(is_fraud)
            step += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        if conn is not None:
            conn.close()
        print(f"\nStopped. Sent {sent} transactions ({frauds} fraudulent).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic transaction producer")
    parser.add_argument(
        "--rate", type=float, default=1.0, help="Transactions per second (default: 1.0)"
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.05,
        help="Fraction of transactions that are fraudulent (default: 0.05)",
    )
    parser.add_argument(
        "--count", type=int, default=None, help="Total transactions to send (default: unlimited)"
    )
    parser.add_argument(
        "--fraud-profile",
        choices=sorted(FRAUD_PROFILES),
        default="paysim",
        help=(
            "'paysim' drains the origin account, the pattern the model was trained on. "
            "'shifted' takes small amounts and leaves a balance — invisible to the model "
            "and nearly invisible to data-drift monitoring."
        ),
    )
    parser.add_argument(
        "--record-ground-truth",
        action="store_true",
        help="Write true labels to the ground_truth table (needed for replay_labels.py)",
    )
    parser.add_argument(
        "--sender-pool",
        type=Path,
        default=None,
        help="Parquet of senders with Feast features (default: data/interim/paysim_repeat_senders.parquet)",
    )
    parser.add_argument(
        "--no-sender-pool",
        action="store_true",
        help="Draw senders at random, so Feast lookups miss (the old behaviour)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.record_ground_truth:
        from dotenv import load_dotenv

        load_dotenv()

    pool = None if args.no_sender_pool else load_sender_pool(args.sender_pool, size=5000)

    produce(
        rate=args.rate,
        fraud_rate=args.fraud_rate,
        count=args.count,
        fraud_profile=args.fraud_profile,
        sender_pool=pool,
        record_truth=args.record_ground_truth,
    )


if __name__ == "__main__":
    main()
