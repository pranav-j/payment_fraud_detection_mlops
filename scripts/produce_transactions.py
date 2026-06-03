#!/usr/bin/env python3
"""Synthetic transaction producer for fraud-mlops.

Pumps PaySim-style transactions into the Kinesis stream at a
configurable rate. Supports controlled fraud injection for demos.

Usage:
    python scripts/produce_transactions.py
    python scripts/produce_transactions.py --rate 5 --fraud-rate 0.1
    python scripts/produce_transactions.py --count 100 --fraud-rate 0.5
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from datetime import datetime

import boto3

STREAM_NAME = "fraud-transactions"
REGION = "ap-south-1"

# Realistic PaySim feature distributions (from EDA in Week 1)
TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]  # fraud only occurs in these types


def generate_normal_transaction(step: int) -> dict:
    txn_type = random.choice(TRANSACTION_TYPES)
    amount = random.lognormvariate(7.0, 1.5)
    old_balance = random.lognormvariate(8.0, 2.0)
    new_balance = max(0.0, old_balance - amount)

    return {
        "transaction_id": str(uuid.uuid4()),
        "sender_id": f"C{random.randint(1000000000, 9999999999)}",
        "step": step,
        "type": txn_type,
        "amount": round(amount, 2),
        "oldbalanceOrg": round(old_balance, 2),
        "newbalanceOrig": round(new_balance, 2),
        "oldbalanceDest": round(random.lognormvariate(7.0, 2.0), 2),
        "newbalanceDest": round(random.lognormvariate(7.0, 2.0), 2),
    }


def generate_fraud_transaction(step: int) -> dict:
    old_balance = round(random.lognormvariate(8.5, 1.5), 2)
    txn_type = random.choice(FRAUD_TYPES)

    return {
        "transaction_id": str(uuid.uuid4()),
        "sender_id": f"C{random.randint(1000000000, 9999999999)}",
        "step": step,
        "type": txn_type,
        "amount": old_balance,
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }


def produce(rate: float, fraud_rate: float, count: int | None) -> None:
    client = boto3.client("kinesis", region_name=REGION)
    interval = 1.0 / rate
    step = 1
    sent = 0

    print(
        f"Producing to stream '{STREAM_NAME}' at {rate} txn/sec " f"(fraud rate: {fraud_rate:.0%})"
    )
    print("Ctrl-C to stop\n")

    try:
        while count is None or sent < count:
            is_fraud = random.random() < fraud_rate
            txn = (
                generate_fraud_transaction(step) if is_fraud else generate_normal_transaction(step)
            )

            client.put_record(
                StreamName=STREAM_NAME,
                Data=json.dumps(txn).encode("utf-8"),
                PartitionKey=txn["transaction_id"],
            )

            label = "FRAUD  " if is_fraud else "normal "
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] {label} "
                f"txn {txn['transaction_id'][:8]}... "
                f"amount={txn['amount']:>10.2f}"
            )

            sent += 1
            step += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\nStopped. Sent {sent} transactions.")


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
    args = parser.parse_args()

    produce(args.rate, args.fraud_rate, args.count)


if __name__ == "__main__":
    main()
