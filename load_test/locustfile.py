"""Load test for the FastAPI `/predict` endpoint.

    uv run --with locust locust -f load_test/locustfile.py --host http://<alb-dns>

    # headless, for the measurement session:
    uv run --with locust locust -f load_test/locustfile.py \
        --host http://<alb-dns> --headless \
        --users 50 --spawn-rate 10 --run-time 3m \
        --csv reports/perf/$(date +%F)/locust-50rps

Payloads are realistic rather than constant. A load test that posts the same
body every time measures a cache-friendly path: XGBoost takes a different route
through the trees for different inputs, and a fixed payload also hides whatever
the one-hot encoder does with unseen categories.

The mix deliberately includes fraud-shaped transactions at roughly the rate the
producer generates them, so the measured latency covers both branches of the
decision.
"""

from __future__ import annotations

import random
import uuid

from locust import HttpUser, between, events, task

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

# Matches scripts/produce_transactions.py, so the two paths are exercised with
# the same traffic shape.
FRAUD_RATE = 0.05


def _normal_payload() -> dict:
    old_balance = round(random.lognormvariate(8.0, 2.0), 2)
    amount = round(old_balance * random.uniform(0.01, 0.6), 2)
    txn_count_24h = random.randint(0, 20)

    return {
        "step": random.randint(1, 743),
        "type": random.choice(TRANSACTION_TYPES),
        "amount": max(amount, 0.01),
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": round(max(old_balance - amount, 0.0), 2),
        "oldbalanceDest": round(random.lognormvariate(7.0, 2.0), 2),
        "newbalanceDest": round(random.lognormvariate(7.0, 2.0), 2),
        "sender_txn_count_1h": random.randint(0, 3),
        "sender_txn_count_24h": txn_count_24h,
        "sender_amount_sum_24h": round(random.lognormvariate(8.0, 1.5), 2),
        "sender_amount_mean_historical": round(random.lognormvariate(7.0, 1.0), 2),
        "sender_time_since_last_txn": round(random.uniform(0.1, 48.0), 2),
        "amount_to_oldbalance_ratio": round(amount / old_balance, 4) if old_balance else -1.0,
        "drains_origin": 0,
    }


def _fraud_payload() -> dict:
    """PaySim's fraud signature: the origin account is emptied."""
    old_balance = round(random.lognormvariate(8.5, 1.5), 2)
    payload = _normal_payload()
    payload.update(
        {
            "type": random.choice(FRAUD_TYPES),
            "amount": old_balance,
            "oldbalanceOrg": old_balance,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "amount_to_oldbalance_ratio": 1.0,
            "drains_origin": 1,
            # A first-time sender, which is what Feast returns defaults for.
            "sender_txn_count_1h": 0,
            "sender_txn_count_24h": 0,
            "sender_amount_sum_24h": 0.0,
            "sender_amount_mean_historical": 0.0,
            "sender_time_since_last_txn": -1.0,
        }
    )
    return payload


class FraudScoringUser(HttpUser):
    """A caller asking for a synchronous fraud decision."""

    # Think time between requests. Locust's RPS is users / mean wait, so the
    # measurement session sets the user count to hit a target rate.
    wait_time = between(0.5, 1.5)

    @task(19)
    def predict_normal(self) -> None:
        self._predict(_normal_payload(), name="/predict [normal]")

    @task(1)
    def predict_fraud(self) -> None:
        self._predict(_fraud_payload(), name="/predict [fraud-shaped]")

    @task(2)
    def health(self) -> None:
        """Cheap endpoint, as a control: isolates model time from network time."""
        self.client.get("/health", name="/health")

    def _predict(self, payload: dict, name: str) -> None:
        with self.client.post(
            "/predict",
            json=payload,
            name=name,
            catch_response=True,
            headers={"X-Request-Id": str(uuid.uuid4())},
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            body = response.json()
            if "is_fraud" not in body or "probability" not in body:
                response.failure(f"Unexpected response shape: {sorted(body)}")
                return
            response.success()


@events.test_start.add_listener
def on_test_start(environment, **_kwargs) -> None:
    print(f"Load test starting against {environment.host}")
    print(f"Fraud-shaped payload share: {FRAUD_RATE:.0%}")


@events.test_stop.add_listener
def on_test_stop(environment, **_kwargs) -> None:
    stats = environment.stats.total
    print(
        f"\nTotal: {stats.num_requests} requests, {stats.num_failures} failures "
        f"({stats.fail_ratio:.2%})"
    )
    if stats.num_requests:
        print(
            f"Latency: p50 {stats.get_response_time_percentile(0.5):.0f} ms, "
            f"p95 {stats.get_response_time_percentile(0.95):.0f} ms, "
            f"p99 {stats.get_response_time_percentile(0.99):.0f} ms"
        )
