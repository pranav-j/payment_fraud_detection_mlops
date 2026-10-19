"""Shared pytest fixtures.

The important one here is `require_mlflow`. Tests marked `integration` need a
live MLflow tracking server with a `fraud-detector@production` model. When that
server isn't up — which is the case in CI, and on a laptop whenever the stack is
torn down — the MLflow client retries six times before failing, which turned
`make test` into a four-minute run that ended red. A cheap reachability probe
turns that into a skip.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest

# Keep MLflow's own HTTP timeout short so that a server which accepts the TCP
# connection but never answers can't stall the suite either.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")

PROBE_TIMEOUT_SECONDS = 2.0


def _resolved_tracking_uri() -> str:
    """The tracking URI `setup_mlflow()` would use, without importing MLflow."""
    # Imported lazily so that collecting the unit tests doesn't pull in MLflow.
    from fraud_mlops.tracking import DEFAULT_TRACKING_DB

    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{DEFAULT_TRACKING_DB}")


def mlflow_reachable(uri: str | None = None) -> bool:
    """Is the configured MLflow backend actually there?

    For HTTP(S) URIs this is a TCP connect probe — enough to tell "server up"
    from "nothing listening" without waiting on MLflow's retry ladder. For
    local SQLite/file stores it checks that the backing file exists.
    """
    uri = uri or _resolved_tracking_uri()
    parsed = urlparse(uri)

    if parsed.scheme in {"http", "https"}:
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_SECONDS):
                return True
        except OSError:
            return False

    if parsed.scheme in {"sqlite", ""}:
        # sqlite:////abs/path.db -> path is the remainder after the scheme
        db_path = uri.split("://", 1)[-1].lstrip("/")
        return Path("/" + db_path).exists() or Path(db_path).exists()

    if parsed.scheme == "file":
        return Path(parsed.path).exists()

    # postgresql:// and friends: assume reachable and let the test fail loudly.
    return True


@pytest.fixture
def require_mlflow() -> None:
    """Skip the test unless the MLflow backend is reachable."""
    uri = _resolved_tracking_uri()
    if not mlflow_reachable(uri):
        pytest.skip(f"MLflow backend not reachable at {uri} — skipping integration test")


# ── Synthetic PaySim ────────────────────────────────────────────────────────
# The real PaySim CSV is 470 MB and is never committed (see data/README.md), so
# tests that need transaction-shaped data generate it. The generator mirrors
# PaySim's own structure closely enough to exercise the pipeline: fraud occurs
# only in TRANSFER/CASH_OUT, and a fraudulent transaction empties the origin
# account — which is what makes the real dataset's balance columns such a
# giveaway (see the leakage section of reports/metrics/evaluation.md).


def make_synthetic_paysim(
    n_rows: int = 4000,
    n_senders: int = 300,
    n_steps: int = 48,
    fraud_rate: float = 0.05,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a PaySim-schema dataframe with a learnable fraud signal.

    Args:
        n_rows: Number of transactions.
        n_senders: Distinct sender accounts — fewer than rows, so senders repeat
            and the rolling velocity features have something to roll over.
        n_steps: Distinct simulated hours.
        fraud_rate: Fraction of rows labelled fraudulent. Much higher than real
            PaySim's 0.13% so that small fixtures still contain enough
            positives for a precision-recall curve to mean anything.
        seed: RNG seed; the generator is deterministic.

    Returns:
        A dataframe with exactly PaySim's columns, in PaySim's order.
    """
    rng = np.random.default_rng(seed)

    step = np.sort(rng.integers(1, n_steps + 1, size=n_rows))
    sender_ids = np.array([f"C{1_000_000_000 + i}" for i in range(n_senders)])
    name_orig = rng.choice(sender_ids, size=n_rows)
    name_dest = np.array([f"M{2_000_000_000 + i}" for i in rng.integers(0, n_senders, n_rows)])

    is_fraud = (rng.random(n_rows) < fraud_rate).astype(int)

    # Legitimate transactions: any type, partial debit of the origin balance.
    txn_type = rng.choice(
        ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
        size=n_rows,
        p=[0.34, 0.09, 0.35, 0.20, 0.02],
    )
    old_balance_org = np.round(rng.lognormal(mean=8.0, sigma=1.5, size=n_rows), 2)
    amount = np.round(old_balance_org * rng.uniform(0.01, 0.6, size=n_rows), 2)

    # Fraudulent transactions: TRANSFER/CASH_OUT only, and they drain the account.
    fraud_mask = is_fraud == 1
    txn_type[fraud_mask] = rng.choice(["TRANSFER", "CASH_OUT"], size=int(fraud_mask.sum()))
    amount[fraud_mask] = old_balance_org[fraud_mask]

    new_balance_orig = np.round(np.maximum(old_balance_org - amount, 0.0), 2)
    old_balance_dest = np.round(rng.lognormal(mean=7.0, sigma=2.0, size=n_rows), 2)
    new_balance_dest = np.round(old_balance_dest + amount, 2)
    new_balance_dest[fraud_mask] = 0.0

    return pd.DataFrame(
        {
            "step": step,
            "type": txn_type,
            "amount": amount,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_balance_org,
            "newbalanceOrig": new_balance_orig,
            "nameDest": name_dest,
            "oldbalanceDest": old_balance_dest,
            "newbalanceDest": new_balance_dest,
            "isFraud": is_fraud,
            # PaySim's own rule-based flag: transfers over 200,000. It leaks the
            # label by construction and is dropped before training.
            "isFlaggedFraud": ((txn_type == "TRANSFER") & (amount > 200_000)).astype(int),
        }
    )


@pytest.fixture
def synthetic_paysim():
    """A small PaySim-shaped dataframe, regenerated per test."""
    return make_synthetic_paysim()


@pytest.fixture(scope="session")
def paysim_factory():
    """`make_synthetic_paysim` itself, for tests that need custom sizes."""
    return make_synthetic_paysim
