"""Fraud detection inference.

Loads the production model from MLflow's registry, exposes a clean
function-based API for predictions. No HTTP, no async — just deterministic
Python that returns typed results.

Designed so the same module can be imported by:
  - FastAPI (week 3)
  - AWS Lambda (week 5)
  - Prefect evaluation flows (week 6)
  - Tests (always)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from fraud_mlops.tracking import setup_mlflow

logger = logging.getLogger(__name__)

# The name of the registered model in MLflow.
REGISTERED_MODEL_NAME = "fraud-detector"

# The alias we resolve to find the current production version.
PRODUCTION_ALIAS = "production"

# The exact column order the pipeline expects. Wrong order = silent
# corruption (the ColumnTransformer indexes by position when given a
# DataFrame). Keep this list synced with the feature set used at training.
EXPECTED_FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "sender_txn_count_1h",
    "sender_txn_count_24h",
    "sender_amount_sum_24h",
    "sender_amount_mean_historical",
    "sender_time_since_last_txn",
    "amount_to_oldbalance_ratio",
    "drains_origin",
]


@dataclass(frozen=True)
class Prediction:
    """The result of a single fraud check.

    Frozen so callers can't accidentally mutate it. Includes the model
    version and threshold so the downstream audit log can record exactly
    which model produced which decision.
    """

    is_fraud: bool
    probability: float
    threshold: float
    model_version: str
    feature_set: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FraudDetector:
    """Wraps a loaded model + its decision threshold + its metadata.

    Constructed via `load_production_detector()` in normal use. Tests
    can build their own with a mock model.
    """

    def __init__(
        self,
        model: Any,
        threshold: float,
        version: str,
        feature_set: str,
    ) -> None:
        if not isinstance(version, str):
            raise TypeError(f"version must be str, got {type(version).__name__}={version!r}")
        self._model = model
        self._threshold = threshold
        self._version = version
        self._feature_set = feature_set

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def version(self) -> str:
        return self._version

    @property
    def feature_set(self) -> str:
        return self._feature_set

    def predict_one(self, transaction: dict[str, Any]) -> Prediction:
        """Score a single transaction.

        Args:
            transaction: dict with all keys from EXPECTED_FEATURE_COLUMNS.

        Returns:
            Prediction dataclass.

        Raises:
            KeyError: if any required feature is missing.
        """
        return self.predict_batch([transaction])[0]

    def predict_batch(self, transactions: list[dict[str, Any]]) -> list[Prediction]:
        """Score multiple transactions in one model call.

        More efficient than looping predict_one — sklearn vectorizes
        across rows.
        """
        if not transactions:
            return []

        # Validate keys upfront with a clear error rather than a cryptic
        # pandas error later.
        for i, txn in enumerate(transactions):
            missing = set(EXPECTED_FEATURE_COLUMNS) - set(txn.keys())
            if missing:
                raise KeyError(f"Transaction {i} is missing required features: {sorted(missing)}")

        # Build the DataFrame with explicit column order. Critical.
        df = pd.DataFrame(transactions)[EXPECTED_FEATURE_COLUMNS]

        # predict_proba returns shape (n, 2); column 1 is P(fraud)
        probabilities = self._model.predict_proba(df)[:, 1]

        return [
            Prediction(
                is_fraud=bool(p >= self._threshold),
                probability=float(p),
                threshold=self._threshold,
                model_version=self._version,
                feature_set=self._feature_set,
            )
            for p in probabilities
        ]


def load_production_detector() -> FraudDetector:
    """Factory: resolve the production alias and load the model.

    Talks to MLflow. Cache the result at the call site (FastAPI startup,
    Lambda container init) — don't call this on every request.
    """
    setup_mlflow()  # ensure tracking URI is set
    client = MlflowClient()

    version = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=PRODUCTION_ALIAS,
    )

    threshold = float(version.tags["calibrated_threshold"])
    feature_set = version.tags.get("feature_set", "unknown")

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)

    logger.info(
        "Loaded fraud-detector v%s (feature_set=%s, threshold=%.4f)",
        str(version.version),
        feature_set,
        threshold,
    )

    return FraudDetector(
        model=model,
        threshold=threshold,
        version=str(version.version),
        feature_set=feature_set,
    )
