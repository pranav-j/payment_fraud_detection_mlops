"""Shared model construction.

The same preprocessing + XGBoost pipeline was written out three times — twice in
notebooks and once in `flows/retrain_model.py` — with hyperparameters that had
already drifted apart (200 estimators in the notebooks, 100 in the flow). This
module is the single definition. Callers pass the hyperparameters they actually
want, so no caller's behaviour changes by adopting it.

Preprocessing lives *inside* the pipeline on purpose: the fitted scaler and
one-hot encoder travel with the model artifact, so the serving path cannot
preprocess differently from training.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_mlops.config import RANDOM_SEED

logger = logging.getLogger(__name__)

# The one categorical column in the PaySim feature sets.
CATEGORICAL_COLUMNS: list[str] = ["type"]

# Hyperparameters of the registered production model (`enriched_v1`), as logged
# by notebook 03. `flows/retrain_model.py` overrides n_estimators to 100.
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Scale the numeric columns, one-hot the categorical ones.

    Selection is by column *name*, so the caller's column order does not matter.
    """
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]

    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


def compute_scale_pos_weight(y: pd.Series) -> float:
    """XGBoost's class-weighting term: negatives per positive.

    Class weights rather than resampling — see ADR-005.
    """
    n_positive = int((y == 1).sum())
    if n_positive == 0:
        raise ValueError("Cannot compute scale_pos_weight: the label column has no positives")
    return float((y == 0).sum() / n_positive)


def build_xgb_pipeline(
    feature_columns: list[str],
    scale_pos_weight: float,
    **xgb_overrides: Any,
) -> Pipeline:
    """Preprocessing + XGBoost classifier, ready to `.fit(X, y)`.

    Args:
        feature_columns: Columns the model will be trained on.
        scale_pos_weight: From `compute_scale_pos_weight`.
        **xgb_overrides: Any XGBClassifier parameter, overriding
            `DEFAULT_XGB_PARAMS` (e.g. `n_estimators=100`).
    """
    import xgboost as xgb  # imported lazily: heavy, and not needed to import this module

    params = {**DEFAULT_XGB_PARAMS, **xgb_overrides, "scale_pos_weight": scale_pos_weight}
    logger.info(
        "Building XGBoost pipeline with %d features, params=%s", len(feature_columns), params
    )

    return Pipeline(
        [
            ("prep", build_preprocessor(feature_columns)),
            ("clf", xgb.XGBClassifier(**params)),
        ]
    )


def build_logistic_pipeline(feature_columns: list[str]) -> Pipeline:
    """Balanced logistic regression — the linear sanity check.

    If XGBoost only marginally beats this, the gain is coming from somewhere
    other than the model.
    """
    from sklearn.linear_model import LogisticRegression

    return Pipeline(
        [
            ("prep", build_preprocessor(feature_columns)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
