"""Weekly model retraining flow.

Retrains the fraud-detector model on the PaySim dataset and promotes
it to production if it outperforms the current model.
"""

from __future__ import annotations

from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
from sklearn.metrics import precision_recall_curve

from fraud_mlops.features import ENRICHED_FEATURE_COLUMNS
from fraud_mlops.promotion import MIN_RECALL_IMPROVEMENT, PRECISION_FLOOR, evaluate_gate
from fraud_mlops.training import build_xgb_pipeline, compute_scale_pos_weight

load_dotenv()

# This flow trains with 100 estimators where the registered production model
# used 200. Kept as-is rather than silently retuned; see CLAUDE.md non-goals.
RETRAIN_N_ESTIMATORS = 100


@task
def load_training_data(parquet_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and split features/labels from the PaySim parquet."""
    logger = get_run_logger()

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from %s", len(df), parquet_path)

    feature_cols = ENRICHED_FEATURE_COLUMNS

    X = df[feature_cols]
    y = df["isFraud"]

    logger.info(
        "Features: %d cols, Label distribution: %.4f%% fraud", len(feature_cols), y.mean() * 100
    )
    return X, y


@task
def get_current_production_metrics() -> dict:
    """Fetch the current production model's metrics from MLflow."""
    logger = get_run_logger()
    client = MlflowClient()

    try:
        version = client.get_model_version_by_alias("fraud-detector", "production")
        metrics = {
            "version": version.version,
            "threshold": float(version.tags.get("calibrated_threshold", 0.5)),
            "precision": float(version.tags.get("precision_at_calibration", 0.0)),
            "recall": float(version.tags.get("recall_at_calibration", 0.0)),
        }
        logger.info(
            "Current production model: v%s (recall=%.4f, precision=%.4f)",
            metrics["version"],
            metrics["recall"],
            metrics["precision"],
        )
        return metrics
    except Exception as e:
        logger.warning("Could not fetch production metrics: %s. Using defaults.", e)
        return {"version": "none", "threshold": 0.5, "precision": 0.0, "recall": 0.0}


@task(retries=1, retry_delay_seconds=30)
def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train XGBoost model and evaluate with calibrated threshold."""
    from sklearn.model_selection import train_test_split

    logger = get_run_logger()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d rows, Test: %d rows", len(X_train), len(X_test))

    # tree_method/n_jobs are passed as None so this flow keeps XGBoost's own
    # defaults, exactly as it did before the pipeline builder was shared. Only
    # the duplication is gone; the candidate model is unchanged.
    model = build_xgb_pipeline(
        feature_columns=list(X.columns),
        scale_pos_weight=compute_scale_pos_weight(y_train),
        n_estimators=RETRAIN_N_ESTIMATORS,
        tree_method=None,
        n_jobs=None,
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    # Calibrate threshold: max recall subject to precision >= 0.95
    probabilities = model.predict_proba(X_test)[:, 1]
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, probabilities)

    valid = precision_vals[:-1] >= 0.95
    if valid.any():
        best_idx = np.argmax(recall_vals[:-1][valid])
        threshold = float(thresholds[valid][best_idx])
        precision_at_threshold = float(precision_vals[:-1][valid][best_idx])
        recall_at_threshold = float(recall_vals[:-1][valid][best_idx])
    else:
        threshold = 0.5
        precision_at_threshold = 0.0
        recall_at_threshold = 0.0

    logger.info(
        "New model: threshold=%.4f precision=%.4f recall=%.4f",
        threshold,
        precision_at_threshold,
        recall_at_threshold,
    )

    return {
        "model": model,
        "threshold": threshold,
        "precision": precision_at_threshold,
        "recall": recall_at_threshold,
        "X_test": X_test,
        "y_test": y_test,
    }


@task
def register_and_promote_if_better(
    results: dict,
    current_metrics: dict,
    min_recall_improvement: float = MIN_RECALL_IMPROVEMENT,
) -> bool:
    """Register model in MLflow and promote if better than current production."""
    logger = get_run_logger()

    decision = evaluate_gate(
        candidate_precision=results["precision"],
        candidate_recall=results["recall"],
        production_recall=current_metrics["recall"],
        precision_floor=PRECISION_FLOOR,
        min_recall_improvement=min_recall_improvement,
    )
    logger.info("Promotion gate: %s", decision.reason)

    if not decision.promote:
        logger.info("Gate refused promotion. Production alias unchanged.")
        return False

    # Register in MLflow
    with mlflow.start_run(run_name=f"retrain_{datetime.utcnow().strftime('%Y%m%d')}"):
        mlflow.log_params(
            {
                "n_estimators": RETRAIN_N_ESTIMATORS,
                "max_depth": 6,
                "learning_rate": 0.1,
            }
        )
        mlflow.log_metrics(
            {
                "precision_at_calibration": results["precision"],
                "recall_at_calibration": results["recall"],
                "calibrated_threshold": results["threshold"],
            }
        )

        model_info = mlflow.sklearn.log_model(
            results["model"],
            artifact_path="model",
            registered_model_name="fraud-detector",
        )

    # Tag and promote
    client = MlflowClient()
    new_version = model_info.registered_model_version

    client.set_model_version_tag(
        name="fraud-detector",
        version=new_version,
        key="calibrated_threshold",
        value=str(results["threshold"]),
    )
    client.set_model_version_tag(
        name="fraud-detector",
        version=new_version,
        key="precision_at_calibration",
        value=str(results["precision"]),
    )
    client.set_model_version_tag(
        name="fraud-detector",
        version=new_version,
        key="recall_at_calibration",
        value=str(results["recall"]),
    )
    client.set_model_version_tag(
        name="fraud-detector",
        version=new_version,
        key="feature_set",
        value="enriched_v1",
    )
    client.set_registered_model_alias(
        name="fraud-detector",
        alias="production",
        version=new_version,
    )

    logger.info("Promoted fraud-detector v%s to @production", new_version)
    return True


@flow(name="model-retraining", log_prints=True)
def retrain_model_flow(
    parquet_path: str = "data/interim/paysim_with_features.parquet",
    min_recall_improvement: float = MIN_RECALL_IMPROVEMENT,
) -> None:
    """Weekly flow: retrain fraud-detector and promote if better."""
    logger = get_run_logger()
    logger.info("Starting model retraining flow")

    X, y = load_training_data(parquet_path)
    current_metrics = get_current_production_metrics()
    results = train_and_evaluate(X, y)
    promoted = register_and_promote_if_better(
        results=results,
        current_metrics=current_metrics,
        min_recall_improvement=min_recall_improvement,
    )

    if promoted:
        logger.info("New model promoted to production.")
    else:
        logger.info("Current model retained. No promotion.")


if __name__ == "__main__":
    retrain_model_flow()
