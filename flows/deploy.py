"""Register every scheduled deployment with the Prefect server.

    python flows/deploy.py

Note what this file used to do: `serve(...)` was called inside the
`if __name__ == "__main__"` block with only two deployments, and that call
blocks forever — so the drift deployment created below it was never reached.
Imported rather than run, the same file raised NameError. The hourly drift flow
was therefore never actually registered by this script.

One `serve()` call, with every deployment, at the end.
"""

from pathlib import Path

from detect_drift import detect_drift_flow
from materialize_features import materialize_features_flow
from monitor_performance import monitor_performance_flow
from prefect import serve
from retrain_model import retrain_model_flow

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PARQUET = str(PROJECT_ROOT / "data/interim/paysim_with_features.parquet")


def build_deployments() -> list:
    """Every scheduled deployment, as data, so it can be inspected in a test."""
    return [
        materialize_features_flow.to_deployment(
            name="daily-feast-materialization",
            cron="0 2 * * *",
            description="Refresh Redis online store from S3 parquet",
            parameters={"repo_path": str(PROJECT_ROOT / "feature_repo")},
        ),
        retrain_model_flow.to_deployment(
            name="weekly-model-retraining",
            cron="0 2 * * 0",
            description="Retrain fraud-detector and promote if better",
            parameters={"parquet_path": FEATURES_PARQUET},
        ),
        detect_drift_flow.to_deployment(
            name="hourly-drift-detection",
            cron="0 * * * *",
            description="Detect feature drift in recent predictions vs training baseline",
            parameters={"parquet_path": FEATURES_PARQUET},
        ),
        # Daily, not hourly: labels arrive on the order of days, so an hourly
        # run would re-report the same matured windows over and over.
        monitor_performance_flow.to_deployment(
            name="daily-performance-monitoring",
            cron="30 3 * * *",
            description="Realized precision/recall against labels that have matured",
            parameters={"lookback_days": 14, "window_hours": 24},
        ),
    ]


if __name__ == "__main__":
    serve(*build_deployments())
