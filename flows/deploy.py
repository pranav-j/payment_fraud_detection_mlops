from pathlib import Path

from detect_drift import detect_drift_flow
from materialize_features import materialize_features_flow
from prefect import serve
from retrain_model import retrain_model_flow

PROJECT_ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    materialize_deployment = materialize_features_flow.to_deployment(
        name="daily-feast-materialization",
        cron="0 2 * * *",
        description="Refresh Redis online store from S3 parquet",
        parameters={
            "repo_path": str(PROJECT_ROOT / "feature_repo"),
        },
    )

    retrain_deployment = retrain_model_flow.to_deployment(
        name="weekly-model-retraining",
        cron="0 2 * * 0",
        description="Retrain fraud-detector and promote if better",
        parameters={
            "parquet_path": str(PROJECT_ROOT / "data/interim/paysim_with_features.parquet"),
        },
    )

    serve(materialize_deployment, retrain_deployment)


drift_deployment = detect_drift_flow.to_deployment(
    name="hourly-drift-detection",
    cron="0 * * * *",  # every hour
    description="Detect feature drift in recent predictions vs training baseline",
    parameters={
        "parquet_path": str(PROJECT_ROOT / "data/interim/paysim_with_features.parquet"),
    },
)

serve(materialize_deployment, retrain_deployment, drift_deployment)
