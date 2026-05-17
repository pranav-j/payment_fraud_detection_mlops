"""MLflow tracking setup.

Centralizes MLflow configuration so every notebook and every Prefect flow
uses the same tracking server, the same artifact store, and the same
experiment naming convention.

Week 2 (now): local SQLite backend + local file artifact store.
Week 3 (later): swap to RDS Postgres + S3 by changing two env vars.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow

from fraud_mlops.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Where the SQLite metadata file lives. This is the backend store.
DEFAULT_TRACKING_DB: Path = PROJECT_ROOT / "mlflow.db"

# Where artifacts (model files, plots) get uploaded. This is the artifact store.
DEFAULT_ARTIFACT_ROOT: Path = PROJECT_ROOT / "mlruns"

# Default experiment name. Override via env var if you want.
DEFAULT_EXPERIMENT: str = "fraud_baseline"


def setup_mlflow(
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    artifact_root: Path | None = None,
) -> str:
    """Configure MLflow for this session and return the active experiment name.

    Reads from environment variables if set, otherwise uses local defaults.
    Idempotent — safe to call multiple times in the same Python process.

    Args:
        experiment_name: Override the default experiment name.
        tracking_uri: Override the tracking server URI (default: local SQLite).
        artifact_root: Override the artifact store path.

    Returns:
        The active experiment name.
    """
    experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT)

    tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{DEFAULT_TRACKING_DB}",
    )

    artifact_root = artifact_root or Path(
        os.getenv("MLFLOW_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))
    )

    # Ensure artifact dir exists. SQLite file is auto-created on first use.
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)

    # set_experiment is idempotent — it creates if missing, sets as active if found.
    # The artifact_location is only honored on creation; subsequent calls keep the
    # original location.
    mlflow.set_experiment(experiment_name=experiment_name)

    logger.info("MLflow configured")
    logger.info("  tracking_uri = %s", tracking_uri)
    logger.info("  artifact_root = %s", artifact_root)
    logger.info("  experiment   = %s", experiment_name)

    return experiment_name
