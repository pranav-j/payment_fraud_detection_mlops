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
DEFAULT_TRACKING_DB: Path = PROJECT_ROOT / "mlflow_data" / "mlflow.db"

# Where artifacts (model files, plots) get uploaded. This is the artifact store.
DEFAULT_ARTIFACT_ROOT: Path = PROJECT_ROOT / "mlflow_data" / "mlruns"

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
    # Resolve into separate, non-Optional locals. Reassigning the parameters
    # themselves keeps their declared `str | None` type, which makes every
    # downstream use look nullable to a type checker.
    resolved_experiment: str = experiment_name or os.environ.get(
        "MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT
    )

    resolved_uri: str = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{DEFAULT_TRACKING_DB}",
    )

    resolved_artifact_root: Path = artifact_root or Path(
        os.getenv("MLFLOW_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))
    )

    mlflow.set_tracking_uri(resolved_uri)

    # Only create local artifact directory if we're using a local file-based
    # tracking URI. When tracking_uri is an HTTP URL (Docker, production),
    # the server owns the artifact store and we have no business creating
    # local directories.
    if resolved_uri.startswith("sqlite:") or resolved_uri.startswith("file:"):
        resolved_artifact_root.mkdir(parents=True, exist_ok=True)

    # set_experiment is idempotent — it creates if missing, sets as active if found.
    # The artifact_location is only honored on creation; subsequent calls keep the
    # original location.
    mlflow.set_experiment(experiment_name=resolved_experiment)

    logger.info("MLflow configured")
    logger.info("  tracking_uri = %s", resolved_uri)
    logger.info("  artifact_root = %s", resolved_artifact_root)
    logger.info("  experiment   = %s", resolved_experiment)

    return resolved_experiment
