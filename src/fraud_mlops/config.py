"""Project configuration: paths, seeds, and shared constants.

Everything that might be tweaked between environments lives here.
Notebooks and scripts import from this module rather than hardcoding values.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present. dotenv silently does nothing if the file is missing,
# which is fine — defaults below cover all required values.
load_dotenv()

# Project root — resolved from this file's location, not the cwd. This means
# imports work the same whether you run from the project root, from notebooks/,
# or from anywhere else.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_INTERIM_DIR: Path = PROJECT_ROOT / os.getenv("DATA_INTERIM_DIR", "data/interim")
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")

# Model and report paths
MODELS_DIR: Path = PROJECT_ROOT / os.getenv("MODELS_DIR", "models")
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Reproducibility
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# PaySim-specific constants
# The dataset has a single CSV with this filename. Hardcoded because Kaggle
# guarantees the name; revisit if PaySim is ever re-released.
PAYSIM_FILENAME: str = "PS_20174392719_1491204439457_log.csv"
PAYSIM_PATH: Path = DATA_RAW_DIR / PAYSIM_FILENAME

# Columns to drop before training:
# - isFlaggedFraud is a rule-based flag set by PaySim's simulator (transfers > 200K).
#   It's a leak: if the model sees it, it learns the rule, not actual fraud patterns.
# - nameOrig and nameDest are unique IDs; using them as features would memorize, not learn.
LEAKY_COLUMNS: list[str] = ["isFlaggedFraud"]
ID_COLUMNS: list[str] = ["nameOrig", "nameDest"]

# The label column.
LABEL_COLUMN: str = "isFraud"

# Time column (PaySim uses 'step' = simulated hours, 1..744 = 1 month).
TIME_COLUMN: str = "step"


def ensure_dirs() -> None:
    """Create all project directories if they don't exist. Safe to call repeatedly."""
    for d in (
        DATA_RAW_DIR,
        DATA_INTERIM_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
