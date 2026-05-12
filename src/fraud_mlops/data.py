"""Data loading utilities.

The notebook calls these. So will the future Prefect training flow.
Same code path, different invocation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fraud_mlops.config import (
    ID_COLUMNS,
    LABEL_COLUMN,
    LEAKY_COLUMNS,
    PAYSIM_PATH,
    TIME_COLUMN,
)

logger = logging.getLogger(__name__)


def load_paysim(path: Path | None = None, drop_leaky: bool = True) -> pd.DataFrame:
    """Load the PaySim dataset.

    Args:
        path: Path to the PaySim CSV. Defaults to the configured location.
        drop_leaky: If True, drop columns that leak the label or are pure IDs.

    Returns:
        DataFrame with the loaded transactions.

    Raises:
        FileNotFoundError: If the CSV is not at the expected path. Run
            `bash scripts/download_data.sh` first.
    """
    csv_path = path or PAYSIM_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {csv_path}. "
            f"Run `bash scripts/download_data.sh` to download it."
        )

    logger.info("Loading PaySim from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    if drop_leaky:
        cols_to_drop = [c for c in LEAKY_COLUMNS + ID_COLUMNS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info("Dropped leaky/ID columns: %s", cols_to_drop)

    return df


def time_based_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    time_col: str = TIME_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train and test by time.

    Random splits leak future information into the past. Fraud detection
    has temporal patterns (fraudsters adapt over time), so the only honest
    evaluation is to train on earlier data and test on later data.

    Args:
        df: The full dataframe.
        test_fraction: Fraction to use for test (the most recent data).
        time_col: Column to sort by.

    Returns:
        (train_df, test_df) sorted by time.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column {time_col!r} not in dataframe")

    # Sort and split — no shuffling.
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))

    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]

    logger.info(
        "Time-based split: train=%d (steps %d–%d), test=%d (steps %d–%d)",
        len(train),
        train[time_col].min(),
        train[time_col].max(),
        len(test),
        test[time_col].min(),
        test[time_col].max(),
    )

    return train, test


def split_features_label(
    df: pd.DataFrame,
    label_col: str = LABEL_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from label (y)."""
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not in dataframe")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y
