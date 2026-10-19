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


def _snap_boundary(step_counts: pd.Series, target_rows: int) -> int:
    """Pick the time step whose cumulative row count is closest to `target_rows`.

    Returns the last step that belongs to the earlier side of the split.
    """
    cumulative = step_counts.cumsum()
    # Index of the step whose cumulative count is nearest the target.
    position = int((cumulative - target_rows).abs().to_numpy().argmin())
    return int(cumulative.index[position])


def time_based_split_three_way(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    time_col: str = TIME_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / validation / test by time, on whole time steps.

    Two differences from `time_based_split`, both deliberate:

    1. **Three ways, not two.** The threshold has to be tuned somewhere other
       than the set it is reported on. Without a validation split, the only
       options are "tune on train" (optimistic for the wrong reason) or "tune on
       test" (optimistic for the obvious reason).
    2. **Boundaries land between time steps.** `time_based_split` cuts at a row
       index, so the simulated hour at the boundary ends up partly in train and
       partly in test. Here a step belongs wholly to one split.

    Note that PaySim's volume is heavily front-loaded: 80% of *rows* is only
    about 48% of *hours*. These fractions are of rows, with the step boundary
    snapped to whichever whole step comes closest, so the reported step ranges
    will look lopsided against the row counts. That is the data, not a bug.

    Args:
        df: The full dataframe.
        val_fraction: Target fraction of rows for validation (the middle slice).
        test_fraction: Target fraction of rows for test (the latest slice).
        time_col: Column to split on.

    Returns:
        (train_df, val_df, test_df), each sorted by `time_col`.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column {time_col!r} not in dataframe")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + test_fraction must leave room for training "
            f"(got {val_fraction} + {test_fraction})"
        )

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    step_counts = df_sorted.groupby(time_col).size().sort_index()
    n = len(df_sorted)

    train_end = _snap_boundary(step_counts, int(n * (1 - val_fraction - test_fraction)))
    val_end = _snap_boundary(step_counts, int(n * (1 - test_fraction)))

    # Guard against both boundaries snapping to the same step on tiny inputs.
    if val_end <= train_end:
        val_end = train_end + 1

    train = df_sorted[df_sorted[time_col] <= train_end]
    val = df_sorted[(df_sorted[time_col] > train_end) & (df_sorted[time_col] <= val_end)]
    test = df_sorted[df_sorted[time_col] > val_end]

    for name, part in (("train", train), ("validation", val), ("test", test)):
        if part.empty:
            raise ValueError(
                f"The {name} split is empty — too few distinct values of {time_col!r} "
                f"({df_sorted[time_col].nunique()}) for a three-way split"
            )

    logger.info(
        "Three-way time split: train=%d (steps %d–%d), val=%d (steps %d–%d), test=%d (steps %d–%d)",
        len(train),
        train[time_col].min(),
        train[time_col].max(),
        len(val),
        val[time_col].min(),
        val[time_col].max(),
        len(test),
        test[time_col].min(),
        test[time_col].max(),
    )

    return train, val, test
