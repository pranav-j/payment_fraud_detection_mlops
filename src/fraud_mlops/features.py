"""Feature engineering for the enriched (`enriched_v1`) feature set.

This logic used to live in a cell of `notebooks/03_feature_engineering.ipynb`,
which made it impossible to test or to call from a flow. It lives here so the
notebook, the evaluation script and the CI smoke test all compute features
exactly one way.

The feature *definitions* are unchanged — same windows, same sentinels, same
column names; see the "non-goals" in `CLAUDE.md`. Two things did change on the
way over:

1. **A correctness fix.** The notebook assigned each rolling result back to the
   frame after `.sort_index()`, which put the values in timestamp order while
   the frame was in sender order. Every sender's velocity feature landed on
   some other sender's row. `_rolling_per_sender` drops that sort, and
   `_assert_no_lookahead` makes the invariant checkable. Data produced before
   this fix — including `data/interim/paysim_with_features.parquet` and
   anything materialized into Feast from it — has the three rolling columns
   scrambled.
2. **Cosmetics.** `print` became `logging`, and the sender-ID column is a
   parameter because the raw CSV calls it `nameOrig` while the cached parquet
   calls it `sender`.

**The lookahead-leakage trap:** `closed="left"` on each rolling window keeps the
current row out of its own window. Without it every feature would silently
include "now" in "the last 24h".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_mlops.config import TIME_COLUMN

logger = logging.getLogger(__name__)

# The seven columns this module adds.
ENGINEERED_COLUMNS: list[str] = [
    "sender_time_since_last_txn",
    "sender_amount_mean_historical",
    "sender_txn_count_1h",
    "sender_txn_count_24h",
    "sender_amount_sum_24h",
    "amount_to_oldbalance_ratio",
    "drains_origin",
]

# PaySim's `step` is a simulated hour. Rolling windows need a real timestamp, so
# step N is mapped onto this origin + N hours. The origin is arbitrary but fixed:
# it also defines Feast's `event_timestamp`, so it must not drift.
TIMESTAMP_ORIGIN = "2024-01-01"

# ── Feature sets ────────────────────────────────────────────────────────────
# PaySim columns kept as-is by `load_paysim(drop_leaky=True)`. This is the "v1"
# baseline feature set from notebook 02.
BASELINE_FEATURE_COLUMNS: list[str] = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# The "v2" / `enriched_v1` set the production model uses: baseline + engineered.
# `inference.EXPECTED_FEATURE_COLUMNS` is the same set in serving order; a test
# asserts the two cannot drift apart.
ENRICHED_FEATURE_COLUMNS: list[str] = BASELINE_FEATURE_COLUMNS + ENGINEERED_COLUMNS

# PaySim's four raw balance columns, plus the two features derived from them.
# Held out as a group for the leakage experiment in `fraud_mlops.evaluate`:
# in PaySim, fraud almost always zeroes the origin balance, which makes these
# columns a near-deterministic giveaway rather than a learned signal.
BALANCE_COLUMNS: list[str] = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "amount_to_oldbalance_ratio",
    "drains_origin",
]


def _rolling_per_sender(
    df: pd.DataFrame,
    sender_col: str,
    window: str,
    agg: str,
) -> np.ndarray:
    """Rolling `agg` of `amount` over `window`, per sender, excluding the current row.

    `df` must already be sorted by (sender, timestamp). `groupby(sort=False)`
    then walks the groups in that same order, so the result lines up positionally
    with `df` and can be assigned straight back.

    That positional alignment is the whole subtlety here. The original notebook
    version called `.sort_index()` on the result before assigning it, which
    re-sorted the values into *timestamp* order while the frame was still in
    *sender* order — silently scattering each sender's velocity features onto
    other senders' rows. `_assert_no_lookahead` below is the guard that would
    have caught it.
    """
    rolled = (
        df.set_index("_ts")
        .groupby(sender_col, sort=False)["amount"]
        .rolling(window, closed="left")
        .agg(agg)
    )
    values: np.ndarray = np.asarray(rolled.to_numpy())
    if len(values) != len(df):
        raise AssertionError(
            f"rolling {agg} over {window} returned {len(values)} values for {len(df)} rows"
        )
    return values


def _assert_no_lookahead(df: pd.DataFrame, sender_col: str) -> None:
    """Cheap invariants that any correctly aligned feature set must satisfy.

    Vectorized and O(n), so it runs on every call rather than only in tests —
    these are exactly the properties the original implementation violated, and
    nothing was checking them.
    """
    first_of_sender = df.groupby(sender_col, sort=False).cumcount() == 0

    for col, expected in (
        ("sender_txn_count_1h", 0),
        ("sender_txn_count_24h", 0),
        ("sender_amount_sum_24h", 0.0),
    ):
        offenders = int((df.loc[first_of_sender, col] != expected).sum())
        if offenders:
            raise AssertionError(
                f"{offenders} rows are a sender's first transaction yet have "
                f"{col} != {expected} — the rolling features are misaligned"
            )

    inverted = int((df["sender_txn_count_24h"] < df["sender_txn_count_1h"]).sum())
    if inverted:
        raise AssertionError(
            f"{inverted} rows have a 24h count below their 1h count — "
            f"the rolling windows are misaligned"
        )


def engineer_features(
    df: pd.DataFrame,
    sender_col: str = "nameOrig",
) -> pd.DataFrame:
    """Build temporal and ratio features per sender.

    Vectorized implementation: avoids per-group Python apply loops, which
    are catastrophically slow on PaySim's 6M+ rows.

    Args:
        df: PaySim transactions, including `sender_col`, `step`, `amount`,
            `oldbalanceOrg` and `newbalanceOrig`.
        sender_col: Name of the sender-ID column to group by.

    Returns:
        A copy of `df` with `ENGINEERED_COLUMNS` added, sorted by `step`.
        Helper columns (`_ts`, `nameDest`) are dropped.
    """
    if sender_col not in df.columns:
        raise ValueError(f"Sender column {sender_col!r} not in dataframe: {list(df.columns)}")

    df = df.copy()

    # Treat `step` (hours) as a datetime for time-based windows
    df["_ts"] = pd.to_datetime(df[TIME_COLUMN], unit="h", origin=TIMESTAMP_ORIGIN)

    # Sort by sender, then time within sender. Crucial for the rolling logic below.
    df = df.sort_values([sender_col, "_ts"]).reset_index(drop=True)

    # ---- Time since last transaction (vectorized, fast) ----
    logger.info("Computing time since last transaction...")
    df["sender_time_since_last_txn"] = (
        df.groupby(sender_col, sort=False)["_ts"].diff().dt.total_seconds().div(3600).fillna(-1.0)
    )

    # ---- Historical mean per sender (vectorized via cumsum/cumcount) ----
    # mean_at_row_i = sum_of_previous_rows / count_of_previous_rows
    # This is equivalent to expanding().mean().shift(1) but ~50x faster.
    logger.info("Computing historical mean per sender...")
    grp = df.groupby(sender_col, sort=False)
    cum_sum_inclusive = grp["amount"].cumsum()
    cum_count_inclusive = grp.cumcount() + 1  # 1-indexed count of rows so far

    # Exclude current row: subtract current value, decrement count
    cum_sum_prior = cum_sum_inclusive - df["amount"]
    cum_count_prior = cum_count_inclusive - 1

    df["sender_amount_mean_historical"] = np.where(
        cum_count_prior > 0,
        cum_sum_prior / cum_count_prior.replace(0, 1),  # avoid div-by-zero
        0.0,
    )

    # ---- Rolling 1h and 24h windows ----
    # The genuinely tricky ones. We use the indexed groupby-rolling pattern,
    # which is much faster than apply because pandas vectorizes per group.
    logger.info("Computing 1h transaction counts...")
    df["sender_txn_count_1h"] = _rolling_per_sender(df, sender_col, "1h", "count")

    logger.info("Computing 24h transaction counts...")
    df["sender_txn_count_24h"] = _rolling_per_sender(df, sender_col, "24h", "count")

    logger.info("Computing 24h amount sums...")
    df["sender_amount_sum_24h"] = _rolling_per_sender(df, sender_col, "24h", "sum")

    # Fill the NaNs from rolling on the first row of each sender
    df["sender_txn_count_1h"] = df["sender_txn_count_1h"].fillna(0).astype("int32")
    df["sender_txn_count_24h"] = df["sender_txn_count_24h"].fillna(0).astype("int32")
    df["sender_amount_sum_24h"] = df["sender_amount_sum_24h"].fillna(0.0)

    # ---- Ratios (per-row, trivially vectorized) ----
    logger.info("Computing balance ratios...")
    df["amount_to_oldbalance_ratio"] = (
        df["amount"] / df["oldbalanceOrg"].replace(0, np.nan)
    ).fillna(-1.0)

    df["drains_origin"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype("int8")

    # Restore chronological order across the whole dataset
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    # Drop helper columns
    drop_cols = [c for c in ("_ts", "nameDest") if c in df.columns]
    df = df.drop(columns=drop_cols)

    _assert_no_lookahead(df, sender_col)

    return df


def build_features_parquet(
    csv_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Engineer features from the raw PaySim CSV and write the cached parquet.

    This is the artifact `flows/retrain_model.py` trains on and that Feast's
    offline store is built from, so its schema is a contract:

    - the sender column is named ``sender`` (not ``nameOrig``), which is the
      Feast entity key;
    - ``event_timestamp`` is present, which is Feast's timestamp field.

    The notebook that originally produced this file did neither rename, so the
    committed parquet could not be reproduced from the committed code. Writing
    it from here closes that gap.

    Returns:
        The path written.
    """
    from fraud_mlops.config import DATA_INTERIM_DIR, PAYSIM_PATH
    from fraud_mlops.data import load_paysim

    csv_path = csv_path or PAYSIM_PATH
    out_path = out_path or DATA_INTERIM_DIR / "paysim_with_features.parquet"

    # drop_leaky=False keeps nameOrig, which the grouping needs. isFlaggedFraud
    # is dropped explicitly — it is PaySim's own rule, not a learned signal.
    df = load_paysim(csv_path, drop_leaky=False).drop(columns=["isFlaggedFraud"])
    logger.info("Engineering features for %d rows", len(df))

    out = engineer_features(df, sender_col="nameOrig")
    out = out.rename(columns={"nameOrig": "sender"})
    out["event_timestamp"] = pd.to_datetime(out[TIME_COLUMN], unit="h", origin=TIMESTAMP_ORIGIN)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows x %d cols to %s", len(out), out.shape[1], out_path)
    return out_path


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild the engineered-feature parquet")
    parser.add_argument("--csv", type=Path, default=None, help="Raw PaySim CSV")
    parser.add_argument("--out", type=Path, default=None, help="Output parquet path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_features_parquet(csv_path=args.csv, out_path=args.out)
