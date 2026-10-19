"""Honest evaluation of the fraud models.

    make evaluate            # full dataset, writes reports/metrics/
    python -m fraud_mlops.evaluate --sample-frac 0.1   # quick pass

What this script exists to fix. The numbers the README used to quote came from
notebooks that:

- split 80/20 by row index after sorting by `step`, so one simulated hour sat on
  both sides of the boundary;
- had no validation set at all, and so tuned the decision threshold on the very
  test set they then reported from;
- reported a single point estimate with no interval, so a 4-point difference
  between two models looked like a result;
- quoted recall and precision but never PR-AUC, the one number that does not
  depend on where the threshold landed;
- and kept PaySim's balance columns, which encode the simulator's own fraud
  rule almost deterministically.

Everything here follows one protocol, applied identically to every model:

1. **Split by time, on whole steps.** Train on the earliest hours, validate on
   the next, test on the latest. A random split is also reported, as a
   secondary comparison, because `flows/retrain_model.py` uses one.
2. **Tune on validation, report on test.** The threshold maximises recall
   subject to precision >= 95% on validation; every reported number comes from
   the test split, which nothing was tuned on.
3. **Interval, not just a point.** Percentile bootstrap over the test rows for
   PR-AUC and for recall at 95% precision.
4. **Same splits for every baseline**, so the ladder from "a rule anyone could
   write" to "XGBoost on engineered features" is a fair one.

Outputs `reports/metrics/evaluation.json` (the source of truth for every number
in the README) and `reports/metrics/evaluation.md` (the readable version).

Runs are logged to a **local** MLflow file store, never to the tracking server
and never to the model registry: this script evaluates, it does not promote.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_mlops.config import (
    ID_COLUMNS,
    LABEL_COLUMN,
    LEAKY_COLUMNS,
    PROJECT_ROOT,
    RANDOM_SEED,
    TIME_COLUMN,
)
from fraud_mlops.data import time_based_split_three_way
from fraud_mlops.features import (
    BALANCE_COLUMNS,
    BASELINE_FEATURE_COLUMNS,
    ENRICHED_FEATURE_COLUMNS,
)
from fraud_mlops.metrics import (
    bootstrap_pr_summary,
    evaluate_at_threshold,
    find_threshold_for_precision,
    summarize_pr_curve,
)
from fraud_mlops.training import (
    build_logistic_pipeline,
    build_xgb_pipeline,
    compute_scale_pos_weight,
)

logger = logging.getLogger(__name__)

# The business constraint the threshold is chosen under: false positives are
# declined payments for real customers, so precision is a floor.
MIN_PRECISION = 0.95

# Fraud only ever occurs in these two PaySim transaction types.
FRAUD_ELIGIBLE_TYPES = ("TRANSFER", "CASH_OUT")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "metrics"

# A file store under the repo. Deliberately not the env-configured tracking
# server: nothing this script does should be able to reach the registry.
EVAL_TRACKING_DIR = PROJECT_ROOT / "mlruns_evaluation"
EVAL_EXPERIMENT = "fraud_evaluation"


# ── Result containers ───────────────────────────────────────────────────────


@dataclass
class SplitSummary:
    """Row counts, time ranges and fraud counts for one split."""

    name: str
    n_rows: int
    n_fraud: int
    fraud_rate: float
    step_min: int
    step_max: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelResult:
    """Everything reported about one model on one split scheme."""

    name: str
    description: str
    feature_set: str
    n_features: int
    split_scheme: str

    # Chosen on validation, never on test.
    threshold: float
    validation_precision: float
    validation_recall: float

    # Measured on test.
    pr_auc: float
    pr_auc_ci_low: float
    pr_auc_ci_high: float
    recall_at_95_precision: float
    recall_at_95_precision_ci_low: float
    recall_at_95_precision_ci_high: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    # False when no threshold on the validation set could reach the precision
    # floor. The threshold is then a fallback (the highest-precision point), and
    # the test numbers at it do not describe an operating point anyone would
    # deploy. Reported rather than hidden.
    threshold_met_precision_floor: bool = True

    train_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """The whole run, serialized to evaluation.json."""

    generated_at: str
    dataset_rows: int
    dataset_fraud: int
    sample_frac: float | None
    random_seed: int
    n_bootstrap: int
    min_precision: float
    excluded_columns: dict[str, str]
    splits: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    leakage: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Data loading and splitting ──────────────────────────────────────────────


def load_engineered(
    parquet_path: Path | None = None,
    sample_frac: float | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Load transactions with the enriched features already computed.

    Reads the cached parquet when it exists, otherwise builds it from the raw
    CSV. Either way the features come from `fraud_mlops.features`, so this
    cannot diverge from what the retrain flow trains on.
    """
    from fraud_mlops.config import DATA_INTERIM_DIR
    from fraud_mlops.features import build_features_parquet

    parquet_path = parquet_path or DATA_INTERIM_DIR / "paysim_with_features.parquet"

    if not parquet_path.exists():
        logger.info("No cached features at %s — building from the raw CSV", parquet_path)
        build_features_parquet(out_path=parquet_path)

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from %s", len(df), parquet_path)

    missing = set(ENRICHED_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{parquet_path} is missing engineered columns {sorted(missing)}. "
            f"Rebuild it with `python -m fraud_mlops.features`."
        )

    if sample_frac is not None:
        # Sample whole time steps, not rows: sampling rows would thin out each
        # hour and quietly change what a "velocity" feature means relative to
        # the split boundaries.
        steps = np.sort(df[TIME_COLUMN].unique())
        keep = steps[: max(3, int(len(steps) * sample_frac))]
        df = df[df[TIME_COLUMN].isin(keep)]
        logger.info("Sampled to %d rows (%d steps)", len(df), len(keep))

    return df.reset_index(drop=True)


def summarize_split(name: str, part: pd.DataFrame) -> SplitSummary:
    return SplitSummary(
        name=name,
        n_rows=len(part),
        n_fraud=int(part[LABEL_COLUMN].sum()),
        fraud_rate=float(part[LABEL_COLUMN].mean()),
        step_min=int(part[TIME_COLUMN].min()),
        step_max=int(part[TIME_COLUMN].max()),
    )


def make_splits(
    df: pd.DataFrame,
    scheme: str,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way split, either by time (primary) or at random (secondary)."""
    if scheme == "time":
        return time_based_split_three_way(df, val_fraction=0.2, test_fraction=0.2)

    if scheme == "random":
        # Stratified, matching what flows/retrain_model.py does, extended to a
        # third split so the protocol is otherwise identical.
        from sklearn.model_selection import train_test_split

        train, holdout = train_test_split(
            df, test_size=0.4, random_state=seed, stratify=df[LABEL_COLUMN]
        )
        val, test = train_test_split(
            holdout, test_size=0.5, random_state=seed, stratify=holdout[LABEL_COLUMN]
        )
        return train, val, test

    raise ValueError(f"Unknown split scheme {scheme!r}; expected 'time' or 'random'")


# ── Scorers ─────────────────────────────────────────────────────────────────


def amount_rule_scores(df: pd.DataFrame) -> np.ndarray:
    """Score for the rule baseline: the amount, but only for risky types.

    "Flag large TRANSFER and CASH_OUT transactions" is the rule a fraud analyst
    would write on day one without any model at all. Scoring by amount (and
    zero for the three types where PaySim never puts fraud) turns it into a
    ranking, so it gets a PR curve and a tuned cut-off on the same footing as
    the models.
    """
    eligible = df["type"].isin(FRAUD_ELIGIBLE_TYPES).to_numpy()
    scores: np.ndarray = np.where(eligible, df["amount"].to_numpy(), 0.0)
    return scores


def balance_signature_scores(df: pd.DataFrame) -> np.ndarray:
    """PaySim's fraud signature, written out by hand as a three-condition rule.

    A fraudulent PaySim transaction is a TRANSFER or CASH_OUT that moves exactly
    the origin account's whole balance, leaving it at zero. This scorer returns
    1.0 for rows matching that pattern and 0.0 otherwise. It is not a baseline
    anyone should ship — it is the measurement that shows how much of the task
    the balance columns simply give away.
    """
    eligible = df["type"].isin(FRAUD_ELIGIBLE_TYPES).to_numpy()
    drains = np.isclose(df["amount"].to_numpy(), df["oldbalanceOrg"].to_numpy()) & (
        df["newbalanceOrig"].to_numpy() == 0
    )
    scores: np.ndarray = np.asarray(eligible & drains, dtype=float)
    return scores


# ── Evaluation of one model ─────────────────────────────────────────────────


def _evaluate_scores(
    name: str,
    description: str,
    feature_set: str,
    n_features: int,
    split_scheme: str,
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    n_bootstrap: int,
    seed: int,
    train_seconds: float = 0.0,
    notes: str = "",
    min_precision: float = MIN_PRECISION,
) -> ModelResult:
    """Tune on validation scores, then report everything from test scores."""
    threshold = find_threshold_for_precision(y_val, val_scores, min_precision=min_precision)
    val_metrics = evaluate_at_threshold(y_val, val_scores, threshold)
    test_metrics = evaluate_at_threshold(y_test, test_scores, threshold)

    curve = summarize_pr_curve(y_test, test_scores, min_precision=min_precision)
    intervals = bootstrap_pr_summary(
        y_test,
        test_scores,
        min_precision=min_precision,
        n_resamples=n_bootstrap,
        seed=seed,
    )

    return ModelResult(
        name=name,
        description=description,
        feature_set=feature_set,
        n_features=n_features,
        split_scheme=split_scheme,
        threshold=float(threshold),
        validation_precision=val_metrics.precision,
        validation_recall=val_metrics.recall,
        pr_auc=curve.pr_auc,
        pr_auc_ci_low=intervals["pr_auc"].low,
        pr_auc_ci_high=intervals["pr_auc"].high,
        recall_at_95_precision=curve.recall_at_min_precision,
        recall_at_95_precision_ci_low=intervals["recall_at_min_precision"].low,
        recall_at_95_precision_ci_high=intervals["recall_at_min_precision"].high,
        precision_at_threshold=test_metrics.precision,
        recall_at_threshold=test_metrics.recall,
        f1_at_threshold=test_metrics.f1,
        true_positives=test_metrics.true_positives,
        false_positives=test_metrics.false_positives,
        false_negatives=test_metrics.false_negatives,
        true_negatives=test_metrics.true_negatives,
        threshold_met_precision_floor=bool(val_metrics.precision >= min_precision),
        train_seconds=train_seconds,
        notes=notes,
    )


def evaluate_model(
    name: str,
    description: str,
    feature_set: str,
    features: list[str],
    builder: str,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    split_scheme: str,
    n_bootstrap: int,
    seed: int,
    notes: str = "",
) -> ModelResult:
    """Fit one model on train, calibrate on validation, report on test."""
    train_df, val_df, test_df = splits
    y_train = train_df[LABEL_COLUMN].to_numpy()

    if builder == "xgboost":
        model = build_xgb_pipeline(
            feature_columns=features,
            scale_pos_weight=compute_scale_pos_weight(train_df[LABEL_COLUMN]),
        )
    elif builder == "logistic":
        model = build_logistic_pipeline(features)
    else:
        raise ValueError(f"Unknown builder {builder!r}")

    logger.info("[%s / %s] fitting on %d rows", split_scheme, name, len(train_df))
    started = time.perf_counter()
    model.fit(train_df[features], y_train)
    train_seconds = time.perf_counter() - started
    logger.info("[%s / %s] fitted in %.1fs", split_scheme, name, train_seconds)

    return _evaluate_scores(
        name=name,
        description=description,
        feature_set=feature_set,
        n_features=len(features),
        split_scheme=split_scheme,
        y_val=val_df[LABEL_COLUMN].to_numpy(),
        val_scores=model.predict_proba(val_df[features])[:, 1],
        y_test=test_df[LABEL_COLUMN].to_numpy(),
        test_scores=model.predict_proba(test_df[features])[:, 1],
        n_bootstrap=n_bootstrap,
        seed=seed,
        train_seconds=train_seconds,
        notes=notes,
    )


def evaluate_rule(
    name: str,
    description: str,
    scorer: Any,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    split_scheme: str,
    n_bootstrap: int,
    seed: int,
    notes: str = "",
) -> ModelResult:
    """Evaluate a hand-written rule on exactly the same footing as a model."""
    _, val_df, test_df = splits
    return _evaluate_scores(
        name=name,
        description=description,
        feature_set="rule",
        n_features=0,
        split_scheme=split_scheme,
        y_val=val_df[LABEL_COLUMN].to_numpy(),
        val_scores=scorer(val_df),
        y_test=test_df[LABEL_COLUMN].to_numpy(),
        test_scores=scorer(test_df),
        n_bootstrap=n_bootstrap,
        seed=seed,
        notes=notes,
    )


# ── MLflow logging (local file store only) ──────────────────────────────────


def _log_to_local_mlflow(results: list[ModelResult]) -> str | None:
    """Log every result to a local file store. Never touches the registry."""
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        logger.warning("mlflow not installed; skipping experiment logging")
        return None

    EVAL_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    uri = EVAL_TRACKING_DIR.as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EVAL_EXPERIMENT)

    for result in results:
        with mlflow.start_run(run_name=f"{result.split_scheme}:{result.name}"):
            mlflow.log_params(
                {
                    "model": result.name,
                    "feature_set": result.feature_set,
                    "n_features": result.n_features,
                    "split_scheme": result.split_scheme,
                    "min_precision": MIN_PRECISION,
                    "random_seed": RANDOM_SEED,
                }
            )
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in result.to_dict().items()
                    if isinstance(v, int | float) and not isinstance(v, bool)
                }
            )
    logger.info("Logged %d runs to the local MLflow store at %s", len(results), uri)
    return uri


# ── Reporting ───────────────────────────────────────────────────────────────


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_ci(point: float, low: float, high: float, as_pct: bool = True) -> str:
    if as_pct:
        return f"{_fmt_pct(point)} [{_fmt_pct(low)}, {_fmt_pct(high)}]"
    return f"{point:.4f} [{low:.4f}, {high:.4f}]"


def render_markdown(report: EvaluationReport) -> str:
    """The human-readable twin of evaluation.json."""
    lines: list[str] = []
    add = lines.append

    add("# Evaluation")
    add("")
    add(
        f"Generated {report.generated_at} by `python -m fraud_mlops.evaluate` "
        f"(`make evaluate`). Every number in the README's evaluation section comes "
        f"from the JSON alongside this file."
    )
    add("")
    add(
        f"Dataset: **{report.dataset_rows:,} transactions**, "
        f"{report.dataset_fraud:,} fraudulent "
        f"({report.dataset_fraud / max(report.dataset_rows, 1):.4%})."
    )
    if report.sample_frac:
        add("")
        add(
            f"> **Partial run.** This report covers a {report.sample_frac:.0%} sample of "
            f"time steps, not the full dataset. Run `make evaluate` for the real numbers."
        )
    add("")

    # --- protocol ---
    add("## Protocol")
    add("")
    add(
        "- **Split by time, on whole steps.** PaySim's `step` is a simulated hour. "
        "Train on the earliest hours, validate on the next block, test on the latest. "
        "A step is never divided between two splits."
    )
    add(
        f"- **Threshold tuned on validation only**, to maximise recall subject to "
        f"precision >= {report.min_precision:.0%}. Test is scored at that threshold "
        f"and is never used to choose it."
    )
    add(
        f"- **Intervals** are percentile bootstraps over the test rows, "
        f"{report.n_bootstrap:,} resamples, 95% level."
    )
    add(f"- **Seed** {report.random_seed} throughout.")
    add("")

    for scheme, splits in report.splits.items():
        label = "Time-based split (primary)" if scheme == "time" else "Random split (secondary)"
        add(f"### {label}")
        add("")
        add("| Split | Rows | Steps | Fraud | Fraud rate |")
        add("|---|---:|---|---:|---:|")
        for s in splits:
            steps = f"{s['step_min']}–{s['step_max']}"
            add(
                f"| {s['name']} | {s['n_rows']:,} | {steps} | {s['n_fraud']:,} | "
                f"{s['fraud_rate']:.4%} |"
            )
        add("")

    if "time" in report.splits:
        add(
            "Note how lopsided the time split looks against the row counts: PaySim's "
            "transaction volume is heavily front-loaded, so 60% of *rows* is a much "
            "smaller share of *hours*. The fraud rate also differs between splits, "
            "which is a property of the simulator rather than of any model."
        )
        add("")

    # --- results ---
    for scheme in report.splits:
        # The leakage experiment has its own section below; mixing it into the
        # headline ladder would invite reading a diagnostic as a candidate model.
        rows = [
            r
            for r in report.results
            if r["split_scheme"] == scheme and r["notes"] != "leakage experiment"
        ]
        if not rows:
            continue
        label = "Time-based split (primary)" if scheme == "time" else "Random split (secondary)"
        add(f"## Results — {label}")
        add("")
        add(
            "| Model | Features | PR-AUC [95% CI] | Recall @95% precision [95% CI] | "
            "Precision @ threshold | Recall @ threshold | Threshold |"
        )
        add("|---|---:|---|---|---:|---:|---:|")
        for r in rows:
            marker = "" if r["threshold_met_precision_floor"] else " †"
            add(
                f"| {r['name']}{marker} | {r['n_features'] or '—'} | "
                f"{_fmt_ci(r['pr_auc'], r['pr_auc_ci_low'], r['pr_auc_ci_high'], as_pct=False)} | "
                f"{_fmt_ci(r['recall_at_95_precision'], r['recall_at_95_precision_ci_low'], r['recall_at_95_precision_ci_high'])} | "
                f"{_fmt_pct(r['precision_at_threshold'])} | "
                f"{_fmt_pct(r['recall_at_threshold'])} | "
                f"{r['threshold']:.4f} |"
            )
        add("")
        add(
            "**Recall @95% precision** is the best recall the ranking could reach on "
            "test at a 95% precision floor — an upper bound, found with hindsight. "
            "**Recall @ threshold** is what the deployed cut-off actually delivers, "
            "having been chosen on validation. The gap between the two columns is the "
            "cost of not being able to tune on the future."
        )
        if any(r["feature_set"] == "rule" for r in rows):
            add("")
            add(
                "A rule's threshold is in currency units, not a probability — the rule "
                "ranks by amount, so its tuned cut-off is an amount."
            )
        if any(not r["threshold_met_precision_floor"] for r in rows):
            add("")
            add(
                "† No threshold on the validation set reached 95% precision for this "
                "row, so its threshold is a fallback (the highest-precision point "
                "available) rather than a calibrated operating point. Read its "
                "@-threshold columns as 'the best this could do', not as a deployable "
                "configuration."
            )
        add("")

    # --- leakage ---
    add("## Leakage and excluded columns")
    add("")
    add("| Column | Why it is excluded |")
    add("|---|---|")
    for column, reason in report.excluded_columns.items():
        add(f"| `{column}` | {reason} |")
    add("")

    leak = report.leakage
    if leak:
        add("### What the balance columns are worth")
        add("")
        add(leak.get("narrative", ""))
        add("")
        add("| Configuration | PR-AUC [95% CI] | Recall @95% precision [95% CI] |")
        add("|---|---|---|")
        for row in leak.get("comparison", []):
            add(
                f"| {row['name']} | "
                f"{_fmt_ci(row['pr_auc'], row['pr_auc_ci_low'], row['pr_auc_ci_high'], as_pct=False)} | "
                f"{_fmt_ci(row['recall_at_95_precision'], row['recall_at_95_precision_ci_low'], row['recall_at_95_precision_ci_high'])} |"
            )
        add("")
        if leak.get("signature_stats"):
            stats = leak["signature_stats"]
            add(
                f"The hand-written balance signature — a TRANSFER or CASH_OUT that moves "
                f"exactly the origin account's whole balance, leaving it at zero — matches "
                f"**{stats['fraud_matched']:,} of {stats['fraud_total']:,} fraudulent "
                f"transactions ({stats['fraud_share']:.2%})** and "
                f"**{stats['legit_matched']:,} of {stats['legit_total']:,} legitimate ones "
                f"({stats['legit_share']:.4%})** across the whole dataset."
            )
            add("")

    add("## Environment")
    add("")
    for key, value in report.environment.items():
        add(f"- **{key}:** {value}")
    add("")
    add(f"Total runtime: {report.runtime_seconds:.0f}s.")
    add("")

    return "\n".join(lines)


# ── Orchestration ───────────────────────────────────────────────────────────

EXCLUDED_COLUMNS = {
    "isFlaggedFraud": (
        "PaySim's own rule-based flag (transfers over 200,000). It is the simulator's "
        "output, not an observable input — a model that sees it learns the rule."
    ),
    "nameOrig": (
        "Sender account ID. Used to *group* by when computing history, never as a "
        "feature: with 6.3M near-unique values a tree would memorise individuals."
    ),
    "nameDest": "Destination account ID. Same reasoning as `nameOrig`.",
}

# The reasons above are written by hand, but *which* columns are excluded is not:
# it comes from `config`, so a column added to either list there and left
# undocumented here fails loudly instead of quietly disappearing from the report.
_undocumented = set(LEAKY_COLUMNS) | set(ID_COLUMNS) - set(EXCLUDED_COLUMNS)
if _undocumented - set(EXCLUDED_COLUMNS):
    raise AssertionError(
        f"Columns excluded in config but not explained in EXCLUDED_COLUMNS: "
        f"{sorted(_undocumented - set(EXCLUDED_COLUMNS))}"
    )


def run_evaluation(
    parquet_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_bootstrap: int = 1000,
    sample_frac: float | None = None,
    schemes: tuple[str, ...] = ("time", "random"),
    seed: int = RANDOM_SEED,
    skip_mlflow: bool = False,
) -> EvaluationReport:
    """Run the whole protocol and write the reports."""
    started = time.perf_counter()
    df = load_engineered(parquet_path, sample_frac=sample_frac, seed=seed)

    report = EvaluationReport(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        dataset_rows=len(df),
        dataset_fraud=int(df[LABEL_COLUMN].sum()),
        sample_frac=sample_frac,
        random_seed=seed,
        n_bootstrap=n_bootstrap,
        min_precision=MIN_PRECISION,
        excluded_columns=EXCLUDED_COLUMNS,
        environment={
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "features_source": str(parquet_path or "data/interim/paysim_with_features.parquet"),
        },
    )

    results: list[ModelResult] = []

    for scheme in schemes:
        splits = make_splits(df, scheme, seed=seed)
        report.splits[scheme] = [
            summarize_split(name, part).to_dict()
            for name, part in zip(("train", "validation", "test"), splits, strict=True)
        ]

        results.append(
            evaluate_rule(
                name="Rule: large TRANSFER/CASH_OUT",
                description=(
                    "Flag transfers and cash-outs above an amount threshold tuned on "
                    "validation. No model, no training."
                ),
                scorer=amount_rule_scores,
                splits=splits,
                split_scheme=scheme,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )
        results.append(
            evaluate_model(
                name="Logistic regression (v1 features)",
                description="Balanced logistic regression on the raw PaySim columns.",
                feature_set="baseline_v1",
                features=BASELINE_FEATURE_COLUMNS,
                builder="logistic",
                splits=splits,
                split_scheme=scheme,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )
        results.append(
            evaluate_model(
                name="XGBoost (v1 features)",
                description="The notebook-02 baseline: XGBoost on the raw PaySim columns.",
                feature_set="baseline_v1",
                features=BASELINE_FEATURE_COLUMNS,
                builder="xgboost",
                splits=splits,
                split_scheme=scheme,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )
        results.append(
            evaluate_model(
                name="XGBoost (v2 features)",
                description=(
                    "The production configuration: XGBoost on the raw columns plus the "
                    "seven engineered sender-history features."
                ),
                feature_set="enriched_v1",
                features=ENRICHED_FEATURE_COLUMNS,
                builder="xgboost",
                splits=splits,
                split_scheme=scheme,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )

    # --- leakage experiment, on the primary split only ---
    if "time" in schemes:
        splits = make_splits(df, "time", seed=seed)
        without_balance = [c for c in ENRICHED_FEATURE_COLUMNS if c not in BALANCE_COLUMNS]

        no_balance = evaluate_model(
            name="XGBoost (v2 features, balance columns removed)",
            description=(
                "The same model with the four balance columns and the two features "
                "derived from them held out."
            ),
            feature_set="enriched_v1_no_balance",
            features=without_balance,
            builder="xgboost",
            splits=splits,
            split_scheme="time",
            n_bootstrap=n_bootstrap,
            seed=seed,
            notes="leakage experiment",
        )
        signature = evaluate_rule(
            name="Rule: PaySim balance signature",
            description=(
                "A three-condition hand rule on the balance columns: TRANSFER/CASH_OUT "
                "moving exactly the whole origin balance, leaving it at zero."
            ),
            scorer=balance_signature_scores,
            splits=splits,
            split_scheme="time",
            n_bootstrap=n_bootstrap,
            seed=seed,
            notes="leakage experiment",
        )

        with_balance = next(
            r for r in results if r.name == "XGBoost (v2 features)" and r.split_scheme == "time"
        )

        matches = balance_signature_scores(df) > 0
        is_fraud = df[LABEL_COLUMN].to_numpy() == 1
        report.leakage = {
            "narrative": (
                "PaySim generates fraud by emptying the origin account, so the balance "
                "columns do not merely correlate with the label — they encode the "
                "simulator's rule. The comparison below is the honest way to show it: "
                "the same model, same splits, with and without those columns, against a "
                "hand-written rule that uses nothing else. A real deployment has no "
                "column this clean, which is why the no-balance row is the better guide "
                "to what this system would do on real payments."
            ),
            "comparison": [
                with_balance.to_dict(),
                no_balance.to_dict(),
                signature.to_dict(),
            ],
            "excluded_feature_columns": BALANCE_COLUMNS,
            "signature_stats": {
                "fraud_matched": int((matches & is_fraud).sum()),
                "fraud_total": int(is_fraud.sum()),
                "fraud_share": float((matches & is_fraud).sum() / max(is_fraud.sum(), 1)),
                "legit_matched": int((matches & ~is_fraud).sum()),
                "legit_total": int((~is_fraud).sum()),
                "legit_share": float((matches & ~is_fraud).sum() / max((~is_fraud).sum(), 1)),
            },
        }
        results.extend([no_balance, signature])

    report.results = [r.to_dict() for r in results]
    report.runtime_seconds = time.perf_counter() - started

    if not skip_mlflow:
        uri = _log_to_local_mlflow(results)
        if uri:
            report.environment["mlflow_store"] = uri

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation.json"
    md_path = output_dir / "evaluation.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    md_path.write_text(render_markdown(report))
    logger.info("Wrote %s and %s", json_path, md_path)

    return report


def render_only(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Rewrite evaluation.md from an existing evaluation.json.

    The JSON is the source of truth, so the prose around the numbers can be
    improved without re-running three minutes of model fits.
    """
    json_path = output_dir / "evaluation.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} does not exist — run `make evaluate` first")

    payload = json.loads(json_path.read_text())
    report = EvaluationReport(**payload)
    md_path = output_dir / "evaluation.md"
    md_path.write_text(render_markdown(report))
    logger.info("Re-rendered %s from %s", md_path, json_path)
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parquet", type=Path, default=None, help="Engineered-feature parquet")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Bootstrap resamples (default 1000)"
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Evaluate on this fraction of time steps — for a quick check, not for reporting",
    )
    parser.add_argument(
        "--splits",
        default="time,random",
        help="Comma-separated split schemes to run (default: time,random)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-mlflow", action="store_true", help="Don't log runs to MLflow")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Rewrite evaluation.md from the existing evaluation.json and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.render_only:
        path = render_only(args.out)
        print(f"Re-rendered {path}")
        return

    report = run_evaluation(
        parquet_path=args.parquet,
        output_dir=args.out,
        n_bootstrap=args.bootstrap,
        sample_frac=args.sample_frac,
        schemes=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        seed=args.seed,
        skip_mlflow=args.skip_mlflow,
    )

    print(f"\nWrote {args.out}/evaluation.json and {args.out}/evaluation.md")
    print(f"Ran {len(report.results)} evaluations in {report.runtime_seconds:.0f}s\n")
    for row in report.results:
        print(
            f"  {row['split_scheme']:>6}  {row['name']:<50} "
            f"PR-AUC {row['pr_auc']:.4f}  "
            f"recall@95%P {row['recall_at_95_precision']:.4f}  "
            f"recall@threshold {row['recall_at_threshold']:.4f}"
        )


if __name__ == "__main__":
    main()
