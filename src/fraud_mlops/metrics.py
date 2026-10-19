"""Metrics for imbalanced binary classification.

Accuracy is misleading at 0.13% fraud rate — predicting "always legit" gets
99.87% accuracy. The right metrics are precision-recall based, evaluated at
a calibrated threshold.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Evaluation metrics at a specific decision threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    pr_auc: float  # Area under precision-recall curve — threshold-independent
    roc_auc: float  # Area under ROC curve — threshold-independent

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def pretty_print(self) -> str:
        """Human-readable summary."""
        return (
            f"Threshold:     {self.threshold:.4f}\n"
            f"Precision:     {self.precision:.4f}  (of flagged, how many were fraud)\n"
            f"Recall:        {self.recall:.4f}  (of frauds, how many we caught)\n"
            f"F1:            {self.f1:.4f}\n"
            f"PR AUC:        {self.pr_auc:.4f}  (threshold-independent)\n"
            f"ROC AUC:       {self.roc_auc:.4f}  (threshold-independent)\n"
            f"Confusion:     TP={self.true_positives:,}  FP={self.false_positives:,}  "
            f"TN={self.true_negatives:,}  FN={self.false_negatives:,}"
        )


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> ClassificationMetrics:
    """Compute classification metrics at a given decision threshold.

    Args:
        y_true: Ground-truth labels (0 or 1).
        y_proba: Predicted probability of fraud (0..1).
        threshold: Decision boundary. Predict 1 if proba >= threshold.

    Returns:
        ClassificationMetrics dataclass.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ClassificationMetrics(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
    )


def find_threshold_for_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.95,
) -> float:
    """Find the decision threshold that achieves at least `min_precision`,
    while maximizing recall.

    Why this matters: in fraud, false positives are costly (you blocked a
    real customer's payment). Setting a precision floor and maximizing
    recall under that constraint encodes the business preference.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted probabilities.
        min_precision: Minimum acceptable precision.

    Returns:
        The threshold (in [0, 1]) to use for production decisions.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns N+1 precision/recall values for N thresholds —
    # the last entry is precision=1, recall=0 with no threshold. Slice to align.
    precision = precision[:-1]
    recall = recall[:-1]

    # Mask: thresholds that meet the precision floor.
    mask = precision >= min_precision

    if not mask.any():
        logger.warning(
            "No threshold achieves precision >= %.2f. "
            "Returning the highest-precision point (precision=%.4f).",
            min_precision,
            precision.max(),
        )
        return float(thresholds[precision.argmax()])

    # Among thresholds that meet the precision floor, pick the one with highest recall.
    valid_recalls = recall[mask]
    valid_thresholds = thresholds[mask]
    best_idx = valid_recalls.argmax()
    chosen = float(valid_thresholds[best_idx])

    logger.info(
        "Chose threshold=%.4f for min_precision=%.2f → actual precision=%.4f, recall=%.4f",
        chosen,
        min_precision,
        precision[mask][best_idx],
        recall[mask][best_idx],
    )
    return chosen


@dataclass
class PrSummary:
    """Threshold-free summary of a precision-recall curve."""

    pr_auc: float
    recall_at_min_precision: float
    min_precision: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ConfidenceInterval:
    """A bootstrap percentile interval."""

    point: float
    low: float
    high: float
    level: float
    n_resamples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


def _grouped_curve_inputs(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort by score descending and collapse ties into groups.

    Returns `(group_starts, y_sorted)`, where `group_starts` indexes the first
    row of each distinct score. Collapsing ties matters: a model that gives
    1,000 rows the same score cannot discriminate between them, and splitting
    them across thresholds would quietly inflate the curve.
    """
    order = np.argsort(-y_proba, kind="stable")
    p_sorted = y_proba[order]
    y_sorted = y_true[order].astype(np.int64)

    # First index of each run of equal scores.
    is_new = np.empty(len(p_sorted), dtype=bool)
    is_new[0] = True
    np.not_equal(p_sorted[1:], p_sorted[:-1], out=is_new[1:])
    group_starts = np.flatnonzero(is_new)

    return group_starts, y_sorted


def _curve_from_weights(
    group_starts: np.ndarray,
    y_sorted: np.ndarray,
    weights: np.ndarray | None,
    min_precision: float,
) -> tuple[float, float]:
    """Average precision and recall-at-min-precision for one (re)sample.

    `weights` is the multiplicity of each row — `None` means every row counts
    once, which is the point estimate.
    """
    if weights is None:
        positives = np.add.reduceat(y_sorted, group_starts)
        totals = np.diff(np.append(group_starts, len(y_sorted)))
    else:
        positives = np.add.reduceat(weights * y_sorted, group_starts)
        totals = np.add.reduceat(weights, group_starts)

    tp = np.cumsum(positives)
    seen = np.cumsum(totals)
    total_positives = tp[-1]

    if total_positives == 0:
        return 0.0, 0.0

    precision = tp / np.maximum(seen, 1)
    recall = tp / total_positives

    # Average precision: sum of precision weighted by the gain in recall.
    average_precision = float(np.sum(np.diff(recall, prepend=0.0) * precision))

    eligible = precision >= min_precision
    recall_at_floor = float(recall[eligible].max()) if eligible.any() else 0.0

    return average_precision, recall_at_floor


def summarize_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.95,
) -> PrSummary:
    """PR-AUC and the best recall available at or above `min_precision`.

    Both are threshold-free properties of the ranking. `recall_at_min_precision`
    is an upper bound on what any threshold could achieve on *this* data — it
    is not what the deployed threshold delivers, because that threshold was
    chosen on the validation set. Report both; they answer different questions.
    """
    group_starts, y_sorted = _grouped_curve_inputs(y_true, y_proba)
    pr_auc, recall_at_floor = _curve_from_weights(group_starts, y_sorted, None, min_precision)
    return PrSummary(
        pr_auc=pr_auc,
        recall_at_min_precision=recall_at_floor,
        min_precision=min_precision,
    )


def bootstrap_pr_summary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.95,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> dict[str, ConfidenceInterval]:
    """Percentile bootstrap intervals for PR-AUC and recall-at-min-precision.

    Resamples rows with replacement `n_resamples` times. With ~4,000 fraud cases
    in a test split, the interval on recall is a couple of points wide — which
    is the point of reporting it: a 1-point difference between two models is
    not a difference.

    Implementation note: the rows are sorted by score once, and each resample is
    drawn as a multiplicity vector over those sorted rows. That keeps every
    resample to a handful of linear passes instead of a re-sort.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    group_starts, y_sorted = _grouped_curve_inputs(y_true, y_proba)
    point_pr_auc, point_recall = _curve_from_weights(group_starts, y_sorted, None, min_precision)

    n = len(y_sorted)
    rng = np.random.default_rng(seed)
    pr_aucs = np.empty(n_resamples)
    recalls = np.empty(n_resamples)

    for i in range(n_resamples):
        # Multiplicities of a with-replacement resample of n rows. Rows are
        # exchangeable under the draw, so these can be used directly against
        # the score-sorted arrays without a reindex.
        weights = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.int64)
        pr_aucs[i], recalls[i] = _curve_from_weights(group_starts, y_sorted, weights, min_precision)

    tail = (1.0 - level) / 2.0 * 100.0
    return {
        "pr_auc": ConfidenceInterval(
            point=point_pr_auc,
            low=float(np.percentile(pr_aucs, tail)),
            high=float(np.percentile(pr_aucs, 100 - tail)),
            level=level,
            n_resamples=n_resamples,
        ),
        "recall_at_min_precision": ConfidenceInterval(
            point=point_recall,
            low=float(np.percentile(recalls, tail)),
            high=float(np.percentile(recalls, 100 - tail)),
            level=level,
            n_resamples=n_resamples,
        ),
    }
