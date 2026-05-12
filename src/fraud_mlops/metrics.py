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
        "Chose threshold=%.4f for min_precision=%.2f → "
        "actual precision=%.4f, recall=%.4f",
        chosen,
        min_precision,
        precision[mask][best_idx],
        recall[mask][best_idx],
    )
    return chosen
