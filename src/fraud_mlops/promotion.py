"""The promotion gate: does a freshly trained candidate replace production?

The decision used to be inline in `flows/retrain_model.py`, inside a Prefect
task, which meant it could not be unit-tested without a Prefect runtime and an
MLflow server. It is lifted out here as a pure function.

The rule is unchanged from the flow it came from, and deliberately so — see the
non-goals in `CLAUDE.md`:

    promote  <=>  candidate precision >= PRECISION_FLOOR
                  AND (candidate recall - production recall) >= MIN_RECALL_IMPROVEMENT

Why this shape: false positives are expensive (you declined a real customer's
payment), so precision is a floor, not something to trade away. And a candidate
has to be *meaningfully* better to justify a production change, not merely
better by noise — hence 2 percentage points rather than "any improvement".
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The precision a candidate must hold to be promotable at all.
PRECISION_FLOOR: float = 0.95

# How much recall a candidate must add, in absolute terms, over the incumbent.
MIN_RECALL_IMPROVEMENT: float = 0.02


@dataclass(frozen=True)
class GateDecision:
    """The outcome of one promotion-gate evaluation.

    Frozen, and carries the inputs alongside the verdict, so the flow can log a
    complete record of *why* a promotion did or didn't happen.
    """

    promote: bool
    reason: str
    candidate_precision: float
    candidate_recall: float
    production_recall: float
    recall_improvement: float
    precision_floor: float
    min_recall_improvement: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_gate(
    candidate_precision: float,
    candidate_recall: float,
    production_recall: float,
    precision_floor: float = PRECISION_FLOOR,
    min_recall_improvement: float = MIN_RECALL_IMPROVEMENT,
) -> GateDecision:
    """Decide whether `candidate` should take the `@production` alias.

    Both conditions are inclusive at the boundary: a candidate sitting exactly
    on the precision floor, or improving recall by exactly the minimum, is
    promoted. The alternative — rejecting at the boundary — would make the
    documented thresholds off-by-one against the code.

    Args:
        candidate_precision: Candidate precision at its calibrated threshold.
        candidate_recall: Candidate recall at its calibrated threshold.
        production_recall: Recall of the incumbent `@production` version. Pass
            0.0 when there is no incumbent — any candidate meeting the
            precision floor then clears the improvement condition.
        precision_floor: Minimum acceptable precision.
        min_recall_improvement: Minimum absolute recall gain required.

    Returns:
        A `GateDecision`. Callers should act only on `.promote`.
    """
    recall_improvement = candidate_recall - production_recall
    precision_ok = candidate_precision >= precision_floor
    recall_ok = recall_improvement >= min_recall_improvement

    if not precision_ok:
        reason = (
            f"precision {candidate_precision:.4f} is below the floor {precision_floor:.4f} — "
            f"a candidate that declines more genuine payments is not an improvement"
        )
    elif not recall_ok:
        reason = (
            f"recall improvement {recall_improvement:+.4f} is below the required "
            f"{min_recall_improvement:.4f} (candidate {candidate_recall:.4f} vs "
            f"production {production_recall:.4f})"
        )
    else:
        reason = (
            f"precision {candidate_precision:.4f} holds the floor and recall improves by "
            f"{recall_improvement:+.4f} (candidate {candidate_recall:.4f} vs "
            f"production {production_recall:.4f})"
        )

    decision = GateDecision(
        promote=precision_ok and recall_ok,
        reason=reason,
        candidate_precision=candidate_precision,
        candidate_recall=candidate_recall,
        production_recall=production_recall,
        recall_improvement=recall_improvement,
        precision_floor=precision_floor,
        min_recall_improvement=min_recall_improvement,
    )

    logger.info("Promotion gate: %s — %s", "PROMOTE" if decision.promote else "HOLD", reason)
    return decision
