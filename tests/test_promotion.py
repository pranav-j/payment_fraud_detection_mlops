"""Tests for the promotion gate.

This is the one piece of logic that can change what production serves without a
human in the loop, so it gets the most careful tests in the suite. The cases
below are the four the gate exists to get right, plus both boundaries.
"""

from __future__ import annotations

from fraud_mlops.promotion import (
    MIN_RECALL_IMPROVEMENT,
    PRECISION_FLOOR,
    evaluate_gate,
)

# The incumbent in most of these cases: the registered production model's
# recall at its calibrated threshold.
PRODUCTION_RECALL = 0.8452


def test_refuses_a_weaker_candidate() -> None:
    """Lower recall at acceptable precision: production stays put."""
    decision = evaluate_gate(
        candidate_precision=0.9601,
        candidate_recall=PRODUCTION_RECALL - 0.05,
        production_recall=PRODUCTION_RECALL,
    )
    assert decision.promote is False
    assert "recall improvement" in decision.reason
    assert decision.recall_improvement < 0


def test_refuses_a_candidate_that_buys_recall_with_precision() -> None:
    """The expensive failure mode: +10 pp recall, but precision below the floor.

    A model like this declines more genuine customer payments. The gate must
    refuse it however good the recall looks.
    """
    decision = evaluate_gate(
        candidate_precision=0.90,
        candidate_recall=PRODUCTION_RECALL + 0.10,
        production_recall=PRODUCTION_RECALL,
    )
    assert decision.promote is False
    assert "below the floor" in decision.reason
    # Precision is checked first: the reason names the real objection.
    assert "recall improvement" not in decision.reason


def test_accepts_a_candidate_that_clears_both_conditions() -> None:
    decision = evaluate_gate(
        candidate_precision=0.9512,
        candidate_recall=PRODUCTION_RECALL + 0.03,
        production_recall=PRODUCTION_RECALL,
    )
    assert decision.promote is True
    assert decision.recall_improvement > MIN_RECALL_IMPROVEMENT


def test_boundary_recall_improvement_of_exactly_two_points_is_accepted() -> None:
    """Exactly +2 pp promotes — the threshold is inclusive.

    Expressed against a production recall of 0.0 so the subtraction is exact in
    floating point; 0.8452 -> 0.8652 differs from 0.02 in the last bits and
    would test float arithmetic rather than the gate.
    """
    decision = evaluate_gate(
        candidate_precision=PRECISION_FLOOR,
        candidate_recall=MIN_RECALL_IMPROVEMENT,
        production_recall=0.0,
    )
    assert decision.recall_improvement == MIN_RECALL_IMPROVEMENT
    assert decision.promote is True


def test_boundary_just_under_two_points_is_refused() -> None:
    decision = evaluate_gate(
        candidate_precision=0.99,
        candidate_recall=MIN_RECALL_IMPROVEMENT - 1e-9,
        production_recall=0.0,
    )
    assert decision.promote is False


def test_boundary_precision_of_exactly_95_percent_is_accepted() -> None:
    """Exactly 0.95 precision clears the floor — also inclusive."""
    decision = evaluate_gate(
        candidate_precision=PRECISION_FLOOR,
        candidate_recall=PRODUCTION_RECALL + 0.05,
        production_recall=PRODUCTION_RECALL,
    )
    assert decision.promote is True


def test_boundary_just_under_the_precision_floor_is_refused() -> None:
    decision = evaluate_gate(
        candidate_precision=PRECISION_FLOOR - 1e-9,
        candidate_recall=PRODUCTION_RECALL + 0.05,
        production_recall=PRODUCTION_RECALL,
    )
    assert decision.promote is False
    assert "below the floor" in decision.reason


def test_first_ever_model_promotes_against_no_incumbent() -> None:
    """With no production model, `production_recall=0.0` is passed in.

    Any candidate holding the precision floor then clears the improvement
    condition. This is the path `get_current_production_metrics` takes when the
    registry lookup fails.
    """
    decision = evaluate_gate(
        candidate_precision=0.9501,
        candidate_recall=0.5,
        production_recall=0.0,
    )
    assert decision.promote is True


def test_custom_thresholds_are_honoured() -> None:
    """Callers can tighten the gate; the defaults aren't baked into the logic."""
    kwargs = {
        "candidate_precision": 0.96,
        "candidate_recall": 0.90,
        "production_recall": 0.85,
    }
    assert evaluate_gate(**kwargs).promote is True
    assert evaluate_gate(**kwargs, min_recall_improvement=0.10).promote is False
    assert evaluate_gate(**kwargs, precision_floor=0.99).promote is False


def test_decision_serializes_for_logging() -> None:
    """The flow logs the decision, so it has to round-trip to a plain dict."""
    decision = evaluate_gate(
        candidate_precision=0.9501,
        candidate_recall=0.86,
        production_recall=0.8452,
    )
    payload = decision.to_dict()
    assert payload["promote"] is False
    assert payload["precision_floor"] == PRECISION_FLOOR
    assert set(payload) == {
        "promote",
        "reason",
        "candidate_precision",
        "candidate_recall",
        "production_recall",
        "recall_improvement",
        "precision_floor",
        "min_recall_improvement",
    }
