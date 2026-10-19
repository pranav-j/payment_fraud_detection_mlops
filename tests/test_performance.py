"""Tests for realized-performance monitoring under label delay.

Four things have to be right here, and each of them is a way the monitoring
could lie rather than merely be wrong:

1. **The maturity join.** A label that has not matured must be absent, not
   joined as a null that something downstream reads as "not fraud".
2. **Window boundaries.** Windows must be stable across runs and must not
   overlap, or two consecutive daily runs are not comparable.
3. **Delay handling.** `as_of` moving forward must bring labels into scope and
   never take them out.
4. **The alert rule.** It must fire on a genuine recall drop and must *not* fire
   because labels have not arrived yet — the newest window is always empty, and
   an alert that cries wolf every night gets muted.

No database and no AWS: the logic takes dataframes and returns dataclasses.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from fraud_mlops.performance import (
    compute_window_metrics,
    evaluate_alert,
    join_matured_labels,
    label_delay_to_wall_clock,
    latest_evaluable_window,
)

T0 = datetime(2026, 3, 1, 0, 0, 0)


def _decisions(rows: list[tuple[str, bool, datetime]]) -> pd.DataFrame:
    return pd.DataFrame([{"transaction_id": t, "is_fraud": p, "scored_at": s} for t, p, s in rows])


LABEL_COLUMNS = ["transaction_id", "is_fraud", "event_time", "label_available_at"]


def _labels(rows: list[tuple[str, bool, datetime]]) -> pd.DataFrame:
    """A labels frame, with the right columns even when it has no rows.

    An empty result from Postgres still carries its column names, so a test
    double that drops them would be testing a situation that cannot happen.
    """
    return pd.DataFrame(
        [
            {
                "transaction_id": t,
                "is_fraud": actual,
                "event_time": available - timedelta(hours=1),
                "label_available_at": available,
            }
            for t, actual, available in rows
        ],
        columns=LABEL_COLUMNS,
    )


# --- The maturity join ---


def test_only_matured_labels_are_joined() -> None:
    decisions = _decisions(
        [
            ("a", True, T0),
            ("b", False, T0),
            ("c", True, T0),
        ]
    )
    labels = _labels(
        [
            ("a", True, T0 + timedelta(hours=1)),  # matured
            ("b", True, T0 + timedelta(hours=1)),  # matured
            ("c", True, T0 + timedelta(hours=99)),  # still in flight
        ]
    )

    joined = join_matured_labels(decisions, labels, as_of=T0 + timedelta(hours=2))

    assert set(joined["transaction_id"]) == {"a", "b"}
    assert "c" not in set(joined["transaction_id"])


def test_an_unmatured_label_is_absent_not_null() -> None:
    """The failure mode this guards: a null label read as 'not fraud'.

    If `c` were joined with a null outcome and something coerced that to False,
    a caught fraud would be counted as a false positive and a missed one as a
    true negative.
    """
    decisions = _decisions([("c", True, T0)])
    labels = _labels([("c", True, T0 + timedelta(hours=99))])

    joined = join_matured_labels(decisions, labels, as_of=T0)

    assert joined.empty
    assert "actual_fraud" in joined.columns  # shape is stable even when empty


def test_maturity_boundary_is_inclusive() -> None:
    """A label maturing exactly at `as_of` counts as available."""
    decisions = _decisions([("a", True, T0)])
    labels = _labels([("a", True, T0 + timedelta(hours=5))])

    assert len(join_matured_labels(decisions, labels, as_of=T0 + timedelta(hours=5))) == 1
    assert (
        len(join_matured_labels(decisions, labels, as_of=T0 + timedelta(hours=4, minutes=59))) == 0
    )


def test_advancing_the_clock_only_adds_labels() -> None:
    """Monotonicity: time passing can never un-mature a label."""
    decisions = _decisions([(f"t{i}", True, T0) for i in range(5)])
    labels = _labels([(f"t{i}", True, T0 + timedelta(hours=i)) for i in range(5)])

    sizes = [
        len(join_matured_labels(decisions, labels, as_of=T0 + timedelta(hours=h))) for h in range(6)
    ]
    assert sizes == sorted(sizes)
    assert sizes[-1] == 5


def test_decisions_without_any_label_are_dropped() -> None:
    decisions = _decisions([("a", True, T0), ("orphan", False, T0)])
    labels = _labels([("a", True, T0)])

    joined = join_matured_labels(decisions, labels, as_of=T0 + timedelta(hours=1))
    assert set(joined["transaction_id"]) == {"a"}


def test_join_rejects_frames_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="decisions is missing"):
        join_matured_labels(pd.DataFrame({"transaction_id": ["a"]}), _labels([]), as_of=T0)
    with pytest.raises(ValueError, match="labels is missing"):
        join_matured_labels(
            _decisions([("a", True, T0)]), pd.DataFrame({"transaction_id": ["a"]}), as_of=T0
        )


def test_naive_and_aware_timestamps_compare_correctly() -> None:
    """Postgres returns tz-aware timestamps; a test double may not.

    Mixing the two raises in pandas rather than silently comparing wrong, but
    only if something normalizes them — which is what `_utc` is for.
    """
    decisions = _decisions([("a", True, T0)])
    labels = _labels([("a", True, T0)])
    labels["label_available_at"] = pd.to_datetime(labels["label_available_at"], utc=True)

    joined = join_matured_labels(decisions, labels, as_of=T0 + timedelta(hours=1))
    assert len(joined) == 1


# --- Window boundaries ---


def test_windows_are_aligned_to_the_clock_not_to_the_run_time() -> None:
    """Two runs at different times must bucket the same decision identically.

    Windows anchored to `as_of` would shift every night, so yesterday's recall
    would be computed over a different set of transactions depending on when the
    flow happened to fire.
    """
    decisions = _decisions([("a", True, datetime(2026, 3, 2, 13, 30))])
    joined = join_matured_labels(decisions, _labels([("a", True, T0)]), as_of=T0)

    early = compute_window_metrics(
        decisions, joined, window_hours=24, as_of=datetime(2026, 3, 3, 1, 0)
    )
    late = compute_window_metrics(
        decisions, joined, window_hours=24, as_of=datetime(2026, 3, 3, 23, 0)
    )

    assert [w.window_start for w in early if w.n_decisions] == [
        w.window_start for w in late if w.n_decisions
    ]


def test_windows_do_not_overlap_and_cover_every_decision() -> None:
    decisions = _decisions([(f"t{i}", False, T0 + timedelta(hours=i * 6)) for i in range(12)])
    joined = join_matured_labels(decisions, _labels([]), as_of=T0)

    windows = compute_window_metrics(decisions, joined, window_hours=24, lookback_windows=10)

    assert sum(w.n_decisions for w in windows) == len(decisions)
    starts = [w.window_start for w in windows]
    assert starts == sorted(starts)
    assert len(starts) == len(set(starts))


def test_lookback_limits_how_far_back_windows_go() -> None:
    decisions = _decisions([(f"t{i}", False, T0 + timedelta(days=i)) for i in range(10)])
    joined = join_matured_labels(decisions, _labels([]), as_of=T0)

    windows = compute_window_metrics(
        decisions, joined, window_hours=24, as_of=T0 + timedelta(days=9), lookback_windows=3
    )
    assert len(windows) == 3


def test_window_reports_precision_and_recall_from_matured_rows_only() -> None:
    """Six decisions in one window, four labels matured."""
    decisions = _decisions(
        [
            ("tp", True, T0),  # caught fraud
            ("fp", True, T0),  # false alarm
            ("fn", False, T0),  # missed fraud
            ("tn", False, T0),  # correctly allowed
            ("late1", True, T0),  # label not matured
            ("late2", False, T0),  # label not matured
        ]
    )
    labels = _labels(
        [
            ("tp", True, T0),
            ("fp", False, T0),
            ("fn", True, T0),
            ("tn", False, T0),
            ("late1", True, T0 + timedelta(days=30)),
            ("late2", True, T0 + timedelta(days=30)),
        ]
    )
    joined = join_matured_labels(decisions, labels, as_of=T0 + timedelta(hours=1))
    window = compute_window_metrics(decisions, joined, window_hours=24)[-1]

    assert window.n_decisions == 6
    assert window.n_matured == 4
    assert window.maturity_share == pytest.approx(4 / 6)
    assert (window.true_positives, window.false_positives) == (1, 1)
    assert (window.false_negatives, window.true_negatives) == (1, 1)
    assert window.precision == pytest.approx(0.5)
    assert window.recall == pytest.approx(0.5)


def test_window_with_no_matured_labels_reports_none_not_zero() -> None:
    """Zero would be indistinguishable from a model that caught nothing."""
    decisions = _decisions([("a", True, T0), ("b", False, T0)])
    labels = _labels([("a", True, T0 + timedelta(days=30))])

    joined = join_matured_labels(decisions, labels, as_of=T0)
    window = compute_window_metrics(decisions, joined, window_hours=24)[-1]

    assert window.n_decisions == 2
    assert window.n_matured == 0
    assert window.maturity_share == 0.0
    assert window.precision is None
    assert window.recall is None


def test_no_decisions_produces_no_windows() -> None:
    empty = _decisions([])
    assert compute_window_metrics(empty, empty, window_hours=24) == []


# --- Choosing which window to judge ---


def _window_series() -> list:
    """Three days: two fully matured, the newest still in flight."""
    decisions = _decisions(
        [(f"old{i}", i % 4 == 0, T0 + timedelta(hours=i % 20)) for i in range(120)]
        + [(f"new{i}", i % 4 == 0, T0 + timedelta(days=2, hours=i % 20)) for i in range(120)]
    )
    labels = _labels(
        [(f"old{i}", i % 4 == 0, T0) for i in range(120)]
        + [(f"new{i}", i % 4 == 0, T0 + timedelta(days=30)) for i in range(120)]
    )
    joined = join_matured_labels(decisions, labels, as_of=T0 + timedelta(days=3))
    return compute_window_metrics(
        decisions, joined, window_hours=24, as_of=T0 + timedelta(days=3), lookback_windows=5
    )


def test_latest_evaluable_window_skips_the_unmatured_tail() -> None:
    """The newest window is the one with no labels; judgement needs the one before."""
    windows = _window_series()
    chosen = latest_evaluable_window(windows, min_matured_labels=50, min_matured_positives=10)

    assert chosen is not None
    assert chosen.recall is not None
    assert chosen.n_matured >= 50
    # Not the final window — that one's labels are 30 days out.
    assert chosen.window_start != windows[-1].window_start


def test_latest_evaluable_window_is_none_when_nothing_has_matured() -> None:
    decisions = _decisions([(f"t{i}", True, T0) for i in range(100)])
    labels = _labels([(f"t{i}", True, T0 + timedelta(days=30)) for i in range(100)])
    joined = join_matured_labels(decisions, labels, as_of=T0)
    windows = compute_window_metrics(decisions, joined, window_hours=24)

    assert latest_evaluable_window(windows) is None


# --- The alert rule ---


def _single_window(tp: int, fn: int, fp: int = 0, tn: int = 200):
    """A window with an exact confusion matrix, for alert-threshold tests."""
    decisions = (
        [(f"tp{i}", True, T0) for i in range(tp)]
        + [(f"fn{i}", False, T0) for i in range(fn)]
        + [(f"fp{i}", True, T0) for i in range(fp)]
        + [(f"tn{i}", False, T0) for i in range(tn)]
    )
    labels = (
        [(f"tp{i}", True, T0) for i in range(tp)]
        + [(f"fn{i}", True, T0) for i in range(fn)]
        + [(f"fp{i}", False, T0) for i in range(fp)]
        + [(f"tn{i}", False, T0) for i in range(tn)]
    )
    d = _decisions(decisions)
    joined = join_matured_labels(d, _labels(labels), as_of=T0 + timedelta(hours=1))
    return compute_window_metrics(d, joined, window_hours=24)


def test_alert_fires_when_realized_recall_falls_below_the_floor() -> None:
    windows = _single_window(tp=20, fn=80)  # recall 20%
    decision = evaluate_alert(windows, recall_floor=0.70)

    assert decision.should_alert is True
    assert "below the floor" in decision.reason
    assert decision.window is not None
    assert decision.window.recall == pytest.approx(0.2)


def test_alert_does_not_fire_when_recall_holds() -> None:
    windows = _single_window(tp=90, fn=10)  # recall 90%
    decision = evaluate_alert(windows, recall_floor=0.70)

    assert decision.should_alert is False
    assert "holds the floor" in decision.reason


def test_alert_boundary_is_exclusive_at_the_floor() -> None:
    """Recall exactly at the floor is acceptable; a hair under is not."""
    at_floor = evaluate_alert(_single_window(tp=70, fn=30), recall_floor=0.70)
    assert at_floor.should_alert is False

    under = evaluate_alert(_single_window(tp=69, fn=31), recall_floor=0.70)
    assert under.should_alert is True


def test_alert_never_fires_on_a_window_with_too_few_matured_labels() -> None:
    """The most important negative case.

    Recall here is 0% — every fraud was missed — but it is 0% over two labels.
    Alerting on that would mean alerting every night on the newest window.
    """
    windows = _single_window(tp=0, fn=2, tn=3)
    decision = evaluate_alert(windows, recall_floor=0.70, min_matured_labels=50)

    assert decision.should_alert is False
    assert "matured" in decision.reason
    assert decision.window is None


def test_alert_never_fires_on_too_few_matured_fraud_cases() -> None:
    """Plenty of labels, but only three of them are fraud.

    Recall over three positives moves 33 points per case; it is not a signal.
    """
    windows = _single_window(tp=0, fn=3, tn=500)
    decision = evaluate_alert(
        windows, recall_floor=0.70, min_matured_labels=50, min_matured_positives=10
    )

    assert decision.should_alert is False
    assert decision.window is None


def test_alert_decision_serializes_for_sns_and_logging() -> None:
    decision = evaluate_alert(_single_window(tp=20, fn=80), recall_floor=0.70)
    payload = decision.to_dict()

    assert payload["should_alert"] is True
    assert payload["recall_floor"] == 0.70
    assert payload["window"]["recall"] == pytest.approx(0.2)


# --- The two clocks ---


def test_label_delay_is_wall_clock_hours_not_paysim_steps() -> None:
    """A demo compresses the delay; production uses the real chargeback window."""
    assert label_delay_to_wall_clock(0.05) == timedelta(minutes=3)
    assert label_delay_to_wall_clock(720) == timedelta(days=30)
