#!/usr/bin/env python3
"""Demo: why data-drift monitoring alone is not enough.

The point this demonstrates is the one a sharp interviewer goes looking for.
Drift detection compares the *input distribution* to a training baseline. It
cannot see a change in the relationship between inputs and outcomes — and that
relationship is what a model actually encodes.

So the scenario moves the fraud pattern without moving the input distribution
much:

    phase 1  normal traffic, PaySim-style fraud (the origin account is emptied)
             -> the model catches most of it, realized recall is high

    phase 2  the fraud pattern shifts: many small amounts, balance left behind
             -> input distribution barely moves, so drift stays quiet
             -> the model misses almost all of it
             -> but nobody knows yet, because labels have not matured

    phase 3  the label delay elapses, labels land, realized recall collapses,
             the performance flow alerts

The timeline is the exhibit. Drift share stays flat throughout; realized recall
only falls once the labels arrive, which is *after* the damage.

Requires the AWS stack to be up (Kinesis, Lambda, RDS). It produces real traffic
and writes real rows. Run `python -m fraud_mlops.db migrate` first.

Usage:
    python scripts/demo_performance_drop.py --plan          # print, send nothing
    python scripts/demo_performance_drop.py --delay-hours 0.05
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("demo")

PRODUCER = str(REPO_ROOT / "scripts" / "produce_transactions.py")
REPLAY = str(REPO_ROOT / "scripts" / "replay_labels.py")
PERF_FLOW = str(REPO_ROOT / "flows" / "monitor_performance.py")


def _run(command: list[str], plan: bool, description: str) -> None:
    logger.info("%s", description)
    logger.info("  $ %s", " ".join(command))
    if plan:
        return
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--baseline-count", type=int, default=400, help="Transactions in the healthy phase"
    )
    parser.add_argument(
        "--shifted-count", type=int, default=400, help="Transactions after the pattern shifts"
    )
    parser.add_argument("--rate", type=float, default=8.0, help="Transactions per second")
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.15,
        help=(
            "Fraud share. Far above PaySim's real 0.13% so that a few minutes of "
            "traffic contains enough matured positives for recall to mean something "
            "(the alert rule requires at least 10)."
        ),
    )
    parser.add_argument(
        "--delay-hours",
        type=float,
        default=0.05,
        help="Label delay in wall-clock hours (0.05 = 3 minutes)",
    )
    parser.add_argument(
        "--plan", action="store_true", help="Print the commands without running them"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    wait_seconds = args.delay_hours * 3600

    logger.info("=" * 72)
    logger.info("Demo: a fraud pattern shift that data-drift monitoring cannot see")
    logger.info("=" * 72)
    logger.info(
        "Label delay %.3fh (%.0fs). Baseline %d txns, shifted %d txns, %.0f txn/s, %.0f%% fraud.",
        args.delay_hours,
        wait_seconds,
        args.baseline_count,
        args.shifted_count,
        args.rate,
        args.fraud_rate * 100,
    )

    # ── Phase 1: healthy traffic ────────────────────────────────────────────
    _run(
        [
            sys.executable,
            PRODUCER,
            "--count",
            str(args.baseline_count),
            "--rate",
            str(args.rate),
            "--fraud-rate",
            str(args.fraud_rate),
            "--fraud-profile",
            "paysim",
            "--record-ground-truth",
        ],
        args.plan,
        "Phase 1 — normal traffic with PaySim-style fraud (the model's own pattern)",
    )

    # ── Phase 2: the pattern shifts ─────────────────────────────────────────
    _run(
        [
            sys.executable,
            PRODUCER,
            "--count",
            str(args.shifted_count),
            "--rate",
            str(args.rate),
            "--fraud-rate",
            str(args.fraud_rate),
            "--fraud-profile",
            "shifted",
            "--record-ground-truth",
        ],
        args.plan,
        "Phase 2 — fraud shifts to small amounts that leave a balance behind. "
        "The input distribution barely moves; the model's signal is gone.",
    )

    # ── The waiting, which is the whole point ───────────────────────────────
    logger.info(
        "Phase 2.5 — waiting %.0fs for labels to mature. In this window the system "
        "has already been failing and has no way to know it: drift is quiet and no "
        "outcome has arrived. This is the gap the performance flow exists to close, "
        "not to eliminate.",
        wait_seconds,
    )
    if not args.plan:
        time.sleep(wait_seconds + 5)

    # ── Phase 3: labels land, the flow sees the damage ─────────────────────
    _run(
        [sys.executable, REPLAY, "--delay-hours", str(args.delay_hours)],
        args.plan,
        "Phase 3 — releasing matured labels",
    )
    _run(
        [sys.executable, PERF_FLOW],
        args.plan,
        "Phase 3 — running the performance flow: realized recall should now be "
        "below the floor, and SNS should fire",
    )

    logger.info("=" * 72)
    logger.info("What to look at now:")
    logger.info("  - Grafana 'Realized performance' panel: recall drops in phase 2's window")
    logger.info("  - Grafana drift panels: flat throughout — the inputs did not change much")
    logger.info("  - Inbox: the SNS performance alert")
    logger.info(
        "  - The lag between the recall drop and the alert is the label delay, and it "
        "is the honest cost of outcome-based monitoring."
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
