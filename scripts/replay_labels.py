#!/usr/bin/env python3
"""Release outcome labels on a delay, the way real fraud labels arrive.

In production nobody knows a transaction was fraudulent at the moment it is
scored. The label is a chargeback or a dispute, and it lands days to weeks
later. A monitoring setup that joins `decisions` to `labels` the same night is
measuring nothing.

This script simulates that delay:

    for each decision that has no label yet,
        if (now - scored_at) >= LABEL_DELAY_HOURS:
            copy the true outcome from `ground_truth` into `labels`,
            stamping label_available_at = scored_at + LABEL_DELAY_HOURS

`ground_truth` is written by the producer at send time — it is what the
simulator knows. `labels` is what the business has been *told*. Keeping them in
two tables is the point: the performance flow can only ever read `labels`, and
only rows whose `label_available_at` has passed.

**Two clocks.** PaySim's `step` is a simulated hour and never advances in real
time; it is a column in a static file. The replay runs on wall-clock time, so
`LABEL_DELAY_HOURS` is real hours since the decision. For a demo, set it small
(0.05 = three minutes) to watch a month of label delay compress into a coffee
break. Production would set it to the real chargeback window (720 = 30 days).

Usage:
    python scripts/replay_labels.py                  # release whatever is due
    python scripts/replay_labels.py --delay-hours 0.05
    python scripts/replay_labels.py --watch --interval 30
    python scripts/replay_labels.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_mlops.db import connect  # noqa: E402

logger = logging.getLogger("replay_labels")

# How long after a decision its label becomes knowable, in wall-clock hours.
DEFAULT_DELAY_HOURS = float(os.environ.get("LABEL_DELAY_HOURS", "0.05"))

# Rows that have a decision and a known outcome, where the delay has elapsed and
# no label has been published yet. `scored_at + delay` is the maturity time, so
# a label released now is stamped with when it *became* knowable rather than
# when this script happened to run — which keeps the data the same whether the
# script runs every minute or once a day.
RELEASE_SQL = """
INSERT INTO labels (transaction_id, is_fraud, event_time, label_available_at, source)
SELECT
    d.transaction_id,
    g.is_fraud,
    d.scored_at,
    d.scored_at + make_interval(secs => %(delay_seconds)s),
    'paysim_replay'
FROM decisions d
JOIN ground_truth g ON g.transaction_id = d.transaction_id
LEFT JOIN labels l ON l.transaction_id = d.transaction_id
WHERE l.transaction_id IS NULL
  AND d.scored_at + make_interval(secs => %(delay_seconds)s) <= NOW()
ON CONFLICT (transaction_id) DO NOTHING
"""

DUE_COUNT_SQL = """
SELECT COUNT(*)
FROM decisions d
JOIN ground_truth g ON g.transaction_id = d.transaction_id
LEFT JOIN labels l ON l.transaction_id = d.transaction_id
WHERE l.transaction_id IS NULL
  AND d.scored_at + make_interval(secs => %(delay_seconds)s) <= NOW()
"""

PENDING_SQL = """
SELECT
    COUNT(*) FILTER (WHERE l.transaction_id IS NULL)                           AS unlabelled,
    COUNT(*) FILTER (WHERE l.transaction_id IS NULL AND g.transaction_id IS NULL) AS no_ground_truth,
    COUNT(*) FILTER (
        WHERE l.transaction_id IS NULL
          AND g.transaction_id IS NOT NULL
          AND d.scored_at + make_interval(secs => %(delay_seconds)s) > NOW()
    )                                                                          AS still_maturing
FROM decisions d
LEFT JOIN ground_truth g ON g.transaction_id = d.transaction_id
LEFT JOIN labels l ON l.transaction_id = d.transaction_id
"""


def release_due_labels(conn, delay_hours: float, dry_run: bool = False) -> int:
    """Publish every label whose delay has elapsed. Returns how many."""
    params = {"delay_seconds": delay_hours * 3600.0}

    with conn.cursor() as cur:
        cur.execute(PENDING_SQL, params)
        unlabelled, no_truth, maturing = cur.fetchone()

        if dry_run:
            cur.execute(DUE_COUNT_SQL, params)
            released = int(cur.fetchone()[0])
        else:
            cur.execute(RELEASE_SQL, params)
            released = cur.rowcount

    if not dry_run:
        conn.commit()

    logger.info(
        "%s %d label(s). Still pending: %d unlabelled "
        "(%d awaiting maturity, %d with no ground truth recorded)",
        "Would release" if dry_run else "Released",
        released,
        unlabelled,
        maturing,
        no_truth,
    )
    return released


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--delay-hours",
        type=float,
        default=DEFAULT_DELAY_HOURS,
        help=f"Wall-clock hours between decision and label (default {DEFAULT_DELAY_HOURS})",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep running, releasing labels as they mature",
    )
    parser.add_argument(
        "--interval", type=float, default=30.0, help="Seconds between --watch passes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what is pending without publishing anything",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    conn = connect()
    try:
        if not args.watch:
            release_due_labels(conn, args.delay_hours, dry_run=args.dry_run)
            return

        logger.info(
            "Watching for labels to mature every %.0fs (delay %.3fh). Ctrl-C to stop.",
            args.interval,
            args.delay_hours,
        )
        while True:
            release_due_labels(conn, args.delay_hours, dry_run=args.dry_run)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
