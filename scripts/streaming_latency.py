#!/usr/bin/env python3
"""Summarise ingest-to-decision latency from the decisions table.

The Lambda records, for every scored record, when Kinesis received it
(`ingested_at`, from ApproximateArrivalTimestamp) and when the decision was
written (`scored_at`). `latency_ms` is the difference.

That span is the one that matters for the streaming path, and it is strictly
larger than the Lambda's own duration: it includes time the record spent
waiting in the shard for a batch to form and for a container to pick it up.
`scripts/lambda_latency.py` measures the function; this measures the pipeline.

Usage:
    python scripts/streaming_latency.py --hours 1
    python scripts/streaming_latency.py --hours 1 --out reports/perf/2026-10-19
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_mlops.db import connect  # noqa: E402

logger = logging.getLogger("streaming_latency")

SUMMARY_SQL = """
SELECT
    COUNT(*)                                                              AS n_decisions,
    COUNT(latency_ms)                                                     AS n_with_latency,
    MIN(latency_ms)                                                       AS min_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms)              AS p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)              AS p95_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms)              AS p99_ms,
    MAX(latency_ms)                                                       AS max_ms,
    AVG(latency_ms)                                                       AS mean_ms
FROM decisions
WHERE scored_at > NOW() - make_interval(hours => %(hours)s)
"""

# Split by model version too: a version change mid-window is the usual reason
# for a bimodal latency distribution, and seeing both is better than averaging
# them into a number that describes neither.
BY_VERSION_SQL = """
SELECT
    model_version,
    COUNT(*)                                                 AS n,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
FROM decisions
WHERE scored_at > NOW() - make_interval(hours => %(hours)s)
  AND latency_ms IS NOT NULL
GROUP BY model_version
ORDER BY n DESC
"""


def collect(hours: float) -> dict:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(SUMMARY_SQL, {"hours": hours})
            columns = [c[0] for c in cur.description]
            summary = dict(zip(columns, cur.fetchone(), strict=True))

            cur.execute(BY_VERSION_SQL, {"hours": hours})
            version_columns = [c[0] for c in cur.description]
            by_version = [dict(zip(version_columns, row, strict=True)) for row in cur.fetchall()]
    finally:
        conn.close()

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_hours": hours,
        "summary": {k: (float(v) if v is not None else None) for k, v in summary.items()},
        "by_model_version": [
            {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()}
            for row in by_version
        ],
    }


def render(report: dict) -> str:
    s = report["summary"]
    n, n_lat = int(s["n_decisions"]), int(s["n_with_latency"])

    lines = [
        "# Streaming latency (ingest to decision)",
        "",
        f"Window: last {report['window_hours']}h, as of {report['generated_at']}.",
        "",
        f"- **{n:,} decisions**, {n_lat:,} with a recorded arrival time",
    ]
    if n and n_lat < n:
        lines.append(
            f"- {n - n_lat:,} rows predate the `ingested_at` column "
            f"(migration 003) and are excluded from the percentiles"
        )
    lines += ["", "| p50 | p95 | p99 | min | max | mean |", "|---:|---:|---:|---:|---:|---:|"]

    if n_lat:
        lines.append(
            f"| {s['p50_ms']:.0f} ms | {s['p95_ms']:.0f} ms | {s['p99_ms']:.0f} ms | "
            f"{s['min_ms']:.0f} ms | {s['max_ms']:.0f} ms | {s['mean_ms']:.0f} ms |"
        )
    else:
        lines.append("| — | — | — | — | — | — |")

    if report["by_model_version"]:
        lines += ["", "| Model version | n | p50 | p95 |", "|---|---:|---:|---:|"]
        for row in report["by_model_version"]:
            lines.append(
                f"| v{row['model_version']} | {int(row['n']):,} | "
                f"{row['p50_ms']:.0f} ms | {row['p95_ms']:.0f} ms |"
            )

    lines += [
        "",
        "This is the whole pipeline: Kinesis receipt to the decision being "
        "committed to Postgres. It includes batch formation and any wait for a "
        "Lambda container, so it is always larger than the function's own "
        "reported duration.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None, help="Directory for the report files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    report = collect(args.hours)
    text = render(report)
    print(text)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "streaming_latency.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.out / "streaming_latency.md").write_text(text)
        print(f"Wrote {args.out}/streaming_latency.json and .md")


if __name__ == "__main__":
    main()
