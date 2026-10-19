#!/usr/bin/env python3
"""Summarise Lambda duration from CloudWatch REPORT lines.

Every Lambda invocation emits a REPORT line to CloudWatch Logs:

    REPORT RequestId: ...  Duration: 412.55 ms  Billed Duration: 413 ms
    Memory Size: 1024 MB  Max Memory Used: 486 MB  Init Duration: 23891.04 ms

`Init Duration` appears only on a cold start, which makes it the clean way to
separate the two populations. Reporting one blended percentile over both hides
the thing anyone actually wants to know: warm invocations are milliseconds and
cold ones are tens of seconds, and the mix depends entirely on traffic shape.

CloudWatch's own metrics can't do this — the `Duration` metric has no cold/warm
dimension — so this reads the log lines.

Usage:
    python scripts/lambda_latency.py --hours 2
    python scripts/lambda_latency.py --hours 24 --out reports/perf/2026-10-19
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger("lambda_latency")

DEFAULT_LOG_GROUP = "/aws/lambda/fraud-mlops-scorer"

REPORT_PATTERN = re.compile(
    r"Duration:\s*(?P<duration>[\d.]+)\s*ms.*?"
    r"Billed Duration:\s*(?P<billed>\d+)\s*ms.*?"
    r"Memory Size:\s*(?P<memory>\d+)\s*MB.*?"
    r"Max Memory Used:\s*(?P<used>\d+)\s*MB"
    r"(?:.*?Init Duration:\s*(?P<init>[\d.]+)\s*ms)?",
    re.DOTALL,
)


@dataclass
class DurationSummary:
    """Percentiles over one population of invocations."""

    population: str
    count: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    min_ms: float | None
    max_ms: float | None
    mean_ms: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. Explicit because numpy's default interpolates."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(q / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[index]


def summarize(population: str, values: list[float]) -> DurationSummary:
    if not values:
        return DurationSummary(population, 0, None, None, None, None, None, None)
    return DurationSummary(
        population=population,
        count=len(values),
        p50_ms=round(_percentile(values, 50), 2),
        p95_ms=round(_percentile(values, 95), 2),
        p99_ms=round(_percentile(values, 99), 2),
        min_ms=round(min(values), 2),
        max_ms=round(max(values), 2),
        mean_ms=round(statistics.fmean(values), 2),
    )


def fetch_reports(log_group: str, start: datetime, end: datetime, region: str) -> list[dict]:
    """Every REPORT line in the window, parsed."""
    client = boto3.client("logs", region_name=region)
    paginator = client.get_paginator("filter_log_events")

    rows: list[dict] = []
    pages = paginator.paginate(
        logGroupName=log_group,
        startTime=int(start.timestamp() * 1000),
        endTime=int(end.timestamp() * 1000),
        filterPattern="REPORT",
    )

    for page in pages:
        for event in page.get("events", []):
            match = REPORT_PATTERN.search(event["message"])
            if not match:
                continue
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(event["timestamp"] / 1000, tz=UTC),
                    "duration_ms": float(match.group("duration")),
                    "billed_ms": int(match.group("billed")),
                    "memory_mb": int(match.group("memory")),
                    "max_memory_used_mb": int(match.group("used")),
                    "init_ms": float(match.group("init")) if match.group("init") else None,
                }
            )

    logger.info("Parsed %d REPORT lines from %s", len(rows), log_group)
    return rows


def build_report(rows: list[dict], start: datetime, end: datetime, log_group: str) -> dict:
    cold = [r for r in rows if r["init_ms"] is not None]
    warm = [r for r in rows if r["init_ms"] is None]

    summaries = [
        summarize("all", [r["duration_ms"] for r in rows]),
        summarize("warm", [r["duration_ms"] for r in warm]),
        # Handler duration on a cold invocation, excluding init.
        summarize("cold (handler only)", [r["duration_ms"] for r in cold]),
        # The init phase itself — importing MLflow, loading the model, Feast.
        summarize("cold (init only)", [r["init_ms"] for r in cold if r["init_ms"]]),
        # What the caller actually waits for on a cold start.
        summarize("cold (init + handler)", [r["duration_ms"] + r["init_ms"] for r in cold]),
    ]

    memory_used = [r["max_memory_used_mb"] for r in rows]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "log_group": log_group,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "invocations": len(rows),
        "cold_starts": len(cold),
        "warm_invocations": len(warm),
        "cold_start_share": (len(cold) / len(rows)) if rows else 0.0,
        "memory_size_mb": rows[0]["memory_mb"] if rows else None,
        "max_memory_used_mb": max(memory_used) if memory_used else None,
        "durations": [s.to_dict() for s in summaries],
    }


def render(report: dict) -> str:
    lines = [
        "# Lambda latency",
        "",
        f"Log group `{report['log_group']}`, {report['window_start']} to {report['window_end']}.",
        "",
        f"- **{report['invocations']:,} invocations** "
        f"({report['cold_starts']} cold, {report['warm_invocations']} warm — "
        f"{report['cold_start_share']:.1%} cold)",
        f"- Memory configured {report['memory_size_mb']} MB, "
        f"peak used {report['max_memory_used_mb']} MB",
        "",
        "| Population | n | p50 | p95 | p99 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["durations"]:
        if not row["count"]:
            continue
        lines.append(
            f"| {row['population']} | {row['count']:,} | {row['p50_ms']:.1f} ms | "
            f"{row['p95_ms']:.1f} ms | {row['p99_ms']:.1f} ms | "
            f"{row['min_ms']:.1f} ms | {row['max_ms']:.1f} ms |"
        )
    lines += [
        "",
        "Cold and warm are reported separately because they are different "
        "distributions, not a tail of one. `init` is the container bootstrap — "
        "importing MLflow, resolving the `@production` alias, loading the model "
        "artifact and the Feast registry — and it happens once per container. "
        "`init + handler` is what the first record of a new container waits for.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hours", type=float, default=2.0, help="Window size, hours back from now")
    parser.add_argument("--log-group", default=DEFAULT_LOG_GROUP)
    parser.add_argument("--region", default=None, help="Defaults to AWS_DEFAULT_REGION")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write lambda_latency.{json,md} into",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    end = datetime.now(UTC)
    start = end - timedelta(hours=args.hours)

    rows = fetch_reports(args.log_group, start, end, region)
    if not rows:
        print(
            f"No REPORT lines found in {args.log_group} for the last {args.hours}h. "
            f"Has the function been invoked? Is the stack up?"
        )
        return

    report = build_report(rows, start, end, args.log_group)
    text = render(report)
    print(text)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "lambda_latency.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.out / "lambda_latency.md").write_text(text)
        print(f"Wrote {args.out}/lambda_latency.json and .md")


if __name__ == "__main__":
    main()
