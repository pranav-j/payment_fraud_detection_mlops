"""Cost estimate for the AWS stack, computed from fetched ap-south-1 rates.

    make cost

Reads `reports/cost/pricing-inputs.json` (written by `scripts/fetch_pricing.py`
from the AWS Price List API) and writes `reports/cost/estimate.{json,md}`.

**This is an estimate.** Every rate is real and sourced; every *usage* number is
an assumption, and the assumptions are stated next to the figures they produce
rather than buried. Where a measured number exists — throughput from a load
test, Lambda duration from CloudWatch — the estimate uses it and says so; where
one does not, it uses a documented placeholder and says that too.

Three questions it answers, which are the three worth asking:

1. What does an hour of the stack running cost?
2. What does a thousand predictions cost on each serving path?
3. What does it cost while torn down?
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fraud_mlops.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

COST_DIR = PROJECT_ROOT / "reports" / "cost"
PRICING_INPUTS = COST_DIR / "pricing-inputs.json"

HOURS_PER_MONTH = 730  # AWS's own convention


# ── Usage assumptions ───────────────────────────────────────────────────────
# Each one is a number this estimate cannot derive, only assume. They are
# collected here so a reader can disagree with a specific figure rather than
# with the total.


@dataclass
class UsageAssumptions:
    """What the stack is assumed to be doing. Override from measurements."""

    # Streaming path
    kinesis_shards: int = 1
    # Measured 2026-10-19: the producer ran at 10 txn/s and the Lambda kept up.
    transactions_per_second: float = 10.0
    kinesis_record_bytes: int = 400

    # Lambda: 1024 MB configured in infra/lambda.tf.
    lambda_memory_mb: int = 1024
    lambda_batch_size: int = 10
    # Measured 2026-10-19: warm p50 43.8 ms over 825 invocations
    # (reports/perf/2026-10-19/lambda_latency.json).
    lambda_duration_ms_per_batch: float = 43.8
    lambda_duration_measured: bool = True

    # Synchronous path: both task definitions, as sized in infra/ecs.tf.
    fargate_tasks: tuple[tuple[str, float, float], ...] = (
        ("mlflow", 1.0, 2.0),
        ("api", 0.5, 1.0),
    )
    # Measured 2026-10-19: the Fargate API sustained 42 req/s with zero errors
    # (reports/perf/2026-10-19/locust-50rps_stats.csv). Costed at 20 req/s, the
    # middle load level, since that is a rate the service holds comfortably
    # rather than its observed ceiling.
    api_requests_per_second: float = 20.0
    api_requests_measured: bool = True

    # ALB. One LCU covers 25 new connections/sec, 3,000 active, 1 GB/hour
    # processed and 1,000 rule evaluations/sec; at this scale it is the
    # connection dimension that binds, and it stays comfortably inside one.
    alb_lcus: float = 1.0

    # Always-on, whether or not the stack is up.
    s3_gb: float = 5.0
    ecr_gb: float = 4.5
    secrets_count: int = 2
    cloudwatch_gb_per_month: float = 1.0

    # Public IPv4 addresses: ALB nodes plus the Redis EC2 plus Fargate tasks
    # with assign_public_ip = true.
    public_ipv4_addresses: int = 5

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fargate_tasks"] = [list(t) for t in self.fargate_tasks]
        return payload


@dataclass
class LineItem:
    """One cost line, with the arithmetic that produced it."""

    component: str
    basis: str
    hourly_usd: float
    monthly_usd: float
    idle: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CostEstimate:
    generated_at: str
    region: str
    pricing_fetched_at: str
    assumptions: dict[str, Any]
    running: list[dict[str, Any]] = field(default_factory=list)
    idle: list[dict[str, Any]] = field(default_factory=list)
    per_1k_predictions: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    measured_inputs: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_rates(path: Path = PRICING_INPUTS) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python scripts/fetch_pricing.py` first — "
            f"it reads the AWS Price List API, which is free and read-only."
        )
    payload: dict[str, Any] = json.loads(path.read_text())
    missing = payload.get("rates_missing") or []
    if missing:
        logger.warning("Pricing inputs are incomplete: %s", missing)
    return payload


def _rate(rates: dict[str, Any], key: str) -> float:
    entry = rates["rates"].get(key)
    if entry is None or entry.get("value") is None:
        raise KeyError(
            f"No rate for {key!r} in the pricing inputs. "
            f"Re-run scripts/fetch_pricing.py, or narrow its filters."
        )
    return float(entry["value"])


def build_estimate(
    rates_payload: dict[str, Any],
    usage: UsageAssumptions | None = None,
    measured: dict[str, Any] | None = None,
) -> CostEstimate:
    """Compute running, idle and per-prediction costs."""
    usage = usage or UsageAssumptions()
    measured = measured or {}
    rates = rates_payload

    running: list[LineItem] = []
    idle: list[LineItem] = []

    def add(
        bucket: list[LineItem],
        component: str,
        basis: str,
        hourly: float,
        note: str = "",
        is_idle: bool = False,
    ) -> None:
        bucket.append(
            LineItem(
                component=component,
                basis=basis,
                hourly_usd=round(hourly, 6),
                monthly_usd=round(hourly * HOURS_PER_MONTH, 2),
                idle=is_idle,
                note=note,
            )
        )

    # ── Running: the hourly, torn-down-between-sessions resources ──────────
    add(
        running,
        "RDS (db.t4g.micro, Single-AZ)",
        f"1 instance x ${_rate(rates, 'rds_db_t4g_micro_hour')}/hr",
        _rate(rates, "rds_db_t4g_micro_hour"),
    )
    add(
        running,
        "RDS storage (20 GB gp2)",
        f"20 GB x ${_rate(rates, 'rds_gp2_gb_month')}/GB-month",
        20 * _rate(rates, "rds_gp2_gb_month") / HOURS_PER_MONTH,
        note="Billed monthly; shown per hour for comparability.",
    )
    add(
        running,
        "EC2 Redis host (t4g.micro)",
        f"1 instance x ${_rate(rates, 'ec2_t4g_micro_hour')}/hr",
        _rate(rates, "ec2_t4g_micro_hour"),
    )
    add(
        running,
        "Application Load Balancer",
        f"${_rate(rates, 'alb_hour')}/hr + {usage.alb_lcus} LCU x "
        f"${_rate(rates, 'alb_lcu_hour')}/LCU-hr",
        _rate(rates, "alb_hour") + usage.alb_lcus * _rate(rates, "alb_lcu_hour"),
        note="The single largest hourly line, and the reason the stack is torn down.",
    )
    add(
        running,
        "Kinesis Data Streams",
        f"{usage.kinesis_shards} shard x ${_rate(rates, 'kinesis_shard_hour')}/shard-hr",
        usage.kinesis_shards * _rate(rates, "kinesis_shard_hour"),
    )

    fargate_hourly = sum(
        vcpu * _rate(rates, "fargate_arm_vcpu_hour") + gb * _rate(rates, "fargate_arm_gb_hour")
        for _, vcpu, gb in usage.fargate_tasks
    )
    task_summary = ", ".join(
        f"{name} {vcpu} vCPU/{gb} GB" for name, vcpu, gb in usage.fargate_tasks
    )
    add(
        running,
        "Fargate tasks (ARM)",
        f"{task_summary} at ${_rate(rates, 'fargate_arm_vcpu_hour')}/vCPU-hr + "
        f"${_rate(rates, 'fargate_arm_gb_hour')}/GB-hr",
        fargate_hourly,
    )

    # Kinesis PUT payload units: one per 25 KB, minimum one per record.
    put_units_per_second = usage.transactions_per_second * max(
        1, -(-usage.kinesis_record_bytes // 25_000)
    )
    add(
        running,
        "Kinesis PUT payload units",
        f"{usage.transactions_per_second:g} txn/s x 3,600 x "
        f"${_rate(rates, 'kinesis_put_payload_unit'):.2e}/unit",
        put_units_per_second * 3600 * _rate(rates, "kinesis_put_payload_unit"),
    )

    # Lambda: one invocation per batch.
    invocations_per_hour = usage.transactions_per_second * 3600 / usage.lambda_batch_size
    gb_seconds_per_hour = (
        invocations_per_hour
        * (usage.lambda_duration_ms_per_batch / 1000.0)
        * (usage.lambda_memory_mb / 1024.0)
    )
    add(
        running,
        "Lambda (arm64, streaming scorer)",
        f"{invocations_per_hour:,.0f} invocations/hr x "
        f"{usage.lambda_duration_ms_per_batch:g} ms x {usage.lambda_memory_mb} MB",
        gb_seconds_per_hour * _rate(rates, "lambda_arm_gb_second")
        + invocations_per_hour * _rate(rates, "lambda_arm_request"),
        note=(
            "Duration measured from CloudWatch."
            if usage.lambda_duration_measured
            else "Duration is an assumption — no measurement session has run yet."
        ),
    )

    add(
        running,
        "Public IPv4 addresses",
        f"{usage.public_ipv4_addresses} x ${_rate(rates, 'public_ipv4_hour')}/hr",
        usage.public_ipv4_addresses * _rate(rates, "public_ipv4_hour"),
        note="Charged since Feb 2024 for every in-use public IPv4 address.",
    )

    # ── Idle: what accrues with everything torn down ───────────────────────
    add(
        idle,
        "S3 (artifacts, parquet, drift reports, TF state)",
        f"{usage.s3_gb:g} GB x ${_rate(rates, 's3_standard_gb_month')}/GB-month",
        usage.s3_gb * _rate(rates, "s3_standard_gb_month") / HOURS_PER_MONTH,
        is_idle=True,
    )
    add(
        idle,
        "ECR (three container images)",
        f"{usage.ecr_gb:g} GB x ${_rate(rates, 'ecr_storage_gb_month')}/GB-month",
        usage.ecr_gb * _rate(rates, "ecr_storage_gb_month") / HOURS_PER_MONTH,
        is_idle=True,
        note="prevent_destroy keeps these across teardowns, by design.",
    )
    add(
        idle,
        "Secrets Manager",
        f"{usage.secrets_count} secrets x ${_rate(rates, 'secretsmanager_secret_month')}/month",
        usage.secrets_count * _rate(rates, "secretsmanager_secret_month") / HOURS_PER_MONTH,
        is_idle=True,
    )
    add(
        idle,
        "CloudWatch Logs",
        f"{usage.cloudwatch_gb_per_month:g} GB/month x "
        f"${_rate(rates, 'cloudwatch_logs_ingest_gb')}/GB ingested",
        usage.cloudwatch_gb_per_month * _rate(rates, "cloudwatch_logs_ingest_gb") / HOURS_PER_MONTH,
        is_idle=True,
        note="7-day retention on all three log groups.",
    )

    running_hourly = sum(item.hourly_usd for item in running)
    idle_hourly = sum(item.hourly_usd for item in idle)

    # ── Cost per 1,000 predictions ─────────────────────────────────────────
    # Marginal cost on each path: what one more thousand predictions adds,
    # given the fixed infrastructure is already running. Reported alongside the
    # fully-loaded figure, which is the one that actually hits the bill.
    streaming_components = {
        "Kinesis PUT payload units",
        "Lambda (arm64, streaming scorer)",
    }
    streaming_marginal_hourly = sum(
        i.hourly_usd for i in running if i.component in streaming_components
    )
    streaming_predictions_per_hour = usage.transactions_per_second * 3600
    api_predictions_per_hour = usage.api_requests_per_second * 3600

    fargate_api_hourly = next(
        (
            vcpu * _rate(rates, "fargate_arm_vcpu_hour") + gb * _rate(rates, "fargate_arm_gb_hour")
            for name, vcpu, gb in usage.fargate_tasks
            if name == "api"
        ),
        0.0,
    )
    alb_hourly = next(i.hourly_usd for i in running if i.component.startswith("Application Load"))

    per_1k = {
        "streaming_marginal_usd": round(
            streaming_marginal_hourly / streaming_predictions_per_hour * 1000, 6
        ),
        "streaming_fully_loaded_usd": round(
            running_hourly / streaming_predictions_per_hour * 1000, 6
        ),
        "synchronous_marginal_usd": round(
            (fargate_api_hourly + alb_hourly) / api_predictions_per_hour * 1000, 6
        ),
        "streaming_throughput_per_second": usage.transactions_per_second,
        "api_throughput_per_second": usage.api_requests_per_second,
        "throughput_measured": bool(usage.api_requests_measured),
        "measured_latency": {
            "lambda_warm_p50_ms": 43.8,
            "lambda_warm_p95_ms": 58.0,
            "lambda_cold_ms": 61412.1,
            "streaming_ingest_to_decision_p50_ms": 610,
            "api_p50_ms": 110,
            "api_p95_ms": 420,
            "source": "reports/perf/2026-10-19/",
        },
        "explanation": (
            "Marginal cost is what one more thousand predictions adds while the stack "
            "is already up: for the streaming path that is Kinesis PUT units plus "
            "Lambda, and for the synchronous path the Fargate API task plus the ALB. "
            "Fully loaded divides the entire running cost of the stack across the "
            "streaming throughput, which is the honest number for a system whose cost "
            "is almost entirely fixed. The gap between them is the point: at this "
            "scale, prediction volume is nearly free and the infrastructure is not."
        ),
    }

    caveats = [
        "Rates are real (AWS Price List API, ap-south-1, on-demand). Usage figures "
        "are assumptions unless explicitly marked as measured.",
        "Data transfer out to the internet is not modelled. At this volume it is "
        "within or near the 100 GB/month free allowance.",
        "Free-tier allowances are deliberately ignored. They expire, and a cost "
        "estimate that depends on them describes a temporary situation.",
        "Taxes (18% GST in India) are not included.",
        "The stack is torn down between sessions, so the monthly figure for the "
        "running line items is what it *would* cost left on for 730 hours — not a "
        "bill anyone has received.",
    ]
    if not usage.lambda_duration_measured:
        caveats.append(
            "Lambda duration is a placeholder. Run scripts/lambda_latency.py after a "
            "measurement session and re-run `make cost` to replace it."
        )
    if not usage.api_requests_measured:
        caveats.append(
            "Throughput is a placeholder. The load test in load_test/locustfile.py "
            "produces the measured figure."
        )

    return CostEstimate(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        region=rates_payload.get("region", "ap-south-1"),
        pricing_fetched_at=rates_payload.get("fetched_at", "unknown"),
        assumptions=usage.to_dict(),
        running=[i.to_dict() for i in running],
        idle=[i.to_dict() for i in idle],
        per_1k_predictions=per_1k,
        totals={
            "running_hourly_usd": round(running_hourly, 4),
            "running_monthly_usd": round(running_hourly * HOURS_PER_MONTH, 2),
            "idle_hourly_usd": round(idle_hourly, 6),
            "idle_monthly_usd": round(idle_hourly * HOURS_PER_MONTH, 2),
            "total_if_always_on_monthly_usd": round(
                (running_hourly + idle_hourly) * HOURS_PER_MONTH, 2
            ),
        },
        measured_inputs=measured,
        caveats=caveats,
    )


def render_markdown(estimate: CostEstimate) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Cost estimate")
    add("")
    add(
        f"**This is an estimate.** Generated {estimate.generated_at} by "
        f"`make cost` from `reports/cost/pricing-inputs.json`, which "
        f"`scripts/fetch_pricing.py` pulled from the AWS Price List API on "
        f"{estimate.pricing_fetched_at} for `{estimate.region}`."
    )
    add("")
    add(
        "Every **rate** below is real and carries its AWS SKU in the pricing-inputs "
        "file. Every **usage** figure is an assumption unless marked measured — "
        "those are listed in full at the bottom."
    )
    add("")

    totals = estimate.totals
    add("## The three numbers")
    add("")
    add("| | Hourly | Monthly (730 h) |")
    add("|---|---:|---:|")
    add(
        f"| **Stack running** | ${totals['running_hourly_usd']:.4f} | "
        f"${totals['running_monthly_usd']:,.2f} |"
    )
    add(
        f"| **Idle (torn down)** | ${totals['idle_hourly_usd']:.5f} | "
        f"${totals['idle_monthly_usd']:,.2f} |"
    )
    add(f"| **If left on permanently** | — | ${totals['total_if_always_on_monthly_usd']:,.2f} |")
    add("")
    add(
        f"The gap between ${totals['running_monthly_usd']:,.0f}/month and "
        f"${totals['idle_monthly_usd']:,.2f}/month is the entire reason this project "
        f"tears the stack down between sessions."
    )
    add("")

    add("## Running: per hour with the stack up")
    add("")
    add("| Component | Basis | $/hour | $/month |")
    add("|---|---|---:|---:|")
    for item in sorted(estimate.running, key=lambda i: -i["hourly_usd"]):
        add(
            f"| {item['component']} | {item['basis']} | "
            f"${item['hourly_usd']:.5f} | ${item['monthly_usd']:,.2f} |"
        )
    add("")
    for item in estimate.running:
        if item["note"]:
            add(f"- **{item['component']}:** {item['note']}")
    add("")

    add("## Idle: with everything torn down")
    add("")
    add("| Component | Basis | $/month |")
    add("|---|---|---:|")
    for item in sorted(estimate.idle, key=lambda i: -i["monthly_usd"]):
        add(f"| {item['component']} | {item['basis']} | ${item['monthly_usd']:,.2f} |")
    add("")
    for item in estimate.idle:
        if item["note"]:
            add(f"- **{item['component']}:** {item['note']}")
    add("")

    per_1k = estimate.per_1k_predictions
    add("## Cost per 1,000 predictions")
    add("")
    measured_note = (
        "measured" if per_1k["throughput_measured"] else "**assumed — not yet measured**"
    )
    add(
        f"At {per_1k['streaming_throughput_per_second']:g} txn/s streaming and "
        f"{per_1k['api_throughput_per_second']:g} req/s synchronous ({measured_note}):"
    )
    add("")
    add("| Path | Marginal | Fully loaded |")
    add("|---|---:|---:|")
    add(
        f"| Streaming (Kinesis + Lambda) | ${per_1k['streaming_marginal_usd']:.6f} | "
        f"${per_1k['streaming_fully_loaded_usd']:.4f} |"
    )
    add(f"| Synchronous (ALB + Fargate) | ${per_1k['synchronous_marginal_usd']:.6f} | — |")
    add("")
    add(per_1k["explanation"])
    add("")

    add("## Assumptions")
    add("")
    add("| Input | Value |")
    add("|---|---|")
    for key, value in estimate.assumptions.items():
        add(f"| `{key}` | {value} |")
    add("")

    add("## Caveats")
    add("")
    for caveat in estimate.caveats:
        add(f"- {caveat}")
    add("")

    if estimate.measured_inputs:
        add("## Measured inputs used")
        add("")
        for key, value in estimate.measured_inputs.items():
            add(f"- **{key}:** {value}")
        add("")
    else:
        add("## Cross-check against an actual bill")
        add("")
        add(
            "Not yet done. The AWS measurement session (§7.3 of the polish plan) "
            "should be cross-checked against Cost Explorer for the day it ran, and "
            "the difference recorded here. An estimate that has never been compared "
            "to a bill is a spreadsheet, not a measurement."
        )
        add("")

    return "\n".join(lines)


def run(
    pricing_path: Path = PRICING_INPUTS,
    output_dir: Path = COST_DIR,
    usage: UsageAssumptions | None = None,
) -> CostEstimate:
    estimate = build_estimate(load_rates(pricing_path), usage=usage)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "estimate.json").write_text(json.dumps(estimate.to_dict(), indent=2) + "\n")
    (output_dir / "estimate.md").write_text(render_markdown(estimate))
    logger.info("Wrote %s/estimate.json and estimate.md", output_dir)
    return estimate


def main() -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pricing", type=Path, default=PRICING_INPUTS)
    parser.add_argument("--out", type=Path, default=COST_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    estimate = run(args.pricing, args.out)

    totals = estimate.totals
    print(
        f"\nRunning: ${totals['running_hourly_usd']:.4f}/hr (${totals['running_monthly_usd']:,.2f}/mo)"
    )
    print(f"Idle:    ${totals['idle_monthly_usd']:,.2f}/mo")
    print(f"\nWrote {args.out}/estimate.md")


if __name__ == "__main__":  # pragma: no cover
    main()
