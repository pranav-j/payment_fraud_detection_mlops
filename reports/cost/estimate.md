# Cost estimate

**This is an estimate.** Generated 2026-10-19 14:56:47 UTC by `make cost` from `reports/cost/pricing-inputs.json`, which `scripts/fetch_pricing.py` pulled from the AWS Price List API on 2026-10-19 14:55:06 UTC for `ap-south-1`.

Every **rate** below is real and carries its AWS SKU in the pricing-inputs file. Every **usage** figure is an assumption unless marked measured — those are listed in full at the bottom.

## The three numbers

| | Hourly | Monthly (730 h) |
|---|---:|---:|
| **Stack running** | $0.1543 | $112.61 |
| **Idle (torn down)** | $0.00280 | $2.04 |
| **If left on permanently** | — | $114.65 |

The gap between $113/month and $2.04/month is the entire reason this project tears the stack down between sessions.

## Running: per hour with the stack up

| Component | Basis | $/hour | $/month |
|---|---|---:|---:|
| Fargate tasks (ARM) | mlflow 1.0 vCPU/2.0 GB, api 0.5 vCPU/1.0 GB at $0.02383/vCPU-hr + $0.00261/GB-hr | $0.04358 | $31.81 |
| Application Load Balancer | $0.0239/hr + 1.0 LCU x $0.008/LCU-hr | $0.03190 | $23.29 |
| Public IPv4 addresses | 5 x $0.005/hr | $0.02500 | $18.25 |
| RDS (db.t4g.micro, Single-AZ) | 1 instance x $0.021/hr | $0.02100 | $15.33 |
| Kinesis Data Streams | 1 shard x $0.0175/shard-hr | $0.01750 | $12.78 |
| Lambda (arm64, streaming scorer) | 1,800 invocations/hr x 250 ms x 1024 MB | $0.00576 | $4.20 |
| EC2 Redis host (t4g.micro) | 1 instance x $0.0056/hr | $0.00560 | $4.09 |
| RDS storage (20 GB gp2) | 20 GB x $0.131/GB-month | $0.00359 | $2.62 |
| Kinesis PUT payload units | 5 txn/s x 3,600 x $1.85e-08/unit | $0.00033 | $0.24 |

- **RDS storage (20 GB gp2):** Billed monthly; shown per hour for comparability.
- **Application Load Balancer:** The single largest hourly line, and the reason the stack is torn down.
- **Lambda (arm64, streaming scorer):** Duration is an assumption — no measurement session has run yet.
- **Public IPv4 addresses:** Charged since Feb 2024 for every in-use public IPv4 address.

## Idle: with everything torn down

| Component | Basis | $/month |
|---|---|---:|
| Secrets Manager | 2 secrets x $0.4/month | $0.80 |
| CloudWatch Logs | 1 GB/month x $0.67/GB ingested | $0.67 |
| ECR (three container images) | 4.5 GB x $0.1/GB-month | $0.45 |
| S3 (artifacts, parquet, drift reports, TF state) | 5 GB x $0.025/GB-month | $0.12 |

- **ECR (three container images):** prevent_destroy keeps these across teardowns, by design.
- **CloudWatch Logs:** 7-day retention on all three log groups.

## Cost per 1,000 predictions

At 5 txn/s streaming and 5 req/s synchronous (**assumed — not yet measured**):

| Path | Marginal | Fully loaded |
|---|---:|---:|
| Streaming (Kinesis + Lambda) | $0.000339 | $0.0086 |
| Synchronous (ALB + Fargate) | $0.002579 | — |

Marginal cost is what one more thousand predictions adds while the stack is already up: for the streaming path that is Kinesis PUT units plus Lambda, and for the synchronous path the Fargate API task plus the ALB. Fully loaded divides the entire running cost of the stack across the streaming throughput, which is the honest number for a system whose cost is almost entirely fixed. The gap between them is the point: at this scale, prediction volume is nearly free and the infrastructure is not.

## Assumptions

| Input | Value |
|---|---|
| `kinesis_shards` | 1 |
| `transactions_per_second` | 5.0 |
| `kinesis_record_bytes` | 400 |
| `lambda_memory_mb` | 1024 |
| `lambda_batch_size` | 10 |
| `lambda_duration_ms_per_batch` | 250.0 |
| `lambda_duration_measured` | False |
| `fargate_tasks` | [['mlflow', 1.0, 2.0], ['api', 0.5, 1.0]] |
| `api_requests_per_second` | 5.0 |
| `api_requests_measured` | False |
| `alb_lcus` | 1.0 |
| `s3_gb` | 5.0 |
| `ecr_gb` | 4.5 |
| `secrets_count` | 2 |
| `cloudwatch_gb_per_month` | 1.0 |
| `public_ipv4_addresses` | 5 |

## Caveats

- Rates are real (AWS Price List API, ap-south-1, on-demand). Usage figures are assumptions unless explicitly marked as measured.
- Data transfer out to the internet is not modelled. At this volume it is within or near the 100 GB/month free allowance.
- Free-tier allowances are deliberately ignored. They expire, and a cost estimate that depends on them describes a temporary situation.
- Taxes (18% GST in India) are not included.
- The stack is torn down between sessions, so the monthly figure for the running line items is what it *would* cost left on for 730 hours — not a bill anyone has received.
- Lambda duration is a placeholder. Run scripts/lambda_latency.py after a measurement session and re-run `make cost` to replace it.
- Throughput is a placeholder. The load test in load_test/locustfile.py produces the measured figure.

## Cross-check against an actual bill

Not yet done. The AWS measurement session (§7.3 of the polish plan) should be cross-checked against Cost Explorer for the day it ran, and the difference recorded here. An estimate that has never been compared to a bill is a spreadsheet, not a measurement.
