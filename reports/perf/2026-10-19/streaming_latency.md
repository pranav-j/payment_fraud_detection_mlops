# Streaming latency (ingest to decision)

Window: last 2.0h, as of 2026-10-19T16:20:20.025577+00:00.

- **1,200 decisions**, 1,200 with a recorded arrival time

| p50 | p95 | p99 | min | max | mean |
|---:|---:|---:|---:|---:|---:|
| 610 ms | 40586 ms | 67539 ms | 48 ms | 72436 ms | 4390 ms |

| Model version | n | p50 | p95 |
|---|---:|---:|---:|
| v2 | 1,200 | 610 ms | 40586 ms |

This is the whole pipeline: Kinesis receipt to the decision being committed to Postgres. It includes batch formation and any wait for a Lambda container, so it is always larger than the function's own reported duration.
