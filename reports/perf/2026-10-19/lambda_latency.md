# Lambda latency

Log group `/aws/lambda/fraud-mlops-scorer`, 2026-10-19T13:48:30.857437+00:00 to 2026-10-19T16:48:30.857437+00:00.

- **826 invocations** (1 cold, 825 warm — 0.1% cold)
- Memory configured 1024 MB, peak used 590 MB

| Population | n | p50 | p95 | p99 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| all | 826 | 43.8 ms | 58.0 ms | 131.0 ms | 31.7 ms | 61412.1 ms |
| warm | 825 | 43.8 ms | 58.0 ms | 130.2 ms | 31.7 ms | 291.1 ms |
| cold (init timed out, folded into invoke) | 1 | 61412.1 ms | 61412.1 ms | 61412.1 ms | 61412.1 ms | 61412.1 ms |

> **This function's initialisation times out.** 1 INIT_REPORT line(s) show `Status: timeout` at 10000 ms — Lambda's init phase gives a function 10 seconds, and loading MLflow, the model artifact and the Feast registry takes longer. Lambda then re-runs initialisation inside the first invocation, so that invocation's REPORT line carries no `Init Duration` and the whole init cost lands in `Duration`. The last row above is the real cold-start latency.

Cold and warm are reported separately because they are different distributions, not a tail of one. `init` is the container bootstrap — importing MLflow, resolving the `@production` alias, loading the model artifact and the Feast registry — and it happens once per container. `init + handler` is what the first record of a new container waits for.
