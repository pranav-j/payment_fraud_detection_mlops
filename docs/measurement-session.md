# AWS measurement session — checklist

Everything in this project that needs real infrastructure is batched into one
session, so the stack is up for as short a time as possible. This is the
checklist for that session; `scripts/measurement_session.sh` is the runner.

**Nothing here runs automatically.** Every step that creates or destroys a
billable resource stops and asks for a typed confirmation, and prints the
Terraform plan — plus every destroy and replace line on its own — first. That
check exists because skipping it once destroyed the production RDS (ADR-019).

Expected cost: the stack runs at about **$0.154/hour** (see
[`reports/cost/estimate.md`](../reports/cost/estimate.md)), so a two-hour
session is roughly **$0.31** plus data transfer.

---

## Before you start

- [ ] `aws sts get-caller-identity` returns the expected account
- [ ] `.env` exists and is filled in
- [ ] Working tree is clean, so a config change during the session is visible
- [ ] `make lint && make test` pass
- [ ] Decide the session date; everything lands in `reports/perf/<date>/`

```bash
./scripts/measurement_session.sh plan     # creates nothing
```

Read the plan output. If it wants to destroy or replace anything, stop and work
out why before going further.

---

## 1. Bring the stack up

```bash
./scripts/measurement_session.sh up
```

- [ ] Plan reviewed, including the destroy/replace section
- [ ] Confirmation typed, apply completed (~15 minutes)
- [ ] Outputs captured (`alb_dns_name`, `rds_endpoint`, `redis_private_ip`)

Then the data plane, which Terraform does not do:

- [ ] `.env` updated with the new RDS host, Redis IP and ALB DNS
- [ ] `python -m fraud_mlops.db migrate` — creates `decisions`, `labels`,
      `ground_truth` and the latency columns
- [ ] `bash scripts/run_mlflow_server.sh` running in its own terminal
- [ ] A model carries the `@production` alias (check `/model-info` through the
      ALB, or the MLflow UI)
- [ ] `python flows/materialize_features.py` — Redis has keys
- [ ] One warm-up request to the Lambda, so the first measurement is not
      entirely cold start

---

## 2. Measure

```bash
./scripts/measurement_session.sh measure
```

Writes to `reports/perf/<date>/`:

- [ ] `session.json` — parameters and instance sizes, so the numbers can be read
      a year later
- [ ] `locust-5rps*`, `locust-20rps*`, `locust-50rps*` — p50/p95/p99 and error
      rates at each level
- [ ] `streaming_latency.{json,md}` — ingest-to-decision percentiles from the
      `decisions` table
- [ ] `lambda_latency.{json,md}` — cold and warm populations reported separately

Watch for, and write down:

- [ ] Error rate at 50 req/s. Anything non-zero is a finding, not a nuisance —
      note whether it is the ALB, the task, or the model
- [ ] Whether the Fargate task CPU saturates (ECS console → Metrics)
- [ ] The cold-start count in the Lambda report. If it is near zero, the warm
      numbers are solid and the cold figure rests on very few samples

---

## 3. The demo (optional, but it is the interesting one)

```bash
./scripts/measurement_session.sh demo -- --delay-hours 0.05
```

- [ ] Phase 1 traffic produced, ground truth recorded
- [ ] Phase 2 shifted-pattern traffic produced
- [ ] Label delay elapsed
- [ ] Performance flow ran and the SNS alert arrived
- [ ] **Screenshot:** Grafana realized-performance panel showing the recall drop
- [ ] **Screenshot:** the drift panels staying flat over the same period — this
      is the whole point, and it needs the two side by side

---

## 4. Screenshots for the README

Capture to `reports/figures/` with these exact names:

- [ ] `grafana-drift-spike.png` — drift share jumping after `scripts/inject_drift.py`
- [ ] `grafana-realized-performance.png` — realized recall falling, with the
      "not yet matured" region visible
- [ ] `mlflow-registry-aliases.png` — the registry showing versions and the
      `@production` alias
- [ ] `sns-alert-email.png` *(optional)* — the alert as it arrives

Crop out anything carrying the AWS account ID, the RDS endpoint or an email
address. `detect-secrets` does not read PNGs.

---

## 5. Tear down

```bash
./scripts/measurement_session.sh down
./scripts/measurement_session.sh verify
```

- [ ] Destroy plan reviewed, including every destroy line
- [ ] Confirmation typed, destroy completed
- [ ] `verify` shows no RDS instance, no running EC2, no load balancer, no
      Kinesis stream
- [ ] S3, ECR, Secrets Manager and the log groups remain — that is the idle
      footprint, and it is deliberate

If destroy fails on the Lambda security group, it is the zombie-ENI problem in
ADR-019. Wait, then retry; do not force it.

---

## 6. Afterwards

- [ ] `reports/perf/<date>/` committed
- [ ] Cost estimate updated with the measured throughput and Lambda duration,
      then `make cost` re-run
- [ ] Cross-check against Cost Explorer for the session day, and record the
      difference in `reports/cost/estimate.md`. An estimate never compared to a
      bill is a spreadsheet
- [ ] README performance section updated from the generated reports
- [ ] Screenshots committed and linked
