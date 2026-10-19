# Real-Time UPI Fraud Detection — MLOps Pipeline

[![CI](https://github.com/pranav-j/payment_fraud_detection_mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/pranav-j/payment_fraud_detection_mlops/actions/workflows/ci.yml)

An end-to-end MLOps pipeline that takes a fraud-detection model from notebook to a
production-shaped system on AWS: streaming inference, a feature store with
online/offline parity, drift monitoring with alerting, scheduled retraining, and the
entire cloud stack defined as infrastructure-as-code.

This is a portfolio project. It uses the public
[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) dataset — synthetic
mobile-money transactions — as a stand-in for UPI payment data. The goal was not to
build the best possible fraud model. It was to build the *system around* a model the
way a real team would, and to be able to defend every architectural decision in an
interview.

---

## What this demonstrates

- Experiment tracking and a model registry with alias-based promotion (`@production`)
- Two serving patterns, each on the compute that fits it — event-driven streaming and
  synchronous request/response
- A feature store that enforces train/serve consistency, the failure mode that quietly
  breaks most real ML systems
- Scheduled orchestration: feature materialization, automated retraining behind a
  promotion gate, and drift detection
- Drift monitoring wired through to metrics dashboards and alerting
- The full AWS footprint as Terraform, with a verified teardown-and-rebuild cycle
- Architecture Decision Records (ADRs) capturing the *why* behind each choice — including
  the incidents and what they taught

---

## Architecture

```
                 PaySim dataset  (Kaggle · ~6.3M synthetic transactions)
                          |
            feature engineering  ->  XGBoost  ->  MLflow (tracking + registry)
                                                     |  metadata  -> RDS (Postgres)
                                                     |  artifacts -> S3
                                                     v
                                       models:/fraud-detector@production
                                                     |
              +--------------------------------------+-------------------------------+
              v                                                                      v
   STREAMING PATH  (production core)                        SYNCHRONOUS PATH  (HTTP exposure)
   producer -> Kinesis -> Lambda                            client -> ALB -> FastAPI on Fargate
                          |                                 returns the prediction in the response
                          | feature lookup -> Redis (Feast online store)
                          v
              decisions table (RDS) — append-only audit of every scored transaction
                          |
                          v
   MONITORING   Evidently drift check --> Prometheus --> Grafana
                                      \--> SNS --> email alert

   ORCHESTRATION (Prefect)    materialize · daily   |   retrain · weekly   |   drift · hourly
   FEATURE STORE (Feast)      offline = S3 parquet  |   online = Redis (EC2)
   INFRASTRUCTURE (Terraform) entire AWS stack as code · state in S3 · native S3 locking
```

### The three data stores


**S3** holds the big files: the trained model artifacts MLflow produces, the PaySim
parquet that Feast reads from, the Feast registry, drift-report HTML, and the Terraform
state. It is the durable filing cabinet — large, cheap, not fast.

**RDS (Postgres)** holds the structured records. Two distinct things live here: all of
MLflow's experiment and registry metadata (which runs happened, their metrics, and
crucially which model version carries the `@production` alias), and the `decisions`
table — an append-only audit log of every scored transaction.

**Redis (on EC2)** holds exactly one thing: the latest computed features for each repeat
sender, keyed by sender ID, for single-digit-millisecond lookups at inference time. Most
senders appear only once and have no history worth caching, so only the ~9,300 repeat
senders are materialized.

### Two serving paths, one model

The same registered model is served two ways, on the compute that fits each.

The **streaming path** is the production-shaped core. A producer pushes transactions into
a Kinesis stream; Lambda is triggered per batch, enriches each transaction with the
sender's historical features from Redis (via Feast), scores it, and writes the verdict to
the `decisions` table. Nothing waits on the result — it's fire-and-forget. This is the
path the rest of the system is built around: the audit log it populates is what
monitoring and retraining read from.

The **synchronous path** is how you would expose the same model for inline authorization,
where a caller needs a blocking yes/no before a payment completes. An HTTP request hits an
Application Load Balancer, which routes to a FastAPI container on ECS Fargate; the
prediction comes straight back in the response.

Both paths are real-time. They differ not in latency tier so much as in *who waits*: the
synchronous path blocks the caller for an answer, while the streaming path processes a
continuous flow with nobody waiting. Fargate is used for the synchronous path precisely
because a kept-warm container has no cold-start tail latency — the right instinct for an
inline gate — while Lambda fits the bursty, scale-to-zero streaming workload.

### Feature store and train/serve consistency

Fraud features like "transactions from this sender in the last 24 hours" can't be computed
from a single transaction in isolation — they need the sender's history. The danger is
computing those features one way in training and a different way at serving time;
that **train/serve skew** is a leading cause of silent production model failure.

Feast removes it by defining each feature once. The **offline store** (parquet in S3)
builds training data; the **online store** (Redis) serves it. A scheduled job materializes
features from offline to online. At inference time the streaming Lambda looks up the
sender's features from Redis and falls back to default zeros for first-time senders —
matching exactly what the model saw for first-time senders during training.

### Orchestration

Prefect runs three scheduled flows that close the loop:

- **Materialization** (daily) refreshes Redis from the S3 parquet so the online features
  don't go stale.
- **Retraining** (weekly) trains a fresh model, compares it to the current production
  model, and promotes only if it clears a gate — recall improvement of at least 2
  percentage points while holding precision at 95% or above. On its first real run the
  candidate was marginally worse and the gate correctly refused to promote it.
- **Drift detection** (hourly) bridges into monitoring.

### Monitoring and alerting

The drift flow pulls recent rows from the `decisions` table and compares their
distribution against the training baseline using **Evidently**. It writes a full HTML
report to S3, pushes the key drift metrics to **Prometheus** (via a Pushgateway, since the
flow is short-lived), which **Grafana** visualizes, and fires an **SNS** email alert when
drift crosses a threshold. The chain was verified end-to-end by injecting transactions
with amounts 100x normal — drift jumped to 100%, the dashboard lit up, and the alert
arrived.

### Infrastructure as code

The entire AWS footprint — VPC and security groups, RDS, the Redis EC2 instance, Kinesis,
Lambda, the ECS cluster and services, the ALB, IAM, Secrets Manager, SNS, CloudWatch — is
defined in Terraform. State lives in S3 with native S3 locking (`use_lockfile`), and
`prevent_destroy` guards the resources that hold data or images (S3, ECR). The full
destroy-and-rebuild cycle was exercised, not just `apply` — including recovering from a
self-inflicted RDS replacement and AWS's zombie-ENI behaviour, both documented in the ADRs.

### Networking and security

Everything on the data plane — the Lambda, the ECS tasks, the Redis EC2 instance, and
RDS — runs inside a VPC, and access between components is granted by **referencing
security groups, not by opening CIDR ranges**. The Lambda's security group is allowed into
the RDS and Redis security groups on only their service ports, so just the function can
reach the database and the cache; the data stores carry no `0.0.0.0/0` ingress. The ALB
sits in its own security group exposing only the HTTP listener.

Because the Lambda runs inside the VPC, it reaches S3 through an **S3 VPC Gateway
Endpoint** rather than the public internet — keeping that traffic on the AWS backbone and
avoiding the hourly cost of a NAT Gateway, the usual way VPC-bound compute reaches AWS
services. The RDS connection URI is held in **Secrets Manager** and injected at runtime,
never baked into an image or a task definition.

Two production gaps (both in the limitations below): RDS is left publicly accessible — with
only the developer IP allowed in its security group — for convenience during the build, and
the ALB serves plain HTTP rather than TLS.

---

## The model

XGBoost classifier on PaySim, with the decision threshold calibrated to maximise recall
subject to a precision floor (fraud is ~0.13% of transactions, so the threshold matters
far more than raw accuracy). The registered versions are:

| Registry version | Feature set | Threshold | Tagged recall | Tagged precision |
|---|---|---|---|---|
| v1, v2 | baseline | 0.9307 | 80.4% | 95.0% |
| v3 (`@production`) | enriched (Feast historical features) | 0.4234 | 84.5% | 95.0% |

Those are the tags carried on the model versions themselves, and they are **not** the
numbers to judge the model by. They came from a protocol that tuned the threshold on the
test set and reported no interval. The [Evaluation](#evaluation) section below has
numbers from a protocol that doesn't do that, and they are lower.

> **An earlier version of this README quoted "v1 80.4% → v2 84.5% recall" and called that
> jump "the entire justification for the feature store".** Re-measured properly — split by
> time on whole steps, threshold tuned on a validation split, bootstrap intervals — the
> enriched features improve PR-AUC by a clear margin (0.9523 → 0.9752, intervals don't
> overlap) but improve recall at 95% precision by 0.75 points, with intervals that overlap
> almost entirely. At the deployed threshold the two feature sets are indistinguishable.
> The feature store earns its place in this project as infrastructure — it is what makes
> train/serve consistency enforceable — but the recall claim did not survive the
> measurement, so it is gone.

---

## Evaluation

Every number here comes from [`reports/metrics/evaluation.json`](reports/metrics/evaluation.json),
regenerated by `make evaluate` from a committed script. The readable version with full
detail is [`reports/metrics/evaluation.md`](reports/metrics/evaluation.md). The protocol
is ADR-021.

### How the data is split

PaySim's `step` is a simulated hour, so the split is by time: train on the earliest
hours, validate on the next block, test on the latest. Boundaries land between whole
steps, so no simulated hour is divided across two splits.

| Split | Rows | Steps | Fraud | Fraud rate |
|---|---:|---|---:|---:|
| train | 3,820,599 | 1–281 | 3,193 | 0.0836% |
| validation | 1,248,498 | 282–354 | 762 | 0.0610% |
| test | 1,293,523 | 355–743 | 4,258 | 0.3292% |

The row counts and the step ranges disagree wildly, and that is the data rather than a
bug: PaySim's volume is heavily front-loaded, so 60% of rows is barely a third of the
hours. The fraud rate differs by 5x between validation and test for the same reason,
which matters below.

**The threshold is tuned on validation only.** The previous protocol tuned it on the
test set, which inflates the reported recall for a threshold nobody could have chosen in
advance.

### Baselines

All on the test split, which nothing was tuned on. Intervals are percentile bootstraps,
1,000 resamples.

| Model | PR-AUC [95% CI] | Recall @95% precision [95% CI] | Precision @ threshold | Recall @ threshold |
|---|---|---|---:|---:|
| Rule: large TRANSFER/CASH_OUT | 0.0831 [0.0755, 0.0920] | 0.0% [0.0%, 0.0%] | 19.5% | 11.2% |
| Logistic regression (v1 features) | 0.6677 [0.6545, 0.6806] | 43.1% [41.2%, 45.2%] | 99.7% | 35.4% |
| XGBoost (v1 features) | 0.9523 [0.9478, 0.9563] | 82.8% [81.3%, 84.0%] | 99.4% | 74.4% |
| XGBoost (v2 features) | 0.9752 [0.9728, 0.9777] | 83.6% [81.9%, 84.8%] | 99.3% | 74.3% |

Two columns, because "recall at 95% precision" is ambiguous and the difference is the
interesting part. **Recall @95% precision** is the best the ranking could do on test at
that precision floor — an upper bound found with hindsight. **Recall @ threshold** is
what the validation-tuned cut-off actually delivers. The ~9-point gap is the cost of
choosing a threshold before seeing the future: the validation window has a much lower
fraud rate than the test window, so the threshold comes out conservative and buys 99.3%
precision where only 95% was asked for.

**What the enriched features are worth.** v1 → v2 improves PR-AUC from 0.9523 to 0.9752
with non-overlapping intervals, so the ranking is genuinely better. Recall at 95%
precision improves by 0.75 points with intervals that overlap almost completely, and at
the deployed threshold v2 is a tenth of a point *behind*. The feature store is justified
here by train/serve consistency, not by a recall number.

**What a random split would have told you.** Run with a stratified random split instead,
the same v2 configuration reports 88.7% recall at 95% precision
against 83.6% on the time split — five points of pure
optimism from letting the model train on the future. `flows/retrain_model.py` still uses
a random split; that is a known gap, recorded rather than quietly fixed.

### Leakage

Excluded from every feature set:

| Column | Why |
|---|---|
| `isFlaggedFraud` | PaySim's own rule (transfers over 200,000). It is the simulator's output, not an observable input. |
| `nameOrig` | Sender account ID — used to *group by* when computing history, never as a feature. 6.3M near-unique values would be memorised, not learned. |
| `nameDest` | Destination account ID. Same reasoning. |

**The balance columns are kept, and they are the problem.** PaySim generates fraud by
emptying the origin account, so the balance columns don't correlate with the label —
they encode the generator's rule:

| Configuration | PR-AUC [95% CI] | Recall @95% precision [95% CI] |
|---|---|---|
| XGBoost (v2 features) | 0.9752 [0.9728, 0.9777] | 83.6% [81.9%, 84.8%] |
| XGBoost (v2 features, balance columns removed) | 0.2146 [0.2023, 0.2263] | 16.8% [15.5%, 18.2%] |
| Rule: PaySim balance signature | 0.9738 [0.9689, 0.9786] | 97.4% [96.9%, 97.8%] |

A three-condition rule anyone could write in a minute — a TRANSFER or CASH_OUT moving
exactly the origin account's whole balance, leaving it at zero — matches
**8,024 of 8,213 frauds (97.7%)** and
**1 of 6,354,407 legitimate transactions**. It beats
every trained model in the table. Strip those six columns out and the production
configuration falls from 0.975 PR-AUC to 0.215.

So the honest reading of every headline number on this page is: *this is what a model
scores on a simulator whose fraud rule is nearly deterministic.* Real payment data has
no column that clean. The no-balance row is the better guide to what the system would do
on real transactions, and the gap between the two is the single most important thing to
know about results on PaySim.

---

## Tech stack

| Concern | Tooling |
|---|---|
| Language / tooling | Python 3.12, `uv`, src-layout package |
| Modelling | XGBoost, scikit-learn |
| Tracking & registry | MLflow (RDS backend, S3 artifacts) |
| Streaming inference | AWS Kinesis + AWS Lambda (container image, arm64) |
| Synchronous inference | FastAPI on ECS Fargate, behind an ALB |
| Feature store | Feast (S3 offline + registry, Redis online) |
| Orchestration | Prefect (three scheduled flows) |
| Monitoring & alerting | Evidently, Prometheus, Grafana, SNS |
| Data / audit stores | S3, RDS Postgres, Redis (EC2) |
| Infrastructure | Terraform (S3 state, native S3 locking) |
| Packaging | Docker, ECR |
| Code quality | ruff, nbstripout, detect-secrets (pre-commit) |

---

## Repository layout

```
fraud-mlops/
├── notebooks/          # EDA, baseline model, feature engineering
├── src/fraud_mlops/    # importable package — inference, tracking, schemas (logic lives here)
├── lambda/             # streaming scorer: handler, Dockerfile, pinned requirements
├── feature_repo/       # Feast: entities, features, data sources, feature_store.yaml
├── flows/              # Prefect flows: materialize, retrain, detect_drift, deploy
├── monitoring/         # Prometheus config, Grafana dashboard
├── infra/              # Terraform for the full AWS stack (+ ECS task defs)
├── scripts/            # data download, transaction producer, seeding, drift injection
├── tests/              # pytest suite
├── docs/adr/           # Architecture Decision Records (decisions.md)
├── Dockerfile.*        # FastAPI and MLflow images
├── docker-compose*.yml # local stacks
├── Makefile            # task runner
└── pyproject.toml      # uv-managed project config
```

The guiding principle: **logic lives in `src/`, not in notebooks.** Notebooks are thin
wrappers that call into modules, which is what made promoting code into Prefect flows and
the Lambda handler straightforward.

---

## Running it

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/) for Python dependency management
- Docker (for building the Lambda / FastAPI / MLflow images)
- An AWS account with the CLI configured (`ap-south-1` in this build)
- Terraform for provisioning the cloud stack

### Local development

```bash
# install runtime + dev dependencies and pre-commit hooks
make install-dev

# configure environment (RDS, Redis, S3, MLflow URIs)
cp .env.example .env        # then fill in values

# download the PaySim dataset
make data

# explore / train
make notebook               # 01_eda → 02_baseline_model → 03_feature_engineering

# quality
make lint        # ruff + type checks
make test        # pytest
```

Most commands that touch AWS expect the environment to be loaded first:

```bash
set -a && source .env && set +a
```

### Provisioning the AWS stack

```bash
cd infra
terraform init
terraform apply          # review the plan before approving
```

After the stack is up, the data-plane is initialised by starting the MLflow server
(`scripts/run_mlflow_server.sh`), registering the model, materialising features into Redis
(`feast -c feature_repo materialize ...`), and deploying the Prefect flows
(`flows/deploy.py`). The streaming path can then be exercised end-to-end with
`scripts/produce_transactions.py`.

> **Note on cost:** stateful and billable resources (RDS, the Redis EC2 instance, the ALB,
> Kinesis) are stopped or destroyed between sessions to keep idle cost near zero. The ALB
> is the main hourly cost when running; it has no "stopped" state and is recreated on
> demand. Idle cost with everything torn down is roughly $1/month (S3, ECR, Secrets
> Manager, CloudWatch).

---

## Known limitations & what production would change

This is a learning project on synthetic data, and several choices were deliberate scope
decisions rather than what a real deployment would do:

- **The two serving paths are partly redundant here.** Both score the same model; the
  streaming path is the integrated core (it writes the decisions audit log that monitoring
  and retraining consume), while the synchronous FastAPI path is a demonstration of the
  request/response pattern and does not itself persist decisions. In production these would
  serve genuinely different functions — an inline authorization gate versus a
  near-real-time monitoring and feature-maintenance layer.
- **Kinesis runs a single shard.** Fine for the project's volume. At scale you would add
  shards and, more importantly, move windowed velocity-feature computation to a stateful
  stream processor (e.g. Apache Flink), which Lambda doesn't do natively.
- **Prefect and the monitoring stack run locally.** The flows and Prometheus/Grafana
  execute on a developer machine and reach out to AWS; production would host them on
  always-on infrastructure so schedules fire without a laptop.
- **Lambda cold start (~24s on first call) is unoptimised.** Acceptable because the
  streaming path is fire-and-forget; a latency-sensitive inline path would use provisioned
  concurrency.
- **No TLS on the ALB.** Acceptable for synthetic data with no PII; production would add an
  ACM certificate and an HTTPS listener (no application code changes required).
- **RDS is publicly accessible.** Convenient for connecting from a laptop during the build,
  with only the developer IP allowed in its security group; production would keep it in
  private subnets and reach it through a bastion or VPN.
- **Single-AZ RDS, no Multi-AZ.** A zone failure would take the backend offline. A
  production deployment would enable Multi-AZ.

Naming these is the point: the project is a faithful miniature of a real system, and the
gaps are understood rather than hidden.

---

## Architecture decision records

Every significant choice — and several instructive failures — is recorded in
[`docs/adr/decisions.md`](docs/adr/decisions.md). A few worth highlighting:

- **Cloud-native state (ADR-013):** why model artifacts and registry metadata had to live
  in S3 + RDS rather than on any one machine's filesystem, so the same
  `models:/fraud-detector@production` resolves identically from a laptop, Fargate, or
  Lambda.
- **Feast registry on S3, not SQL (ADR-016):** the SQL registry caused per-invocation
  Postgres connection timeouts from Lambda; an S3 file registry fixed it. Also covers the
  train/serve consistency design.
- **Terraform IaC (ADR-019):** adopting IaC on an existing stack — including the incident
  where an `apply` on an imported resource replaced (destroyed) the production RDS, and
  the zombie Lambda ENIs that blocked a security-group deletion. The lesson —
  `terraform plan | grep destroy` before every apply on imported resources — is the kind
  of scar that only comes from doing it.

---

## License

MIT
