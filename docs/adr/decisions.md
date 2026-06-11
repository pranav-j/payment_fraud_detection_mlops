# Architecture Decision Records

**Project:** Real-time UPI Fraud Detection MLOps Pipeline
**Author:** [Your Name]
**Last Updated:** [Date]
**Status:** Living document — updated as decisions evolve

---

## What is an ADR?

An Architecture Decision Record captures a single significant architectural decision: the context that forced the choice, the alternatives considered, the decision made, and its consequences. ADRs are written *at the time of the decision* (not retroactively) and immutable thereafter — if a decision changes, a new ADR supersedes the old one.

This document contains the ADRs that shaped this project. Each ADR follows the format: **Context → Decision → Alternatives Considered → Consequences**.

---

## ADR-001: Use streaming inference (Kinesis + Lambda) instead of batch

**Status:** Accepted

### Context

Fraud detection has an inherent latency requirement: a fraudulent transaction blocked after settlement is worthless. Industry benchmarks suggest fraud decisions need to land within 100-300ms of the transaction event for the block to be enforceable upstream. Batch inference (e.g., scoring transactions every 5 minutes) is operationally simpler but cannot meet this requirement.

### Decision

Use AWS Kinesis Data Streams as the event ingestion layer and AWS Lambda as the inference compute layer. Each transaction event flows: producer → Kinesis → Lambda consumer → feature lookup → model inference → decision logged to RDS.

### Alternatives Considered

1. **Batch inference (every 5 min via Prefect)** — Rejected. Fails the latency requirement. Useful only for offline analytics, not enforcement.
2. **Synchronous REST API (FastAPI only)** — Rejected as primary path. Tightly couples the producer to the inference service; one slow inference blocks the producer. Useful as a fallback for clients that need synchronous responses, so retained as a secondary path on ECS Fargate.
3. **Apache Kafka + Kafka Streams** — Rejected for this project. More powerful than Kinesis but operationally heavier (requires Zookeeper or KRaft, broker management, monitoring). The 6-8 week timeline doesn't justify the complexity. In a production setting at scale (>10K events/sec), Kafka would likely win.
4. **AWS MSK (Managed Kafka)** — Rejected. Provides Kafka without the operational burden but minimum cost is ~$150/month for a small cluster. Outside the project budget.

### Consequences

**Positive:**
- Sub-200ms p99 inference latency achievable.
- Decoupled producer and consumer — producer doesn't wait for inference.
- Lambda scales automatically with event volume; no capacity planning.
- Kinesis retains events for 24 hours by default, enabling replay if Lambda fails.

**Negative:**
- Cold-start latency on Lambda (mitigated via container images + provisioned concurrency for the production path).
- Kinesis has a 1MB/sec/shard write limit; need to monitor throughput.
- Per-shard cost (~$0.015/hour) accrues even when idle; must be torn down between dev sessions.
- Debugging async streaming systems is harder than synchronous APIs.

**Mitigations:**
- Lambda packaged as container image from ECR to keep cold start under 2s.
- Provisioned concurrency = 1 in production stack to eliminate cold starts.
- Kinesis stream destroyed via `terraform destroy` between dev sessions.

---

## ADR-002: Use Feast as a feature store with Redis online + Postgres offline

**Status:** Accepted

### Context

Fraud detection features are time-windowed (e.g., "transaction count for sender in last 1h"). These features must be computed identically during training (over historical data) and serving (over the live event window). Computing them in two separate codebases is the classic train-serve skew bug — the #1 cause of production ML model degradation.

### Decision

Use Feast as the feature store. Define features once in Feast's declarative format. Use Postgres (RDS) as the offline store for historical/training queries and Redis (small EC2 instance with Redis) as the online store for sub-50ms lookups during inference.

### Alternatives Considered

1. **Compute features inline in the Lambda function** — Rejected. Forces feature logic to live in the inference code, duplicated in training code. Train-serve skew waiting to happen.
2. **Tecton (managed feature store)** — Rejected. Excellent product but not free; minimum tier is enterprise-priced.
3. **Feathr (LinkedIn's open-source feature store)** — Rejected. Less mature documentation, smaller community than Feast.
4. **DIY: Redis cache populated by a Prefect job, queried by Lambda** — Rejected. This is essentially a worse, hand-rolled Feast. No declarative feature definitions, no point-in-time correctness for historical training, no metadata.
5. **Amazon SageMaker Feature Store** — Considered. Native AWS, well-integrated. Rejected primarily for cost (per-feature-group pricing) and because demonstrating Feast knowledge is more transferable across employers than SageMaker-specific features.
6. **Use ElastiCache for the online store instead of Redis on EC2** — Rejected for this project. ElastiCache is the production-correct choice but adds ~₹500/month. A `t4g.nano` EC2 running Redis costs ~₹250/month and is sufficient for the project's load.

### Consequences

**Positive:**
- Train-serve consistency enforced by design — features defined once.
- Point-in-time correctness for historical training queries (critical for time-series features like rolling aggregates).
- Demonstrably understands a non-trivial production ML concept (feature stores) — strong interview signal.
- Feast is the most widely-used open-source feature store; transferable knowledge.

**Negative:**
- Adds operational complexity: a Postgres instance, a Redis instance, a Feast registry, and materialization flows to maintain.
- Feast's online store needs periodic materialization from offline store; another Prefect flow to manage.
- Sub-50ms p99 online lookup requires Redis to be in the same VPC/AZ as Lambda — networking config needed.

**Mitigations:**
- Feast materialization runs as a scheduled Prefect flow with monitoring.
- Lambda and Redis EC2 colocated in the same private subnet.
- For local dev, both Postgres and Redis run in Docker Compose; Feast configuration switches via environment variable.

---

## ADR-003: Use MLflow for experiment tracking and model registry

**Status:** Accepted

### Context

The project will produce many model variants over its lifecycle: baseline logistic regression, random forest, XGBoost variants, retrained models from weekly Prefect flows. Each needs to be tracked with hyperparameters, metrics, artifacts, and lineage to source code. The "winning" model needs to be discoverable by the inference path, with the ability to roll back if needed.

### Decision

Use MLflow as both the experiment tracking system and the model registry. Backend: SQLite for local dev, RDS Postgres for cloud deployments. Artifact store: S3.

### Alternatives Considered

1. **Weights & Biases (W&B)** — Excellent product, better UI for experiment comparison. Rejected primarily because the free tier has limits on team usage and storage; MLflow being fully self-hosted aligns with the project's "no recurring SaaS costs" constraint. Also: MLflow is more commonly used at Indian enterprise employers (TCS, Fractal, Tiger Analytics) and at AI-first startups.
2. **DVC for model versioning + plain S3 for artifacts** — Rejected. DVC is excellent for *data* versioning but not designed as an experiment tracker. Comparing 50 model runs in DVC is painful.
3. **Neptune.ai / Comet ML** — Rejected for the same reasons as W&B (paid tiers, less industry standard in India).
4. **SageMaker Experiments + Model Registry** — Rejected. Demonstrating MLflow knowledge is more transferable across cloud providers and employers. Also, MLflow is open-source and inspectable.

### Consequences

**Positive:**
- Industry-standard tool with broad recognition.
- Self-hosted; no recurring costs.
- Clean separation of experiment tracking (mutable, exploratory) and model registry (versioned, promoted).
- Native integration with scikit-learn, XGBoost, etc. via MLflow's autolog.

**Negative:**
- MLflow's UI is functional but uglier than W&B/Neptune. Acceptable for a portfolio project.
- Self-hosting means maintaining the tracking server (deployed as ECS Fargate task in production stack).
- Model registry semantics in MLflow ("Staging", "Production") have been deprecated in recent versions in favor of aliases — using the new alias-based API.

---

## ADR-004: Use Prefect for orchestration instead of Airflow

**Status:** Accepted

### Context

Three workflows need scheduling: feature materialization (daily), model retraining (weekly), and drift checks (hourly). Each has dependencies, retry logic, and observability requirements. A scheduler is needed — but the choice signals which orchestration ecosystem the engineer is fluent in.

### Decision

Use Prefect (specifically Prefect 2.x with the `@flow` and `@task` decorators). Self-host the Prefect server on a small EC2 instance for the project; Prefect Cloud's free tier is also sufficient.

### Alternatives Considered

1. **Apache Airflow** — Industry standard, used at most large Indian companies. Rejected for this project because:
   - Heavier setup (DAG files, separate scheduler/webserver/worker processes).
   - The DAG-as-code model is older; Prefect's "regular Python with decorators" is more ergonomic for ML workflows where logic depends on runtime values.
   - Airflow's strength (massive ecosystem of operators) isn't needed for this small project.
   - Trade-off: Airflow recognition is higher in services companies; Prefect signals modern stack awareness in product companies. The project README will note Airflow could swap in for production.
2. **Mage** — Beautiful UI, ML-native. Rejected only because Prefect has more job postings; Mage is great but newer.
3. **Dagster** — Strong data-asset model. Rejected because the learning curve is steeper than Prefect for the same value in this project.
4. **Pure cron + bash scripts** — Rejected. No retry logic, no observability, no failure handling. Acceptable for a hobby project, not for an MLOps portfolio piece.
5. **AWS Step Functions** — Considered. Native AWS, no infrastructure to manage. Rejected because (a) authoring workflows in Amazon States Language is more friction than Python decorators, (b) the project explicitly aims to be cloud-portable, and (c) demonstrating Prefect knowledge is more useful across employers.

### Consequences

**Positive:**
- Modern, ergonomic Python-native API.
- Built-in observability (UI shows flow runs, task status, logs).
- Easy to test flows locally with `flow.run()`.
- Free tier of Prefect Cloud handles the scheduling for the project.

**Negative:**
- Lower industry recognition than Airflow at large Indian enterprises.
- Self-hosted Prefect server needs basic monitoring (which the project provides via CloudWatch).

**Mitigations:**
- The project README explicitly notes "Prefect chosen for development velocity; Airflow is the recommended swap for >100 flows in production."
- Flow code is structured so migration to Airflow would be mechanical (each `@task` becomes an Airflow task).

---

## ADR-005: Class imbalance handled via class weights, not SMOTE

**Status:** Accepted

### Context

Fraud is rare — typically <0.5% of transactions in PaySim and similar datasets. Naïve training produces models that achieve 99.5% accuracy by predicting "not fraud" for everything. The standard responses are: (1) resample the training data (oversample fraud, undersample legit), (2) use synthetic oversampling like SMOTE, (3) adjust class weights in the loss function, or (4) calibrate decision thresholds post-hoc.

### Decision

Use class weights in the model's loss function (`class_weight='balanced'` for sklearn, `scale_pos_weight` for XGBoost). Calibrate the decision threshold post-hoc by selecting the operating point on the precision-recall curve that maximizes recall subject to precision ≥ 0.95.

### Alternatives Considered

1. **SMOTE (Synthetic Minority Oversampling Technique)** — Rejected. SMOTE generates synthetic fraud examples by interpolating between real fraud examples in feature space. For tabular data with strong feature interactions and time dependencies (as in fraud), SMOTE often produces unrealistic samples that hurt generalization. Recent literature (Elor & Averbuch-Elor, 2022) shows SMOTE rarely beats class weights on modern boosted-tree models.
2. **Random oversampling of the minority class** — Rejected. Causes overfitting on the few fraud examples; model memorizes them.
3. **Random undersampling of the majority class** — Rejected. Discards 99% of the data, including useful "near-fraud" patterns.
4. **Cost-sensitive learning with explicit miss/false-alarm costs** — Considered and partially adopted. The threshold calibration step encodes a cost asymmetry: a false negative (missed fraud) is costlier than a false positive (flagged legit transaction).

### Consequences

**Positive:**
- Simple to implement; no extra preprocessing pipeline.
- No risk of data leakage from synthetic samples.
- Threshold calibration explicitly documents the precision-recall trade-off, which is the actual business decision.

**Negative:**
- Class weights alone may not be enough at very extreme imbalance (>1:10000); fraud at 1:1000 is well within their effective range.
- Threshold calibration must be redone whenever the model is retrained (handled by the Prefect retraining flow).

**What this signals to interviewers:** Understanding that the "right metric" for imbalanced classification is not accuracy or even F1 with default 0.5 threshold, but a precision-recall trade-off chosen with business context.

---

## ADR-006: Use Evidently for drift detection, not a custom solution

**Status:** Accepted

### Context

Fraud patterns evolve: fraudsters adapt to detection systems, and legitimate user behavior shifts seasonally (festival spending spikes, new merchant categories appearing). The model will degrade if not monitored. Drift detection needs to cover: (1) input feature distribution drift, (2) prediction distribution drift, (3) model performance drift (when ground-truth labels arrive).

### Decision

Use Evidently AI's open-source library. Run drift reports hourly via a Prefect flow. Push key metrics to Prometheus via Evidently's metrics integration. Visualize in Grafana. Alert via SNS when drift exceeds configurable thresholds.

### Alternatives Considered

1. **DIY drift detection with scipy.stats** — Rejected. Re-implementing PSI, KS-test, and JS-divergence is wheel-reinvention. Evidently has been battle-tested.
2. **WhyLabs / whylogs** — Rejected. Excellent product, free tier exists, but Evidently's open-source library is sufficient and avoids any SaaS dependency.
3. **AWS SageMaker Model Monitor** — Considered. Native, deeply integrated. Rejected for the same portability reasons as ADR-003: Evidently knowledge transfers across employers; SageMaker Model Monitor is AWS-specific.
4. **Arize / Fiddler / Aporia** — Rejected. Commercial products, no free self-hosting option that fits the project budget.

### Consequences

**Positive:**
- Open-source, self-hosted, free.
- Pre-built reports for the standard drift tests (PSI, KS, Jensen-Shannon, Chi-squared).
- Generates HTML reports that can be archived in S3 for audit history.
- Direct Prometheus integration for live dashboards.

**Negative:**
- Evidently's report generation is somewhat slow for large datasets (>100K rows); the project mitigates by sampling 10K rows per hourly check.
- Setting drift thresholds is genuinely hard — initial values come from first-week production data; tuned over the project lifetime.

---

## ADR-007: Synthetic data generation in addition to PaySim dataset

**Status:** Accepted

### Context

PaySim is a static dataset of historical mobile money transactions. It's excellent for offline training but cannot be streamed in real-time. The project needs a "live" data source that mimics realistic transaction patterns and allows for controlled drift injection (to demonstrate that monitoring catches drift).

### Decision

Build a synthetic transaction generator (Python script) that:
- Samples from the same feature distributions as PaySim.
- Adds UPI-specific fields (sender VPA, receiver VPA, amount in INR, merchant category, device fingerprint hash).
- Injects fraud at a configurable rate (default 0.1%).
- Supports controlled drift modes: gradual amount inflation, sudden new merchant category, change in fraud pattern.

The generator pumps events into Kinesis at a configurable rate (default 10 events/sec).

### Alternatives Considered

1. **Replay PaySim from a CSV file** — Rejected. PaySim has ~6M rows but no temporal pattern beyond the synthetic timestamps. Re-streaming it doesn't allow controlled drift experiments.
2. **Use a real public streaming dataset** — Rejected. Public real-time financial datasets at this scale don't exist (for obvious privacy reasons).
3. **Use AWS Glue DataBrew or a synthetic data SaaS** — Rejected. Overkill for the project; a 200-line Python script suffices.

### Consequences

**Positive:**
- Full control over event rate, fraud rate, and drift scenarios.
- Enables interview-quality demos: "Watch the dashboard — I'm now injecting drift at t+30s, and the alert fires at t+45s."
- Realistic enough to demonstrate the pipeline; no PII concerns.

**Negative:**
- Synthetic data is, by definition, simpler than real production data. The model's reported metrics on synthetic data over-state real-world performance.
- The project README explicitly notes this caveat: "All metrics are on synthetic data; production deployment would require domain-adapted retraining."

---

## ADR-008: Terraform over CloudFormation/CDK

**Status:** Accepted

### Context

Infrastructure-as-Code is non-negotiable for this project — it's both a learning goal and a recruiting signal. The choice is between Terraform (cloud-agnostic, HCL), AWS CloudFormation (AWS-native, YAML/JSON), AWS CDK (AWS-native, real programming language), or Pulumi (cloud-agnostic, real programming language).

### Decision

Use Terraform (1.6+) with modules. State stored in S3 with DynamoDB locking. Workspaces for `dev` and `prod`.

### Alternatives Considered

1. **AWS CDK (Python or TypeScript)** — Genuinely strong choice. Rejected because Terraform is more commonly required in Indian MLOps job postings and signals broader cloud awareness.
2. **AWS CloudFormation** — Rejected. YAML is verbose; debugging stuck stacks is painful; community modules ecosystem is much smaller than Terraform's.
3. **Pulumi** — Strong on the programming-language angle but less common in production Indian environments. Rejected.

### Consequences

**Positive:**
- Multi-cloud portability if the project ever needs to run on GCP/Azure (unlikely, but signals adaptability).
- Massive community module ecosystem (terraform-aws-modules organization).
- Standard in DevOps job postings; transferable beyond MLOps.

**Negative:**
- HCL is its own language to learn; not as ergonomic as Python/TypeScript.
- State management has gotchas (the S3 + DynamoDB pattern mitigates).

---

## ADR-009: GitHub Actions over Jenkins/GitLab CI/CircleCI

**Status:** Accepted

### Context

CI/CD must run: linting, type checking, unit tests, integration tests, Docker builds, Terraform plans, and deployments. The choice of CI platform is partly stylistic and partly about which integrations exist.

### Decision

Use GitHub Actions. Workflows live in `.github/workflows/`. Reusable composite actions for repeated steps.

### Alternatives Considered

1. **Jenkins** — Rejected. Self-hosting Jenkins for a side project is operational overhead with no benefit. Jenkins remains relevant in large enterprises but adds nothing here.
2. **GitLab CI** — Excellent product, comparable to GitHub Actions. Rejected only because the project repo is on GitHub; switching the host would add friction.
3. **CircleCI** — Rejected. Free tier exists but smaller than GitHub Actions; the integration with GitHub PRs is one extra step.
4. **AWS CodePipeline + CodeBuild** — Rejected. Native AWS but YAML-only configuration is verbose; debugging is via CloudWatch which is friction-heavy.

### Consequences

**Positive:**
- Native GitHub integration (PR checks, deployment status).
- Free for public repos with generous minutes.
- Massive marketplace of pre-built actions.

**Negative:**
- Vendor lock-in to GitHub (acceptable; the project lives on GitHub).
- Some advanced patterns (matrix builds, reusable workflows) have a learning curve.

---

## ADR-010: Postgres on RDS for decision audit log, not DynamoDB

**Status:** Accepted

### Context

Every fraud-detection decision (transaction ID, model version, prediction, score, latency, ground truth when known) must be persisted for: (a) computing model performance metrics, (b) regulatory audit, (c) generating training data for future retraining. The data is structured, queryable, append-heavy, and needs efficient time-range queries.

### Decision

Use AWS RDS for PostgreSQL (db.t4g.micro under free tier for the first 12 months). Schema includes a `decisions` table with appropriate indexes on `timestamp`, `model_version`, and `transaction_id`.

### Alternatives Considered

1. **DynamoDB** — Rejected. DynamoDB is excellent for known-key-pattern access but the project needs ad-hoc analytical queries ("how did model v2.3 perform on fraud cases between Tuesday 2pm and 3pm?"). DynamoDB's secondary indexes are pricey, and analytical queries become exports to S3.
2. **Direct write to S3 + Athena for queries** — Considered. Cheap and durable. Rejected because individual decision writes to S3 are expensive (per-PUT cost) at the project's event rate, and Athena queries have latency unsuited for real-time dashboards.
3. **Timestream (AWS time-series DB)** — Considered. Purpose-built for time-series. Rejected because the data model (transactions with rich attributes) doesn't fit Timestream's measure-centric model cleanly.
4. **OpenSearch / Elasticsearch** — Rejected. Excellent for log-style queries but minimum cluster cost is outside budget.

### Consequences

**Positive:**
- SQL is universally known; analytical queries are easy.
- Postgres handles ~10K writes/sec with appropriate tuning — well above the project's load.
- Free tier covers the first year.

**Negative:**
- Single point of write contention; doesn't scale horizontally.
- For a real production system at >10K events/sec, the architecture would shift to a streaming write to S3 (via Firehose) with Athena for queries, and Postgres reserved for the *aggregated* metrics layer.

**Mitigations:**
- Lambda writes are batched (10 decisions per write) to reduce DB contention.
- A daily Prefect flow archives decisions older than 30 days to S3 in Parquet format.

---

## ADR-012: Local Docker stack cannot load models registered pre-containerization

**Status:** Accepted as known limitation. Resolved in week 4 via S3 migration.

**Context:** MLflow 3.x stores model artifacts using internal "logged model"
URIs (e.g., `models:/m-c7ab...`) rather than concrete filesystem paths. When
models are registered through a local SQLite client (notebooks), the
artifact metadata is keyed to local filesystem locations. When the same
SQLite database is later served by an MLflow server inside Docker, the
server cannot resolve those logged-model URIs to artifacts the API
container can read.

**Decision:** Acknowledge the limitation. Do not invest further in working
around it locally. The structural fix is to move artifacts to S3 in week 4,
which produces universal URIs (`s3://bucket/key`) that resolve identically
from any environment.

**Consequences:** The Week 3 Docker stack proves the deployment architecture
(FastAPI + MLflow server, Docker network, healthchecks, non-root user) but
cannot serve predictions until artifacts are migrated. This is acceptable
because the integration test against local Python (notebook 03) already
proves the model loading works; Docker only adds the deployment shell.

---

## Decision matrix summary

The following table captures the high-level positioning of each major decision:

| Decision | Optimized for | Trade-off accepted |
|---|---|---|
| Streaming (Kinesis+Lambda) | Latency, decoupling | Cold starts, per-shard cost |
| Feast feature store | Train-serve consistency | Operational complexity |
| MLflow | Industry recognition, self-hosted | Less polished UI than W&B |
| Prefect | Modern Python ergonomics | Less recognition than Airflow |
| Class weights + threshold calibration | Realistic fraud handling | Doesn't beat SMOTE on every dataset |
| Evidently | Open-source, self-hosted | Slower than commercial alternatives |
| Synthetic data generator | Controllable drift demos | Optimistic metrics |
| Terraform | Multi-cloud portability | HCL learning curve |
| GitHub Actions | Native GitHub integration | Vendor lock-in to GitHub |
| Postgres on RDS | SQL queryability | Doesn't scale horizontally |
| Fargate (ECS) for serving | Production-grade cloud inference, no laptop dependency | ALB cost accrues even when tasks are stopped |
| Lambda (Kinesis streaming path) | Event-driven inference, scales to zero | 24s cold start, Kinesis not Free Tier |
| Feast + Redis (feature store) | Train-serve consistency, 42ms warm latency | Redis IP hardcoded in image, EC2 Redis is a single point of failure |
| Prefect (local server) | ML pipeline orchestration, retry logic, audit trail | Scheduled flows require Mac to be on; Prefect Cloud free tier doesn't support hybrid pools |
| Evidently + Prometheus + Grafana + SNS | Full monitoring stack, drift detection, alerting | Local Docker for monitoring services; not always-on |

---

## How to use these ADRs in interviews

When an interviewer asks about a design choice, the structure of an ADR is the structure of a strong answer:

1. **Start with the context** — what problem were you actually solving?
2. **State the decision** — what you did.
3. **Show you considered alternatives** — name 2-3, briefly say why each was rejected.
4. **Acknowledge consequences** — both positive and negative. *Especially* negative — this signals engineering maturity.

Example, in interview form:

> *"Why did you use Kinesis over Kafka?"*
>
> "Fraud detection has a hard latency requirement of around 100-300ms, so I needed a streaming layer. I considered Kafka and AWS MSK, but Kafka has operational overhead — broker management, monitoring — that wasn't justified for an 8-week project, and MSK's minimum cost is around $150/month, outside my budget. Kinesis gives me managed sharding, 24-hour retention for replay, and Lambda integration. The trade-off I accepted is that Kinesis caps at 1MB/sec/shard, which would need re-architecting at >10K events/sec — but for the project's load that's fine. If I were doing this at production scale at a real fintech, I'd revisit Kafka."

That answer is far stronger than "I used Kinesis because the tutorial used it." The ADRs above are your script for two dozen such answers.

---

## Maintaining this document

- New ADRs added at the bottom, numbered sequentially.
- Existing ADRs are immutable. If a decision changes, write a new ADR that supersedes the old one and update the old one's status to `Superseded by ADR-XXX`.
- This document lives at `docs/adr/decisions.md` in the repo and is linked from the main README.

---

*References used:* Michael Nygard's original ADR proposal (2011), Joel Parker Henderson's adr-templates GitHub repo, ThoughtWorks Tech Radar guidance on architectural decisions.





Fair. The earlier ADRs are long because they record decisions with many alternatives; ADR-013 had basically one real choice (move state to cloud, yes/no), so it doesn't need the same surface area. Here's a tighter version:

---

## ADR-013: Migrate MLflow backend to RDS Postgres and artifacts to S3

**Status:** Accepted

### Context

ADR-003 deferred "RDS Postgres for cloud deployments" to a later week. ADR-012 documented that the Week 3 Docker stack couldn't serve predictions because MLflow's logged-model URIs resolved to filesystem paths inside the MLflow container that the FastAPI container couldn't read. Sharing a Docker volume would have masked the symptom; the underlying problem is that MLflow state was tied to a single machine.

### Decision

Move MLflow's backend store from local SQLite to **RDS PostgreSQL 16** (`db.t4g.micro`, Free Tier, `ap-south-1`) and its artifact store from local filesystem to **S3** (`s3://fraud-mlops-kidiloski/mlflow-artifacts`).

- MLflow server runs on the developer's Mac (`127.0.0.1:5000`) pointed at both backends. Server-on-Fargate deferred to a later week.
- Config in `.env` (gitignored), loaded via `python-dotenv`. `setup_mlflow()` calls `load_dotenv()` so notebooks, FastAPI, and future Lambda all configure identically without depending on shell environment.
- Re-trained the model through the new server rather than migrating v3 artifacts. New registry's `fraud-detector` v1 supersedes old v3. `inference.py` needs no changes — it resolves by alias (`@production`), not version.

### Alternatives Considered

1. **Docker volume sharing** — Rejected. Solves ADR-012's symptom but not the underlying state-locality problem; Lambda would hit the same wall.
2. **MinIO + Postgres in Docker Compose** — Rejected. Local emulation of AWS is a weaker signal than real AWS integration; week 4's explicit goal was to stop simulating.
3. **Aurora Serverless v2** — Rejected on cost. ~$65/month minimum vs. Free Tier on `db.t4g.micro`.
4. **Migrate v3 artifacts via `download_artifacts` + re-log** — Rejected. Migration code has zero portfolio value; re-training takes 1-5 min and gives a cleaner new-MLflow story.

### Consequences

**Positive:**
- MLflow state is now location-independent. Any process with AWS credentials resolves `models:/fraud-detector@production` to the same S3 object. Unblocks Week 5's Lambda.
- ADR-012's Docker blocker is structurally fixed — same S3 URI resolves identically from any container.
- The `setup_mlflow()` abstraction from Week 2 paid off: zero code changes, only env vars switched.
- Free Tier covers RDS (750 hrs/month db.t4g.micro, 20 GB storage) and S3 artifact storage (~715 KB per model) through the project.

**Negative:**
- FastAPI startup now requires network. Cold-start 2-4s vs. <500ms locally.
- Home IP rotation breaks RDS connectivity until the security group is updated. Manageable solo; needs bastion/VPN for a team.
- Single-AZ `db.t4g.micro`, no Multi-AZ — zone failure takes MLflow offline. Acceptable for portfolio.
- Cost discipline: RDS bills storage 24/7, stopped instances auto-restart after 7 days. "Stop RDS at end of session" is now a standing rule, backstopped by a $20 billing alarm.

**What this signals to interviewers:** "Cloud-native" means application state lives independently of the machine that produced it, not "I deployed to AWS." Recognizing that ADR-012's symptom and this fix are causally linked — and that alias-based model resolution was specifically designed so this migration required zero code changes.

---

---

## ADR-014: Deploy MLflow and FastAPI as Fargate services behind an ALB

**Status:** Accepted

### Context

ADR-013 moved MLflow state to RDS + S3, making it location-independent. Week 5's goal was to prove inference compute is equally location-independent — running entirely in AWS without the developer's laptop. The Week 3 Docker Compose stack proved the container architecture; this week promoted it to managed cloud compute.

### Decision

Deploy two ECS Fargate services on cluster `fraud-mlops` (`ap-south-1`):

- **MLflow service** — internal only, reachable at `mlflow:5000` via ECS Service Connect. `1024 CPU / 2048 MB` ARM64. Backed by the existing RDS + S3 from Week 4.
- **API service** — public via Application Load Balancer (`fraud-mlops-alb`) on HTTP port 80. `512 CPU / 1024 MB` ARM64. Loads `fraud-detector@production` from S3 at startup.

Both images in ECR tagged `:v1`. RDS password stored in Secrets Manager as a full `postgresql://...?sslmode=require` URI, injected via the task definition `secrets` block. Task roles are least-privilege: MLflow task role has S3 read/write; API task role has S3 read only (needed because the API fetches artifacts directly from S3, not proxied through MLflow).

### Alternatives Considered

1. **HTTPS on ALB** — Rejected for this iteration. Requires a custom domain (~$10/year) with no learning benefit given the project uses synthetic data. ALB default DNS name used. Production deployment would add an ACM cert + HTTPS listener with no application code changes.
2. **MLflow on ALB (public)** — Rejected. MLflow has no authentication; exposing it publicly is a real security hole. Service Connect keeps it private to the ECS network; SSM port-forward used when UI access is needed.
3. **Proxied artifacts (MLflow Mode A, `--artifacts-destination`)** — Rejected for now. Would route all artifact bytes through MLflow, removing the need for S3 permissions on the API task role. Kept Mode B (`--default-artifact-root`) because it matches what was already tested locally and avoids a flag change + image rebuild mid-week.
4. **Default VPC for ECS** — Rejected. RDS was provisioned in a non-default VPC (`vpc-07417979b25261bdc`). Placing ECS in the default VPC would require VPC peering; moving everything into the RDS VPC was simpler.

### Consequences

**Positive:**
- First fully cloud-hosted inference: a public `curl` to the ALB returns a real prediction with no local dependencies.
- Container architecture from Week 3 ported directly — same Dockerfiles, same `MLFLOW_TRACKING_URI=http://mlflow:5000` pattern that worked in Compose now works in Fargate via Service Connect.
- `desired-count=0` between sessions stops task billing while preserving all service configuration.

**Negative:**
- ALB costs ~$0.008/hour (~$6/month) even when tasks are stopped. Must be deleted when the project ends or costs will accrue indefinitely.
- `512 MB` was insufficient for MLflow 3.x — required bumping to `2048 MB`. Task definition went through three revisions (`:1` → `:2` → `:3`) before stabilising.
- No HTTPS — traffic is unencrypted in transit. Acceptable for synthetic data; not acceptable for real PII.
- RDS security group requires manual IP updates when the developer's home IP rotates. Fargate tasks use a fixed security group reference, so this only affects direct Mac → RDS access, not the Fargate → RDS path.




## ADR-015: Lambda container image deployment for streaming fraud inference

**Status:** Accepted

### Context

Week 6 introduced the streaming inference path: Kinesis Data Stream → Lambda → score transaction → write decision to RDS. This ADR records the non-obvious decisions made during Lambda deployment.

### Decision

Deploy the fraud-scorer Lambda as a **container image** (not a zip package) from ECR, running on **arm64** (Graviton2), placed **inside the RDS VPC** with an **S3 Gateway VPC Endpoint** for artifact access.

### Alternatives Considered

1. **Zip package instead of container image** — Rejected. The inference dependencies (xgboost, scikit-learn, mlflow, psycopg2) exceed Lambda's 250MB unzipped zip limit. Container images support up to 10GB.
2. **x86_64 instead of arm64** — Rejected. The Mac build host is arm64; building x86_64 requires QEMU emulation (5-10× slower builds). Graviton2 Lambda is also ~20% cheaper per invocation. All dependencies have arm64 wheels.
3. **OCI image index manifest (default Docker Buildx output)** — Rejected by Lambda. Lambda container functions do not support multi-platform manifest lists. Rebuilt with `--provenance=false` to produce a single-platform manifest.
4. **Lambda outside the VPC** — Rejected. Lambda needs to reach RDS on port 5432. Placing Lambda in the RDS VPC (`vpc-07417979b25261bdc`) with a dedicated security group (`fraud-mlops-lambda-sg`) allows direct private connectivity without exposing RDS to the internet.
5. **NAT Gateway for S3 access** — Rejected on cost (~$32/month). An S3 Gateway VPC Endpoint achieves the same result for free by routing S3 traffic through AWS's internal network rather than the public internet.
6. **Secrets Manager for RDS password** — Deferred. Lambda doesn't natively inject Secrets Manager values into environment variables the way ECS does. Options are: (a) plain-text env var (current approach), (b) boto3 call at runtime (+~200ms cold start), (c) Lambda Extensions (complex). Plain-text env var accepted for this project; Lambda function configuration is not publicly accessible.

### Consequences

**Positive:**
- Full streaming path works end-to-end: producer → Kinesis → Lambda → RDS decisions table.
- Model loads from S3 via VPC endpoint — no internet dependency, no NAT cost.
- Warm invocations score a transaction in ~200ms, well within the 300ms fraud-decision SLA.
- `batchItemFailures` partial batch response prevents a single bad record from blocking the entire Kinesis shard.

**Negative:**
- Cold start is ~24 seconds (model load from S3 dominates). This is unacceptable for a latency-sensitive path; mitigated here by the fact that the streaming path is fire-and-forget, not synchronous. For a production deployment, provisioned concurrency would reduce cold start to under 1 second.
- RDS password is in Lambda environment variables in plain text. Acceptable for a portfolio project; production deployment would use Secrets Manager via a boto3 call in the handler.
- Kinesis Data Streams is not Free Tier eligible (~$0.015/shard-hour). Stream is deleted between sessions; total project cost estimated under $5.

**What this signals to interviewers:** Understanding the Lambda-in-VPC trade-off (RDS access costs internet access; S3 VPC endpoint resolves this without NAT). Knowing that OCI manifest lists are not supported by Lambda and how to fix it. Measuring and acknowledging cold start rather than hiding it, with a concrete mitigation path (provisioned concurrency).


## ADR-016: Feast feature store with Redis online store and S3 registry

**Status:** Accepted

### Context

The fraud-detector model requires five sender-history features (`sender_txn_count_1h`, `sender_txn_count_24h`, `sender_amount_sum_24h`, `sender_amount_mean_historical`, `sender_time_since_last_txn`) that must be computed from historical transaction data. In Weeks 1-6, these were pre-computed during feature engineering and baked into the training dataset — but the Lambda handler received them as part of the Kinesis payload, meaning the producer had to compute them before sending. This is train-serve skew: training uses one computation path, serving uses another.

Week 7 resolves this by introducing Feast as the feature store, with Redis as the online store for sub-50ms lookups during Lambda inference.

### Decision

Deploy Feast 0.63 with:

- **Offline store:** file-based (S3 parquet at `s3://fraud-mlops-kidiloski/feast/paysim_repeat_senders.parquet`)
- **Online store:** Redis on EC2 t4g.micro (`172.30.0.198:6379`, private IP, same VPC as Lambda)
- **Registry:** S3 file (`s3://fraud-mlops-kidiloski/feast/registry.pb`)
- **Entity:** `sender` (PaySim's `nameOrig`, renamed for clarity)
- **Feature view:** `sender_transaction_features` — the five sender-history features, TTL 7 days

Materialize only the **9,298 repeat senders** (senders with more than one transaction) rather than all 6.35M unique senders. First-time senders have no history to look up; their features default to zeros, which is what the model was trained on for new senders. This reduces Redis memory requirements from ~6GB to ~50MB.

Lambda looks up features by `sender_id` from the Kinesis record. If the sender is not in Redis (unknown/first-time sender), the handler falls back to default zero values and logs `feast_hit=False`. This graceful degradation is intentional — the model handles zero-history senders correctly because PaySim's training data includes many first-time senders.

### Alternatives Considered

1. **SQL registry (Postgres)** — Tried first. Feast connects to Postgres on every `get_online_features` call to validate the registry, adding 1-2 seconds of latency per invocation and causing timeouts when the Lambda VPC couldn't reach RDS reliably. Replaced by S3 file registry which is read once at cold start and cached in memory.

2. **Public IP for Redis** — Tried first. Public IPs change on every EC2 stop/start, requiring an image rebuild each session. Private IP (`172.30.0.198`) is stable within the VPC across stop/start cycles.

3. **Materialize all 6.35M senders** — Attempted. OOM-killed Redis twice, first on t4g.nano (512MB), then on t4g.micro (1GB) when Feast tried to pipeline-write all keys in a single batch. Filtered to repeat senders solves the problem at no cost to model quality.

4. **ElastiCache instead of Redis on EC2** — Rejected on cost. ElastiCache minimum is ~$15/month. EC2 t4g.micro with Redis is Free Tier eligible for 12 months.

5. **Skip Feast, keep pre-computed features in Kinesis payload** — Rejected. Defeats the purpose: train-serve skew remains. Producer would need to replicate feature engineering logic, creating two code paths for the same computation.

### Consequences

**Positive:**
- Train-serve consistency enforced: features defined once in Feast, served identically to training and inference.
- Warm Lambda invocation latency dropped to ~42ms (previously ~200ms with pre-computed features in payload).
- Producer now sends only raw transaction fields — simpler, smaller Kinesis records.
- Model v2 (`enriched_v1`, threshold 0.4234) trained with proper Feast features achieves 84.5% recall vs 80.4% for v1 — a meaningful improvement at the same precision.
- Unknown sender fallback is explicit and logged (`feast_hit=False`), making observability straightforward.

**Negative:**
- Redis IP is hardcoded in `feature_store.yaml` baked into the Lambda image. If the EC2 instance is terminated and recreated, the private IP may change, requiring a rebuild. Mitigated by using private IP (stable across stop/start) rather than public IP.
- Feast materialization must be re-run when new transaction data is available (handled by Prefect in Week 8).
- EC2 Redis instance must be running for Lambda to serve Feast features. If Redis is down, Lambda falls back to defaults — predictions still work but without historical features.
- 9,298 keys covers only repeat senders. In a real production system, the online store would be populated incrementally as new transactions arrive, not batch-materialized from a static dataset.

**What this signals to interviewers:** Understanding train-serve skew and why it matters. Knowing that a feature store's online store must be fast (Redis, not Postgres) and that the registry lookup pattern affects inference latency. Making a deliberate trade-off between completeness (all 6.35M senders) and practicality (9,298 repeat senders with meaningful history), and being able to justify it.


## ADR-017: Prefect for ML pipeline orchestration with local server

**Status:** Accepted

### Context

Three ML pipelines need scheduled, monitored, and retryable execution:
feature materialization (daily), model retraining (weekly), and drift
detection (hourly, Week 9). Running these as cron jobs or notebooks would
work functionally but provides no observability, no retry logic, and no
audit trail of past runs.

### Decision

Use **Prefect** for orchestration with a **local server** (`prefect server
start` at `http://localhost:4200`) and a **process work pool**
(`fraud-mlops-pool`). Flows are deployed via `flows/deploy.py` using
`prefect.serve()` which registers deployments and serves them from the
local process.

Two flows deployed:
- `feast-materialization` — daily at 2am, refreshes Redis from S3 parquet
- `model-retraining` — weekly Sundays at 2am, retrains XGBoost and promotes
  if recall improves >2% at precision ≥0.95

### Alternatives Considered

1. **Prefect Cloud free tier** — Rejected. Does not support hybrid or push
   work pools. Would require managed execution on Prefect's infrastructure,
   losing control over the execution environment.
2. **Apache Airflow** — Rejected for this project. Heavier setup (separate
   scheduler, webserver, worker processes), DAG-file model less ergonomic
   for ML workflows. Airflow has higher industry recognition at large
   enterprises; noted in README as the recommended swap for >100 flows in
   production.
3. **Cron + bash scripts** — Rejected. No retry logic, no observability, no
   failure alerting, no audit trail of past runs.
4. **AWS Step Functions** — Rejected. Amazon States Language adds friction
   vs Python decorators; cloud-provider lock-in; less transferable knowledge.

### Consequences

**Positive:**
- `@flow` and `@task` decorators add minimal boilerplate to existing Python.
- Every run logged to UI with task-level status, logs, and duration.
- Retry logic declarative: `@task(retries=2, retry_delay_seconds=30)`.
- Retraining gate correctly skips promotion when new model doesn't improve
  recall by ≥2% — demonstrated in first run (v2 retained over new candidate).
- Absolute paths passed via deployment parameters solve the working-directory
  problem when Prefect worker runs flows from a different cwd than the shell.

**Negative:**
- Local server only runs when developer's Mac is on. Scheduled flows won't
  execute if the machine is asleep. For production, the Prefect worker would
  move to an always-on EC2 instance.
- `REDIS_CONNECTION_STRING` env var must be updated when Redis EC2 public IP
  changes (on every stop/start). Mitigated by using private IP inside Lambda
  and public IP only for local Mac-originated flows.
- Prefect's `serve()` pattern blocks the terminal; deployment process must
  stay running for scheduled flows to execute.


## ADR-018: Full monitoring stack — Evidently, Prometheus, Grafana, SNS

**Status:** Accepted

### Context

A deployed fraud model needs monitoring on two dimensions: (1) are the
inputs drifting from the training distribution, and (2) are we alerted
quickly enough to act? Without monitoring, a model can silently degrade
for days before anyone notices — the classic production ML failure mode.

### Decision

Deploy a four-component monitoring stack:

- **Evidently 0.7.x** — computes drift metrics comparing recent decisions
  against the PaySim training baseline. Runs as a Prefect flow hourly.
  Generates HTML reports saved to `s3://fraud-mlops-kidiloski/drift-reports/`.
- **Prometheus + Pushgateway** — Evidently metrics are pushed to
  Pushgateway after each flow run. Prometheus scrapes Pushgateway every
  15 seconds. Both run as local Docker containers.
- **Grafana** — three-panel dashboard: drift share time series, drift
  status (DRIFT DETECTED / OK with color mapping), and drifted columns
  count. Runs as a local Docker container at `localhost:3000`.
- **SNS** — email alert fires when `drift_share > 0.3` (more than 30%
  of monitored columns drifting). Topic:
  `fraud-mlops-drift-alerts` in `ap-south-1`.

Drift is computed on `amount` and `type` columns — the features most
likely to shift with real fraud pattern changes. Reference dataset is
5,000 randomly sampled rows from the PaySim training parquet.

### Alternatives Considered

1. **WhyLabs / Arize / Fiddler** — Commercial managed monitoring
   platforms. Rejected on cost and portability. Evidently is open-source,
   self-hosted, and the most widely-known ML monitoring library; knowledge
   transfers across employers.
2. **CloudWatch metrics instead of Prometheus** — Native AWS, no extra
   infrastructure. Rejected because Prometheus + Grafana is the industry
   standard for ML monitoring dashboards, and demonstrating it is more
   transferable than CloudWatch-specific knowledge.
3. **Grafana Cloud instead of local Grafana** — Free tier exists.
   Rejected to keep all monitoring co-located and avoid another external
   account dependency.
4. **Alerting via Grafana instead of SNS** — Grafana has built-in
   alerting. Rejected because SNS is already in the AWS stack and adding
   Grafana alerting adds complexity without benefit. SNS also integrates
   with Lambda, PagerDuty, and other downstream systems more naturally.

### Consequences

**Positive:**
- End-to-end monitoring story: drift detected → metrics in Prometheus →
  visible in Grafana → SNS alert sent. Demonstrated with injected drift
  (100x normal transaction amounts).
- HTML reports persisted in S3 provide an audit trail of every drift
  check run.
- Prefect handles scheduling, retries, and observability of the drift
  flow itself.
- `fraud_detector_drift_detected` Gauge allows Grafana alerting rules
  to be added later without changing the flow.

**Negative:**
- Prometheus, Pushgateway, and Grafana run as local Docker containers —
  they're not in AWS and not always-on. For production, these would run
  on EC2 or ECS. Deliberately deferred: the monitoring *code* is
  production-ready; the *deployment* of the monitoring infrastructure
  is out of scope for this project.
- Drift is computed on `amount` and `type` only — not on all 14 model
  features. Full feature monitoring would require all features to be
  logged in the decisions table, which they currently aren't.
- Reference dataset is static (PaySim training data). In production,
  the reference would be updated periodically to account for legitimate
  distribution shifts.

**What this signals to interviewers:** Understanding that monitoring is
not optional in production ML — it's the mechanism that closes the
feedback loop. Choosing Evidently over custom drift code because
reinventing statistical tests (PSI, KS, chi-squared) is wheel-reinvention.
Knowing the difference between data drift detection (Evidently) and
model performance monitoring (requires ground-truth labels from the
decisions table once fraud is confirmed) — and being honest that the
project currently only does the former.





## ADR-019: Terraform IaC for full AWS stack

**Status:** Accepted

### Context

By Week 10, all AWS infrastructure had been created manually over 9 weeks
of iterative development. While the stack worked, it existed only as a
collection of manually-clicked console actions and CLI commands with no
reproducibility guarantee. A new environment couldn't be provisioned
without repeating every manual step. Terraform was introduced to codify
the entire stack as version-controlled, reproducible infrastructure.

### Decision

Use **Terraform** (v1.15.5) with:
- **S3 backend** (`fraud-mlops-kidiloski/terraform/state.tfstate`) for
  remote state
- **DynamoDB locking** (`fraud-mlops-terraform-locks`) for concurrent
  apply protection
- **14 HCL files** covering all resource types: VPC/security groups, RDS,
  ECR, ECS, ALB, Lambda, Kinesis, IAM, Secrets Manager, SNS, CloudWatch,
  Redis EC2, S3 (data source)
- **`prevent_destroy`** lifecycle on ECR repos (contain Docker images) and
  initially on RDS
- **S3 and ECR as protected resources** — never destroyed, only managed

The full destroy/apply cycle was executed to prove reproducibility:
`terraform destroy` (staged) → `terraform apply` → full stack rebuilt
from scratch in ~15 minutes.

### Alternatives Considered

1. **AWS CDK** — Rejected. Python-native but generates CloudFormation
   underneath; less transferable knowledge than HCL since most
   infrastructure jobs ask for Terraform specifically.
2. **Pulumi** — Rejected. Smaller ecosystem, fewer job postings referencing
   it vs Terraform.
3. **CloudFormation** — Rejected. AWS-specific, verbose YAML/JSON, less
   ergonomic than HCL for complex stacks.
4. **Keep manual setup** — Rejected. Not reproducible, not version-controlled,
   not demonstrable in interviews.

### Consequences

**Positive:**
- Entire AWS stack reproducible with `terraform init && terraform apply`.
- Infrastructure changes are code-reviewed via git, not tribal knowledge.
- `terraform plan` shows exactly what will change before any apply —
  critical safety check after the RDS deletion incident (see below).
- Outputs (`alb_dns_name`, `redis_private_ip`, etc.) eliminate manual
  endpoint lookups after each apply.
- Destroy/apply cycle verified: stack rebuilds cleanly in ~15 minutes.

**Negative / Incidents:**
- **RDS deletion incident:** During the first import attempt, the imported
  RDS config didn't match the HCL exactly (engine version, description).
  Terraform decided to replace (destroy + recreate) the RDS instance.
  The plan showed `Destroying...` but was approved without careful review.
  All MLflow schema and decisions data was lost. Recovery took ~30 minutes.
  **Lesson:** Always grep the plan for `destroy` and `must be replaced`
  before typing `yes` on imported resources.
- **Lambda zombie ENIs:** After Lambda function deletion, AWS held two VPC
  ENIs (`ela-attach` type) in `in-use` state for several hours. The Lambda
  security group couldn't be deleted until the ENIs were released. Mitigated
  by removing the SG from Terraform state and letting AWS clean it up.
  This is a known AWS issue with no user-side fix.
- **Secrets Manager recovery window:** Deleted secrets have a 7-day
  recovery window by default. Required `--force-delete-without-recovery`
  flag to immediately recreate a secret with the same name during the
  destroy/apply cycle.
- **Import complexity:** 12+ resources required manual import before the
  first clean apply, each with potential config mismatches. `ignore_changes`
  lifecycle blocks were used as workarounds, then removed before the full
  destroy/apply cycle.

**What this signals to interviewers:** Understanding that IaC adoption
on existing infrastructure is harder than greenfield Terraform. Knowing
the `prevent_destroy` / `ignore_changes` lifecycle patterns. Being able
to explain the RDS incident — what went wrong, why, and what the correct
workflow is (`terraform plan | grep destroy` before every apply on
imported resources). The zombie ENI issue demonstrates real-world AWS
edge cases that don't appear in tutorials.
