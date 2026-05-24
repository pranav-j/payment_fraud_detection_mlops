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
