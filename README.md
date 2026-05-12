# fraud-mlops

Real-time UPI fraud detection MLOps pipeline. A portfolio project that takes a fraud detection model from notebook to production: streaming inference, feature store, drift monitoring, automated retraining, and infrastructure-as-code.

**Status:** Week 1 — local notebook baseline.

## What this project demonstrates

This is a deliberate stretch beyond the standard MLOps Zoomcamp capstone, with three production-realistic challenges layered on top: real streaming (Kinesis + Lambda, not batch), feature store with online/offline parity (Feast), and drift monitoring with alerts that actually trigger retraining (Evidently + Prometheus + Grafana + SNS).

The architecture, hosting locations, and cost trade-offs are documented in [`docs/adr/decisions.md`](docs/adr/decisions.md).

## Stack (planned, by week)

| Concern | Tool |
|---|---|
| Experiment tracking | MLflow |
| Orchestration | Prefect |
| Streaming ingestion | AWS Kinesis |
| Inference compute | AWS Lambda (container image) |
| Synchronous fallback | FastAPI on ECS Fargate |
| Feature store | Feast (Postgres offline + Redis online) |
| Decision audit | RDS Postgres |
| Monitoring | Evidently + Prometheus + Grafana |
| Alerting | SNS |
| IaC | Terraform |
| CI/CD | GitHub Actions |
| Data | PaySim (Kaggle) + synthetic UPI generator |

Week 1 uses none of the above. Week 1 is just `pandas`, `scikit-learn`, `xgboost`, and discipline.

## Quick start

Prerequisite: [`uv`](https://docs.astral.sh/uv/getting-started/installation/). On macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# 1. Install dependencies (runtime + dev)
make install-dev

# 2. Configure environment
cp .env.example .env
# Edit .env and add your Kaggle credentials.

# 3. Download the dataset
make data

# 4. Open the notebooks
make notebook
```

Then run, in order:
1. `notebooks/01_eda.ipynb` — understand the data
2. `notebooks/02_baseline_model.ipynb` — train baseline XGBoost
3. `notebooks/03_feature_engineering.ipynb` — engineer features and beat the baseline (next)

Each notebook is self-contained and runs top-to-bottom on a fresh kernel.

## Repository structure

```
fraud-mlops/
├── data/              # gitignored; raw/interim/processed datasets
├── docs/adr/          # architecture decision records
├── models/            # serialized model artifacts (gitignored)
├── notebooks/         # exploration and prototyping
├── reports/figures/   # plots committed for README/blog
├── scripts/           # data download, deploy helpers
├── src/fraud_mlops/   # importable Python package (logic lives here)
├── tests/             # pytest suite
├── Makefile           # task runner
├── pyproject.toml     # uv-managed project config
└── README.md          # you are here
```

The principle: **logic lives in `src/`, not in notebooks.** Notebooks are thin wrappers that call into modules. This is what makes the eventual transition from notebook to Prefect flow trivial.

## What fraud looks like in this data

(Fill in after running notebook 01)

1.
2.
3.
4.
5.

## Development workflow

```bash
make help          # list available targets
make install-dev   # install with dev dependencies + pre-commit hooks
make lint          # ruff + mypy
make format        # auto-format
make test          # run pytest
make check         # lint + test
make clean         # remove caches
```

Pre-commit hooks run on every commit: `nbstripout` (strip notebook outputs), `ruff` (lint + format), `detect-secrets` (catch accidentally-committed credentials), and standard hygiene checks.

## Architecture decisions

See [`docs/adr/decisions.md`](docs/adr/decisions.md) for the full set of architecture decision records explaining every major design choice (why Kinesis over Kafka, why Feast over a hand-rolled cache, why Prefect over Airflow, etc.) along with what was rejected and why.

## License

MIT
