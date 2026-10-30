# Olist Customer Churn — MLOps

[![CI](https://github.com/Belal237/olist-customer-churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Belal237/olist-customer-churn-mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)

End-to-end ML system predicting customer churn on the
[Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
dataset: SQL feature pipeline, XGBoost model tracked in MLflow, FastAPI service
in Docker, CI on GitHub Actions.

This repository is the first phase of a system that grows over four phases,
from a production-grade ML service (P1) to a customer-support agent with RAG,
tool calling and guardrails (P3–P4). The model is deliberately simple: the
focus is on the reliability of everything around it.

## Architecture

```
data/raw/*.csv (Kaggle)
    │  src/data/load.py                       COPY into PostgreSQL + join-key indexes
    ▼
PostgreSQL (olist_db)
    │  src/features/rfm.sql                   RFM aggregates + churn label (time-split)
    │  src/features/sql_features.py           data-quality checks
    ▼
data/processed/rfm_features.parquet
    │  src/features/build.py                  strict Pydantic contracts + engineered features
    ▼
data/processed/features.parquet ───────────────────────────────┐
    │  src/models/train.py                    CV, metrics, MLflow  │
    ▼                                                           │
MLflow server (tracking + registry + artifact proxy)            │
    │  models:/churn-prediction/latest → pinned version         │
    ▼                                                           ▼
src/api (FastAPI, Docker) ── validated in-memory feature store ─┘
    ├── GET  /health
    ├── POST /predict
    └── POST /batch-predict
```

## Tech stack

| Area | Tools |
|------|-------|
| Data | PostgreSQL 15, SQL (CTEs, window functions), pandas, Parquet |
| ML | XGBoost, scikit-learn, MLflow 3 (tracking server + model registry) |
| Serving | FastAPI, Pydantic v2, Uvicorn, structured JSON logging |
| Quality | pytest (unit + PostgreSQL integration), coverage, ruff, pre-commit |
| Infra | Docker (multi-stage), Docker Compose, GitHub Actions |

## Quickstart

Prerequisites: Docker, Python 3.11, and the Kaggle CSVs in `data/raw/`
(`kaggle datasets download olistbr/brazilian-ecommerce --unzip -p data/raw`).

```bash
# 1. Install dependencies and git hooks
python -m venv .venv && source .venv/bin/activate
make install

# 2. Start PostgreSQL and the MLflow server (UI on http://localhost:5001)
make up

# 3. Load data → build features → train and register the model
make pipeline

# 4. Build and start the API (Swagger UI on http://localhost:8000/docs)
make api

# 5. Run the tests
make test
```

`make help` lists every target. Each step can also be run directly, e.g.
`python -m src.features.build`. All settings (database, churn window, MLflow
URI, decision threshold) are environment variables documented in
[`.env.example`](.env.example) and [`src/config.py`](src/config.py).

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status, model version, feature store size, quarantined rows |
| POST | `/predict` | Churn probability for one customer |
| POST | `/batch-predict` | Churn probabilities for up to 1,000 customers, one model call |

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "01a22e2079ea71e17313b88e5811e54a"}'
```

```json
{
  "customer_id": "01a22e2079ea71e17313b88e5811e54a",
  "churn_probability": 0.9763,
  "churn_prediction": 1,
  "model_version": "churn-prediction/v1",
  "latency_ms": 5.05
}
```

Every response carries an `X-Request-ID` header, every error shares one JSON
schema, and every log line is a JSON object. Details in
[`src/api/README.md`](src/api/README.md).

## Model performance

Latest registered version (`churn-prediction/v1`), 1,632 repeat customers,
churn rate 95.8%.

| Metric | Value | Reference |
|--------|-------|-----------|
| ROC-AUC, repeated 5×5 CV (train) | **0.723 ± 0.058** | logistic regression: 0.707 |
| ROC-AUC, hold-out | 0.637 | only 17 retained customers in the hold-out set |
| PR-AUC, retained class | 0.154 | base rate: 0.042 (×3.7 lift) |
| Brier score | 0.0391 | always predicting the base rate: 0.0399 |

The signal is real but modest, and the evaluation is honest about it: with
only 68 retained customers in the whole dataset, a single hold-out AUC is
noisy, so cross-validation is the reference number. Precision/recall at the
0.5 threshold are logged in MLflow but are not meaningful here (see
[Decisions](#data--modeling-decisions)).

## Testing

```bash
make test          # 71 tests; integration tests are skipped without PostgreSQL
make coverage      # coverage report, fails under 70%
```

| Suite | Scope |
|-------|-------|
| `tests/unit` | feature engineering, data-quality rules, training pipeline (with a throw-away MLflow backend), feature store, API endpoints, error handling, logging |
| `tests/integration` | `rfm.sql` executed on PostgreSQL against a hand-built fixture where every expected value is known, in an isolated schema |

Coverage: 99% of `src/` (the one-off CSV loader is excluded). CI runs lint,
both test suites on a PostgreSQL service, then builds and smoke-tests the
Docker image.

## Incidents and post-mortems

Four failure scenarios were injected and fixed at the end of P1, each with a
written post-mortem in [`docs/incidents/`](docs/incidents/):

1. [Silent NaN features](docs/incidents/01-silent-nan-features.md) — the API returned plausible scores for corrupted rows
2. [Unexplained latency](docs/incidents/02-unexplained-latency.md) — no way to locate a slow stage; batch scoring 190× faster after the fix
3. [Validation errors](docs/incidents/03-validation-errors.md) — unreadable 422s and plain-text 500s
4. [SQL join fan-out](docs/incidents/04-sql-join-fan-out.md) — a join at the wrong grain silently skewed two features

## Data & modeling decisions

**Churn definition.** A customer is churned if they place no delivered order
in the 180 days after a cutoff date. Features use only orders *before* the
cutoff, the label only orders *after* it, so the label cannot leak into the
features (an early version computed recency on the same window as the label,
which gave a suspiciously high AUC). With the latest order on 2018-08-29, the
cutoff is 2018-03-02.

**Repeat customers only (`frequency >= 2`).** About 97% of Olist customers
order exactly once. Churn is only meaningful for customers who showed repeat
behaviour; including one-time buyers gives a 98.8% churn rate and an AUC of
0.58, with no learnable signal.

**180 days rather than 90.** Measured on the data: a 90-day window leaves 41
retained customers out of 2,226 (98.2% churn), 180 days leaves 68 out of 1,632
(95.8%). The longer window gives the minority class 65% more examples, which
matters more than the slightly larger population. The window is configurable
through `CHURN_DAYS`.

**No class re-weighting.** On repeated 5×5 CV, `scale_pos_weight` raised
ROC-AUC from 0.672 to 0.680 but pushed the Brier score from 0.039 to 0.113.
The API serves probabilities, so calibration wins. A shallower, regularized
model (depth 2, 200 trees) gives the best ROC-AUC *and* a calibrated output.

**Probability first, label second.** With a 96% base rate, a calibrated model
scores almost every customer above 0.5, so `churn_prediction` is 1 for nearly
everyone at the default threshold. The probability is the useful output: it
ranks customers by retention potential. The business threshold depends on the
cost of a retention action and is configurable (`CHURN_DECISION_THRESHOLD`).

**XGBoost over logistic regression.** The two are close (0.723 vs 0.707 CV
AUC), which says the current signal is mostly carried by recency. XGBoost is
kept because P2–P3 add non-linear features (reviews, delivery delays, product
categories); the logistic baseline is logged with every run to keep that
choice honest.

## Project structure

```
src/
├── config.py           # all settings, from environment variables
├── logging_config.py   # text logs for CLIs, JSON logs for the API
├── data/               # raw CSV → PostgreSQL loader
├── features/           # rfm.sql, data-quality checks, feature engineering
├── models/             # training, evaluation, MLflow registration
└── api/                # FastAPI app, schemas, validated feature store
tests/
├── unit/
└── integration/        # PostgreSQL-backed tests for rfm.sql
docs/incidents/         # post-mortems
Dockerfile · docker-compose.yml · Makefile · pyproject.toml
```

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| P1 — Foundations | SQL pipeline, model, FastAPI, Docker, CI/CD, tests, incidents | ✅ Done |
| P2 — Cloud + LLM intro | AWS/GCP, Terraform, orchestration, `/explain` endpoint, FinOps | ⬜ Next |
| P3 — MLOps + Agents | Drift monitoring, retraining, RAG, tool calling, guardrails | ⬜ Planned |
| P4 — Senior architecture | Complex agent, red-teaming, ROI, observability, RFC | ⬜ Planned |

Known limitations carried into P2: features are batch-computed and read from a
Parquet file at startup (no online feature store), the model is retrained
manually, and the serving image is ~170 MB compressed / ~600 MB on disk,
dominated by `pyarrow` and `scipy`.
