# Olist Customer Churn MLOps

End-to-end ML system predicting customer churn on Olist e-commerce data — 
FastAPI, Docker, MLflow, LLM agents (in progress).

## Tech Stack
- **Database**: PostgreSQL 15 (Docker)
- **Language**: Python 3.13
- **ML**: XGBoost, MLflow
- **API**: FastAPI
- **Infra**: Docker, GitHub Actions
- **LLM**: RAG + Agents (Phase 3)

## Project Structure

- src/
  - data/  # Data loading and validation
  - features/ # Feature engineering (RFM)
  - models/ # Model training and evaluation
  - api/ # FastAPI prediction service
- tests/ # Unit and integration tests
- docker-compose.yml
- README.md
- requirements.txt

## Tests

Run the test suite:

```bash
pytest -v
```

Run with coverage report:

```bash
pytest -v --cov=src/features --cov-report=term-missing
```

### Current coverage

| Module | Coverage |
|--------|----------|
| `src/features/build.py` | 92% |
| `src/features/sql_features.py` | — (requires PostgreSQL) |

### Test structure

```
tests/
├── __init__.py
└── unit/
    ├── __init__.py
    └── test_build.py    # 11 tests — feature engineering pipeline
```

## Quickstart
```bash
# 1. Start PostgreSQL
docker compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Load dataset into PostgreSQL
python src/data/load.py

# 4. Build RFM features
python src/data/sql_features.py

# 5. Build engineered features
python src/features/build.py

# 6. Run tests
pytest -v
```

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| P1 — Foundations | Pipeline + FastAPI + Docker + CI/CD | 🔄 In progress |
| P2 — Cloud + LLM intro | AWS/GCP + Terraform + Orchestration | ⬜ Pending |
| P3 — MLOps + Agents | MLflow + RAG + Tool calling + Guardrails | ⬜ Pending |
| P4 — Senior architecture | RFC + Red-teaming + FinOps + Observability | ⬜ Pending |