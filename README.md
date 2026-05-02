# Olist Customer Churn MLOps

End-to-end ML system predicting customer churn on Olist e-commerce data — 
FastAPI, Docker, MLflow, LLM agents (in progress).

## Tech Stack
- **Database**: PostgreSQL 15 (Docker)
- **Language**: Python 3.11.9
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
python src/features/sql_features.py

# 5. Build engineered features
python src/features/build.py

# 6. Train the model and log to MLflow
python src/models/train.py

# 7. View MLflow results
mlflow ui  # open http://localhost:5000

# 8. Run tests
pytest -v
```

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| P1 — Foundations | Pipeline + FastAPI + Docker + CI/CD | 🔄 In progress |
| P2 — Cloud + LLM intro | AWS/GCP + Terraform + Orchestration | ⬜ Pending |
| P3 — MLOps + Agents | MLflow + RAG + Tool calling + Guardrails | ⬜ Pending |
| P4 — Senior architecture | RFC + Red-teaming + FinOps + Observability | ⬜ Pending |


## Data & Modeling Decisions

### Why frequency >= 2 ?
~97% of Olist customers placed exactly one order. Churn is only meaningful
for customers who demonstrated repeat purchase behavior. Training on one-time
buyers produces a trivially imbalanced problem (98.8% churn rate, AUC=0.58)
with no learnable signal. The model is restricted to customers with at least
2 orders (~2,226 customers).

### Why churn_days = 90 ?
- 180 days → 98.8% churn rate on repeat customers (no learnable signal)
- 90 days → better class distribution, AUC=0.797 on repeat customers

### Cutoff date approach
Features and labels are computed on separate time windows to prevent data leakage:
- **Features**: computed on orders BEFORE `cutoff_date` (ref_date - churn_days)
- **Label**: defined by presence/absence of orders AFTER `cutoff_date`
- `recency_days` no longer encodes the label directly

### Handling class imbalance
Residual class imbalance (neg/pos ratio = 0.019) handled via XGBoost
`scale_pos_weight`, computed automatically at training time and logged in MLflow.

### Future improvements
- Clustering-based customer segmentation as additional features (planned for P4)
- Fine-grained RFM segments (loyal / seasonal / at-risk)