# API

FastAPI prediction service for the Olist churn model.

## Pipeline position
MLflow Registry (models:/churn-prediction/latest)
data/processed/features.parquet
└── main.py (FastAPI)
└── POST /predict
└── GET  /health
└── POST /batch-predict

## Run

```bash
uvicorn src.api.main:app --reload
```

API available at http://localhost:8000
Swagger UI at http://localhost:8000/docs

## Endpoints

### GET /health

Returns service status and model version.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model_version": "models:/churn-prediction/latest",
  "n_customers_in_store": 2226
}
```

### POST /predict

Returns churn probability for a single customer.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "142dc6250eac579893595d9889411834"}'
```

Response:
```json
{
  "customer_id": "142dc6250eac579893595d9889411834",
  "churn_probability": 0.6549,
  "churn_prediction": 1,
  "model_version": "models:/churn-prediction/latest",
  "latency_ms": 2.47
}
```

## Error handling

| Code | Cause | Message |
|------|-------|---------|
| 404 | customer_id not found | "Customer '...' not found in feature store." |
| 422 | invalid payload | Pydantic validation error with field details |
| 503 | model not loaded | "Model not loaded. Check startup logs." |

## Logging

Every prediction is logged with:

predict | customer_id=... | churn_proba=... | prediction=... | latency_ms=... | model=...