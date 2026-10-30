# API

FastAPI service serving churn probabilities from the latest registered model.

## Pipeline position

```
MLflow registry (models:/churn-prediction/latest) ─┐
                                                   ├─► main.py ─► JSON responses
data/processed/features.parquet ───────────────────┘
```

## Run

```bash
make api                                   # Docker (recommended)
uvicorn src.api.main:app --reload          # local, needs `make up` and a trained model
```

Swagger UI: http://localhost:8000/docs

## Modules

| File | Role |
|------|------|
| `main.py` | App, lifespan (loads feature store + model), middleware, error handlers, endpoints |
| `schemas.py` | Request/response contracts, including the shared error schema |
| `feature_store.py` | In-memory feature store: validation, quarantine, O(1) lookups |
| `log_config.json` | Uvicorn logging config so server logs are JSON too |

## Startup

1. Load `features.parquet` and validate every row. Rows with a non-finite or
   negative model input are **quarantined** (never scored). If more than 5% of
   rows are invalid, startup fails.
2. Resolve `models:/churn-prediction/latest` to the concrete version and load
   it from the MLflow server.

## Endpoints

### `GET /health`

```json
{"status": "ok", "model_version": "churn-prediction/v1", "n_customers_in_store": 1632, "n_quarantined": 0}
```

`status` is `degraded` when the model or the feature store is not loaded.

### `POST /predict`

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

### `POST /batch-predict`

Up to 1,000 ids, scored in a single model call. Unknown or quarantined ids are
not errors: they are returned in request order with a `status` and null scores.

```json
{
  "results": [
    {"customer_id": "01a22e…", "status": "ok", "found": true, "churn_probability": 0.9763, "churn_prediction": 1},
    {"customer_id": "unknown", "status": "not_found", "found": false, "churn_probability": null, "churn_prediction": null}
  ],
  "n_requested": 2, "n_found": 1, "n_not_found": 1, "n_invalid": 0,
  "model_version": "churn-prediction/v1", "latency_ms": 0.9
}
```

## Errors

Every error has the same shape and carries the request id:

```json
{
  "error": "validation_error",
  "message": "Invalid request — customer_id: Field required; customerId: Unknown field",
  "request_id": "7b97d2e51d764f0c8d9d88569e794117",
  "details": [
    {"field": "customer_id", "issue": "Field required"},
    {"field": "customerId", "issue": "Unknown field", "received": "…"}
  ]
}
```

| Status | `error` | Cause |
|--------|---------|-------|
| 404 | `not_found` | `customer_id` not in the feature store |
| 422 | `validation_error` | Invalid payload: wrong type, missing or unknown field, malformed JSON, batch size outside 1–1000 |
| 422 | `invalid_features` | Customer's features failed validation and are quarantined |
| 500 | `internal_error` | Unexpected failure — details are in the logs, under the same `request_id` |
| 503 | `service_unavailable` | Model or feature store not loaded |

## Observability

- `X-Request-ID` is read from the request (or generated) and returned on every
  response.
- Logs are one JSON object per line, all tagged with the request id:

```json
{"timestamp": "2026-10-30T15:34:16.035+00:00", "level": "INFO", "logger": "src.api.main",
 "message": "predict", "request_id": "demo-1", "event": "predict",
 "customer_id": "01a22e…", "churn_probability": 0.976324, "churn_prediction": 1,
 "model_version": "churn-prediction/v1", "lookup_ms": 0.575, "inference_ms": 4.471, "latency_ms": 5.046}
```

| Event | When |
|-------|------|
| `feature_store_loaded`, `feature_store_rows_quarantined`, `model_loaded` | Startup |
| `http_request` | Every request: method, path, status, duration |
| `predict`, `batch_predict` | Scores, model version, `lookup_ms` / `inference_ms` / `latency_ms` |
| `request_rejected` | Validation failures, with a one-line reason |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | `http://mlflow:5000` in Docker Compose |
| `MODEL_URI` | `models:/churn-prediction/latest` | Any MLflow model URI; `latest` is pinned at startup |
| `CHURN_DECISION_THRESHOLD` | `0.5` | Probability above which `churn_prediction = 1` |
| `PROCESSED_DATA_DIR` | `data/processed` | Location of `features.parquet` |

## Docker image

Multi-stage build on `python:3.11-slim`, serving dependencies only
(`requirements-api.txt`: CPU-only XGBoost, `mlflow-skinny`), non-root user,
`HEALTHCHECK` on `/health`. Size: ~170 MB compressed, ~600 MB on disk —
`pyarrow` (129 MB) and `scipy` (75 MB) are the largest layers.
