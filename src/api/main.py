"""
FastAPI prediction service for the Olist churn model.

Endpoints:
    GET  /health        — service status + model version
    POST /predict       — single customer churn score
    POST /batch-predict — batch churn scores for up to 1000 customers

Pipeline position: features.parquet + MLflow Registry → [this file] → JSON response
"""

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"

# ── Model URI ──────────────────────────────────────────────────────────────────
MODEL_URI = "models:/churn-prediction/latest"

# ── Feature columns (must match train.py order) ────────────────────────────────
FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "monetary_log",
    "avg_order_value",
    "days_per_order",
    "avg_days_between_orders",
    "customer_lifetime_days",
    "is_one_time_buyer",
]


# ── Application state ──────────────────────────────────────────────────────────
class AppState:
    model                        = None
    features: pd.DataFrame       = None
    features_indexed: pd.DataFrame = None  # indexed once at startup for O(1) lookup
    model_version: str           = "unknown"


state = AppState()


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading feature store from: %s", FEATURES_PATH)
    if not FEATURES_PATH.exists():
        raise RuntimeError(f"Feature store not found: {FEATURES_PATH}")
    state.features         = pd.read_parquet(FEATURES_PATH)
    state.features_indexed = state.features.set_index("customer_unique_id")
    logger.info("Feature store loaded: %s customers", f"{len(state.features):,}")

    logger.info("Loading model from MLflow Registry: %s", MODEL_URI)
    try:
        state.model         = mlflow.sklearn.load_model(MODEL_URI)
        state.model_version = MODEL_URI
        logger.info("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow Registry: {e}")

    yield  # app is running

    # Shutdown
    logger.info("Shutting down — releasing resources.")
    state.model            = None
    state.features         = None
    state.features_indexed = None


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Olist Churn Prediction API",
    description="Predicts churn probability for Olist repeat customers.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    customer_id: str = Field(
        ...,
        min_length=1,
        description="Unique customer identifier (customer_unique_id in Olist dataset)",
    )


class PredictResponse(BaseModel):
    customer_id:       str
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction:  int   = Field(ge=0, le=1)
    model_version:     str
    latency_ms:        float


class HealthResponse(BaseModel):
    status:               str
    model_version:        str
    n_customers_in_store: int


class BatchPredictRequest(BaseModel):
    customer_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of customer_unique_id to score (max 1000 per call).",
    )


class BatchPredictItem(BaseModel):
    customer_id:       str
    churn_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    churn_prediction:  Optional[int]   = Field(default=None, ge=0, le=1)
    found:             bool


class BatchPredictResponse(BaseModel):
    results:       list[BatchPredictItem]
    n_requested:   int
    n_found:       int
    n_not_found:   int
    model_version: str
    latency_ms:    float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Returns service status and model version.
    Use this endpoint to verify the API is running and the model is loaded.
    """
    return HealthResponse(
        status="ok",
        model_version=state.model_version,
        n_customers_in_store=len(state.features) if state.features is not None else 0,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Returns churn probability for a single customer.

    Errors:
        404 — customer_id not found in feature store
        503 — model not loaded (check startup logs)
    """
    start = time.perf_counter()

    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check startup logs.",
        )

    customer_row = state.features[
        state.features["customer_unique_id"] == request.customer_id
    ]

    if customer_row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{request.customer_id}' not found in feature store.",
        )

    X           = customer_row[FEATURE_COLUMNS].values
    churn_proba = float(state.model.predict_proba(X)[0][1])
    churn_pred  = int(state.model.predict(X)[0])
    latency_ms  = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "predict | customer_id=%s | churn_proba=%.4f | prediction=%d"
        " | latency_ms=%.2f | model=%s",
        request.customer_id,
        churn_proba,
        churn_pred,
        latency_ms,
        state.model_version,
    )

    return PredictResponse(
        customer_id=request.customer_id,
        churn_probability=churn_proba,
        churn_prediction=churn_pred,
        model_version=state.model_version,
        latency_ms=latency_ms,
    )


@app.post("/batch-predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Returns churn scores for a list of customer_ids.

    - IDs not found in the feature store are returned with found=False
      and null scores — they are NOT raised as errors.
    - Empty list → rejected at schema level (min_length=1).
    - Max 1000 IDs per call to prevent abuse.

    Errors:
        503 — model not loaded
    """
    start = time.perf_counter()

    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check startup logs.",
        )

    results: list[BatchPredictItem] = []
    indexed = state.features_indexed

    for cid in request.customer_ids:
        if cid not in indexed.index:
            results.append(BatchPredictItem(
                customer_id=cid,
                churn_probability=None,
                churn_prediction=None,
                found=False,
            ))
            continue

        X           = indexed.loc[cid, FEATURE_COLUMNS].values.reshape(1, -1)
        churn_proba = float(state.model.predict_proba(X)[0][1])
        churn_pred  = int(state.model.predict(X)[0])

        results.append(BatchPredictItem(
            customer_id=cid,
            churn_probability=churn_proba,
            churn_prediction=churn_pred,
            found=True,
        ))

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    n_found    = sum(r.found for r in results)

    logger.info(
        "batch-predict | n_requested=%d | n_found=%d | n_not_found=%d"
        " | latency_ms=%.2f | model=%s",
        len(request.customer_ids),
        n_found,
        len(request.customer_ids) - n_found,
        latency_ms,
        state.model_version,
    )

    return BatchPredictResponse(
        results=results,
        n_requested=len(request.customer_ids),
        n_found=n_found,
        n_not_found=len(request.customer_ids) - n_found,
        model_version=state.model_version,
        latency_ms=latency_ms,
    )