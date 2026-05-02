"""
FastAPI prediction service for the Olist churn model.

Endpoints:
    POST /predict       — single customer churn score
    GET  /health        — service status + model version
    POST /batch-predict — batch churn scores (Week 6)

Pipeline position: features.parquet + MLflow Registry → [this file] → JSON response
"""

import time
import logging
from pathlib import Path

import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Must match tracking URI set in train.py
import mlflow
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

# ── Model URI — loaded from MLflow Registry, no hardcoded run_id ──────────────
MODEL_URI = "models:/churn-prediction/latest"

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Olist Churn Prediction API",
    description="Predicts churn probability for Olist repeat customers.",
    version="0.1.0",
)

# ── Feature columns fed into the model (must match train.py order) ────────────
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


# ── Application state — loaded once at startup, reused on every request ───────
class AppState:
    model = None
    features: pd.DataFrame = None
    model_version: str = "unknown"


state = AppState()


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup() -> None:
    """
    Load model and feature store at startup.
    Fails fast if either is missing — better to crash early
    than to serve wrong predictions silently.
    """
    # Load feature store
    logger.info("Loading feature store from: %s", FEATURES_PATH)
    if not FEATURES_PATH.exists():
        raise RuntimeError(f"Feature store not found: {FEATURES_PATH}")
    state.features = pd.read_parquet(FEATURES_PATH)
    logger.info("Feature store loaded: %s customers", f"{len(state.features):,}")

    # Load model from MLflow Registry
    logger.info("Loading model from MLflow Registry: %s", MODEL_URI)
    try:
        state.model = mlflow.sklearn.load_model(MODEL_URI)
        state.model_version = MODEL_URI
        logger.info("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow Registry: {e}")


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
    status:                str
    model_version:         str
    n_customers_in_store:  int


# ── Endpoints ─────────────────────────────────────────────────────────────────

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

    # Guard: model must be loaded
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check startup logs.",
        )

    # Lookup customer in feature store
    customer_row = state.features[
        state.features["customer_unique_id"] == request.customer_id
    ]

    if customer_row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{request.customer_id}' not found in feature store.",
        )

    # Extract feature vector and predict
    X             = customer_row[FEATURE_COLUMNS].values
    churn_proba   = float(state.model.predict_proba(X)[0][1])
    churn_pred    = int(state.model.predict(X)[0])
    latency_ms    = round((time.perf_counter() - start) * 1000, 2)

    # Structured log on every call
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