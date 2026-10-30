"""
FastAPI prediction service for the Olist churn model.

Endpoints:
    GET  /health        — service status, model version, feature store size
    POST /predict       — single customer churn score
    POST /batch-predict — churn scores for up to 1000 customers

Every request gets a request id (X-Request-ID, generated when absent) that is
returned in the response headers and attached to every JSON log line.

Pipeline position: features.parquet + MLflow Registry → [this file] → JSON response
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

import mlflow
import mlflow.xgboost
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import config
from src.api.feature_store import CustomerStatus, FeatureStore
from src.api.schemas import (
    BatchPredictItem,
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.logging_config import log_event, request_id_var, setup_logging

setup_logging(json_format=True)
logger = logging.getLogger(__name__)

FEATURES_PATH = config.FEATURES_PATH
MODEL_URI = config.MODEL_URI
DECISION_THRESHOLD = config.CHURN_DECISION_THRESHOLD


# ── Application state ──────────────────────────────────────────────────────────
class AppState:
    model = None
    feature_store: FeatureStore | None = None
    model_version: str = "unknown"


state = AppState()


def resolve_model_uri(model_uri: str) -> tuple[str, str]:
    """
    Pin a "models:/<name>/latest" URI to the concrete registry version.

    Logging the alias alone makes it impossible to tell which model served a
    prediction once a new version is registered. Returns (pinned_uri, label).
    """
    prefix, _, ref = model_uri.rpartition("/")
    if not model_uri.startswith("models:/") or ref != "latest":
        return model_uri, model_uri
    name = prefix.removeprefix("models:/")
    versions = mlflow.MlflowClient().search_model_versions(
        f"name='{name}'", order_by=["version_number DESC"], max_results=1
    )
    if not versions:
        raise RuntimeError(f"No registered version found for model '{name}'")
    version = versions[0].version
    return f"models:/{name}/{version}", f"{name}/v{version}"


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    state.feature_store = FeatureStore.from_parquet(FEATURES_PATH)
    log_event(
        logger,
        "feature_store_loaded",
        path=str(FEATURES_PATH),
        n_customers=len(state.feature_store),
        n_quarantined=len(state.feature_store.quarantined),
    )

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    try:
        pinned_uri, state.model_version = resolve_model_uri(MODEL_URI)
        state.model = mlflow.xgboost.load_model(pinned_uri)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from MLflow Registry: {exc}") from exc
    log_event(logger, "model_loaded", model_uri=pinned_uri, model_version=state.model_version)

    yield

    logger.info("Shutting down — releasing resources.")
    state.model = None
    state.feature_store = None


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Olist Churn Prediction API",
    description="Predicts churn probability for Olist repeat customers.",
    version="0.2.0",
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)


# ── Middleware ─────────────────────────────────────────────────────────────────
@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id, time the request and emit one access log per call."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    status_code = 500
    try:
        try:
            response = await call_next(request)
        except Exception:
            # Handled here rather than in an exception handler so the traceback
            # is logged while the request id is still in context.
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            response = error_response(
                request,
                500,
                "internal_error",
                "Unexpected server error. Quote the request_id when reporting the issue.",
            )
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        log_event(
            logger,
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        request_id_var.reset(token)


# ── Error handlers ─────────────────────────────────────────────────────────────
# All errors share the ErrorResponse shape and carry the request id, so a
# client report can be matched with the server logs.


def error_response(request: Request, status_code: int, error: str, message: str, details=None):
    request_id = getattr(request.state, "request_id", None)
    body = ErrorResponse(error=error, message=message, request_id=request_id, details=details)
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(exclude_none=True),
        headers={"X-Request-ID": request_id} if request_id else None,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        loc = [str(part) for part in err["loc"] if part != "body"]
        field = ".".join(loc) or "body"
        if err["type"] == "json_invalid":
            issue = f"Malformed JSON: {err.get('ctx', {}).get('error', 'decode error')}"
            details.append({"field": "body", "issue": issue})
        elif err["type"] == "extra_forbidden":
            details.append({"field": field, "issue": "Unknown field", "received": err["input"]})
        elif err["type"] == "missing":
            details.append({"field": field, "issue": "Field required"})
        else:
            details.append({"field": field, "issue": err["msg"], "received": err.get("input")})

    summary = "; ".join(f"{d['field']}: {d['issue']}" for d in details)
    log_event(logger, "request_rejected", level=logging.WARNING, reason=summary)
    return error_response(request, 422, "validation_error", f"Invalid request — {summary}", details)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    codes = {404: "not_found", 422: "invalid_features", 503: "service_unavailable"}
    return error_response(request, exc.status_code, codes.get(exc.status_code, "error"), exc.detail)


# ── Helpers ────────────────────────────────────────────────────────────────────
def require_ready() -> tuple:
    if state.model is None or state.feature_store is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check startup logs.")
    return state.model, state.feature_store


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service status, model version and feature store size."""
    store = state.feature_store
    return HealthResponse(
        status="ok" if state.model is not None and store is not None else "degraded",
        model_version=state.model_version,
        n_customers_in_store=len(store) if store is not None else 0,
        n_quarantined=len(store.quarantined) if store is not None else 0,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Churn probability for a single customer.

    Errors:
        404 — customer_id not found in the feature store
        422 — invalid payload, or features quarantined for this customer
        503 — model not loaded
    """
    start = time.perf_counter()
    model, store = require_ready()

    status = store.status(request.customer_id)
    if status is CustomerStatus.NOT_FOUND:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{request.customer_id}' not found in feature store.",
        )
    if status is CustomerStatus.INVALID_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"Features for customer '{request.customer_id}' failed validation; "
            "no score is served until the feature store is rebuilt.",
        )

    X = store.get_features([request.customer_id])
    lookup_ms = elapsed_ms(start)

    churn_proba = float(model.predict_proba(X)[0, 1])
    churn_pred = int(churn_proba >= DECISION_THRESHOLD)
    latency_ms = elapsed_ms(start)

    log_event(
        logger,
        "predict",
        customer_id=request.customer_id,
        churn_probability=round(churn_proba, 6),
        churn_prediction=churn_pred,
        model_version=state.model_version,
        lookup_ms=lookup_ms,
        inference_ms=round(latency_ms - lookup_ms, 3),
        latency_ms=latency_ms,
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
    Churn scores for a list of customer_ids, scored in a single model call.

    Unknown or quarantined ids are not errors: they come back with
    status not_found / invalid_features and null scores.

    Errors:
        422 — invalid payload (empty list, more than 1000 ids, wrong types)
        503 — model not loaded
    """
    start = time.perf_counter()
    model, store = require_ready()

    statuses = [store.status(cid) for cid in request.customer_ids]
    scorable = [
        cid for cid, s in zip(request.customer_ids, statuses, strict=True) if s is CustomerStatus.OK
    ]
    X = store.get_features(scorable) if scorable else None
    lookup_ms = elapsed_ms(start)

    probas = iter(model.predict_proba(X)[:, 1] if scorable else [])
    results = []
    for cid, status in zip(request.customer_ids, statuses, strict=True):
        item = BatchPredictItem(
            customer_id=cid, status=status, found=status is not CustomerStatus.NOT_FOUND
        )
        if status is CustomerStatus.OK:
            item.churn_probability = float(next(probas))
            item.churn_prediction = int(item.churn_probability >= DECISION_THRESHOLD)
        results.append(item)
    latency_ms = elapsed_ms(start)

    counts = {s: statuses.count(s) for s in CustomerStatus}
    n_not_found = counts[CustomerStatus.NOT_FOUND]
    log_event(
        logger,
        "batch_predict",
        n_requested=len(results),
        n_scored=counts[CustomerStatus.OK],
        n_not_found=n_not_found,
        n_invalid=counts[CustomerStatus.INVALID_FEATURES],
        model_version=state.model_version,
        lookup_ms=lookup_ms,
        inference_ms=round(latency_ms - lookup_ms, 3),
        latency_ms=latency_ms,
    )
    return BatchPredictResponse(
        results=results,
        n_requested=len(results),
        n_found=len(results) - n_not_found,
        n_not_found=n_not_found,
        n_invalid=counts[CustomerStatus.INVALID_FEATURES],
        model_version=state.model_version,
        latency_ms=latency_ms,
    )
