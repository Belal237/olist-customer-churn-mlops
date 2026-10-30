"""Request and response contracts of the prediction API."""

from pydantic import BaseModel, ConfigDict, Field

from src.api.feature_store import CustomerStatus

CustomerId = Field(
    min_length=1,
    max_length=64,
    description="Unique customer identifier (customer_unique_id in the Olist dataset).",
    examples=["8d50f5eadf50201ccdcedfb9e2ac8455"],
)


class StrictRequest(BaseModel):
    # Unknown fields are rejected: a typo such as "customerId" must not be ignored
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PredictRequest(StrictRequest):
    customer_id: str = CustomerId


class BatchPredictRequest(StrictRequest):
    customer_ids: list[str] = Field(
        min_length=1,
        max_length=1000,
        description="customer_unique_id values to score (max 1000 per call).",
    )


class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: int = Field(ge=0, le=1)
    model_version: str
    latency_ms: float


class BatchPredictItem(BaseModel):
    customer_id: str
    status: CustomerStatus
    found: bool
    churn_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    churn_prediction: int | None = Field(default=None, ge=0, le=1)


class BatchPredictResponse(BaseModel):
    results: list[BatchPredictItem]
    n_requested: int
    n_found: int
    n_not_found: int
    n_invalid: int
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_customers_in_store: int
    n_quarantined: int


class ErrorDetail(BaseModel):
    field: str
    issue: str
    received: object | None = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
    details: list[ErrorDetail] | None = None
