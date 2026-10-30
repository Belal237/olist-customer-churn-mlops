"""
Unit tests for src/api/main.py.
Uses FastAPI TestClient with a fake model and a temporary feature store —
no MLflow server or real model needed.
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, state


@pytest.fixture
def fake_feature_store():
    return pd.DataFrame(
        {
            "customer_unique_id": ["cust_001", "cust_002", "cust_nan"],
            "recency_days": [10, 200, np.nan],
            "frequency": [3, 5, 2],
            "monetary": [150.0, 400.0, 80.0],
            "monetary_log": [np.log1p(150.0), np.log1p(400.0), np.log1p(80.0)],
            "avg_order_value": [50.0, 80.0, 40.0],
            "days_per_order": [30.0, 60.0, 10.0],
            "avg_days_between_orders": [45.0, 60.0, 20.0],
            "customer_lifetime_days": [90, 300, 20],
            "is_one_time_buyer": [0, 0, 0],
            "churn_label": [0, 1, 1],
        }
    )


@pytest.fixture
def fake_model():
    """Returns P(churn) = 0.2 + 0.001 * recency_days, so scores are row-specific."""
    model = MagicMock()
    model.predict_proba.side_effect = lambda X: np.column_stack(
        [0.8 - 0.001 * X["recency_days"].to_numpy(), 0.2 + 0.001 * X["recency_days"].to_numpy()]
    )
    return model


@pytest.fixture
def client(monkeypatch, tmp_path, fake_feature_store, fake_model):
    features_path = tmp_path / "features.parquet"
    fake_feature_store.to_parquet(features_path, index=False)

    monkeypatch.setattr("src.api.main.FEATURES_PATH", features_path)
    monkeypatch.setattr(
        "src.api.main.resolve_model_uri", lambda uri: ("models:/churn/7", "churn/v7")
    )
    monkeypatch.setattr(
        "src.api.main.mlflow.xgboost.load_model", MagicMock(return_value=fake_model)
    )
    # One bad row out of three is above the production 5% limit — allow it here
    monkeypatch.setattr("src.api.feature_store.MAX_QUARANTINE_RATIO", 0.5)

    with TestClient(app) as test_client:
        yield test_client


def events(caplog, name: str) -> list[dict]:
    return [r.fields for r in caplog.records if getattr(r, "fields", {}).get("event") == name]


# ── /health ────────────────────────────────────────────────────────────────────


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "churn/v7",
        "n_customers_in_store": 3,
        "n_quarantined": 1,
    }


# ── /predict ───────────────────────────────────────────────────────────────────


def test_predict_known_customer(client):
    response = client.post("/predict", json={"customer_id": "cust_001"})
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "cust_001"
    assert data["churn_probability"] == pytest.approx(0.21)
    assert data["churn_prediction"] == 0
    assert data["model_version"] == "churn/v7"


def test_predict_applies_decision_threshold(client, monkeypatch):
    monkeypatch.setattr("src.api.main.DECISION_THRESHOLD", 0.2)
    response = client.post("/predict", json={"customer_id": "cust_001"})
    assert response.json()["churn_prediction"] == 1


def test_predict_logs_structured_event_with_stage_timings(client, caplog):
    with caplog.at_level(logging.INFO):
        client.post("/predict", json={"customer_id": "cust_001"})
    [event] = events(caplog, "predict")
    assert event["customer_id"] == "cust_001"
    assert event["churn_probability"] == pytest.approx(0.21)
    assert event["model_version"] == "churn/v7"
    assert {"lookup_ms", "inference_ms", "latency_ms"} <= event.keys()


def test_predict_unknown_customer_returns_404(client):
    response = client.post("/predict", json={"customer_id": "does_not_exist"})
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_predict_quarantined_customer_is_never_scored(client, fake_model):
    response = client.post("/predict", json={"customer_id": "cust_nan"})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_features"
    fake_model.predict_proba.assert_not_called()


def test_predict_model_not_loaded_returns_503(client):
    state.model = None
    response = client.post("/predict", json={"customer_id": "cust_001"})
    assert response.status_code == 503
    assert client.get("/health").json()["status"] == "degraded"


# ── Validation errors ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("payload", "field", "issue"),
    [
        ({"customer_id": ""}, "customer_id", "at least 1 character"),
        ({"customer_id": 123}, "customer_id", "valid string"),
        ({}, "customer_id", "Field required"),
        ({"customer_id": "cust_001", "customerId": "x"}, "customerId", "Unknown field"),
    ],
)
def test_predict_invalid_payload_returns_readable_422(client, payload, field, issue):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert any(d["field"] == field and issue in d["issue"] for d in body["details"])
    assert field in body["message"]


def test_malformed_json_returns_readable_422(client):
    response = client.post(
        "/predict", content="{bad json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["details"][0]["issue"].startswith("Malformed JSON")


# ── Request context & internal errors ──────────────────────────────────────────


def test_request_id_is_propagated(client, caplog):
    with caplog.at_level(logging.INFO):
        response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    [access_log] = events(caplog, "http_request")
    assert access_log["path"] == "/health"
    assert access_log["status_code"] == 200


def test_unhandled_error_returns_json_500_with_request_id(client, fake_model, caplog):
    fake_model.predict_proba.side_effect = ValueError("feature_names mismatch")
    with caplog.at_level(logging.INFO):
        response = client.post("/predict", json={"customer_id": "cust_001"})
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert "feature_names" not in body["message"]  # no internals leaked to clients
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert any(r.exc_info for r in caplog.records)
    assert events(caplog, "http_request")[0]["status_code"] == 500


# ── /batch-predict ─────────────────────────────────────────────────────────────


def test_batch_predict_all_found(client, fake_model):
    response = client.post("/batch-predict", json={"customer_ids": ["cust_001", "cust_002"]})
    assert response.status_code == 200
    data = response.json()
    assert data["n_found"] == 2
    assert data["n_not_found"] == 0
    assert [r["churn_probability"] for r in data["results"]] == pytest.approx([0.21, 0.4])
    # Scored in a single model call, not one call per customer
    assert fake_model.predict_proba.call_count == 1


def test_batch_predict_mixed_statuses_keep_request_order(client):
    ids = ["unknown_id", "cust_002", "cust_nan", "cust_001"]
    response = client.post("/batch-predict", json={"customer_ids": ids})
    assert response.status_code == 200
    data = response.json()
    assert [r["customer_id"] for r in data["results"]] == ids
    assert [r["status"] for r in data["results"]] == [
        "not_found",
        "ok",
        "invalid_features",
        "ok",
    ]
    assert (data["n_found"], data["n_not_found"], data["n_invalid"]) == (3, 1, 1)
    assert data["results"][2]["churn_probability"] is None


def test_batch_predict_none_found(client, fake_model):
    response = client.post("/batch-predict", json={"customer_ids": ["ghost_1", "ghost_2"]})
    assert response.status_code == 200
    data = response.json()
    assert data["n_found"] == 0
    assert data["n_not_found"] == 2
    assert all(item["found"] is False for item in data["results"])
    fake_model.predict_proba.assert_not_called()


def test_batch_predict_logs_structured_event(client, caplog):
    with caplog.at_level(logging.INFO):
        client.post("/batch-predict", json={"customer_ids": ["cust_001", "ghost"]})
    [event] = events(caplog, "batch_predict")
    assert (event["n_requested"], event["n_scored"], event["n_not_found"]) == (2, 1, 1)


@pytest.mark.parametrize("ids", [[], [f"c{i}" for i in range(1001)]])
def test_batch_predict_size_limits_return_422(client, ids):
    response = client.post("/batch-predict", json={"customer_ids": ids})
    assert response.status_code == 422


def test_batch_predict_model_not_loaded_returns_503(client):
    state.model = None
    response = client.post("/batch-predict", json={"customer_ids": ["cust_001"]})
    assert response.status_code == 503
