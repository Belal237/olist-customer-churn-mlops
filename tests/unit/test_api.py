"""
Unit tests for src/api/main.py
Uses FastAPI TestClient — no real model or feature store needed.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app, state, FEATURE_COLUMNS


@pytest.fixture(autouse=True)
def setup_state():
    """
    Inject a minimal fake model and feature store into AppState
    before each test, then clean up after.
    """
    state.features = pd.DataFrame({
        "customer_unique_id":      ["cust_001", "cust_002"],
        "recency_days":            [10, 200],
        "frequency":               [3, 5],
        "monetary":                [150.0, 400.0],
        "monetary_log":            [np.log1p(150.0), np.log1p(400.0)],
        "avg_order_value":         [50.0, 80.0],
        "days_per_order":          [30.0, 60.0],
        "avg_days_between_orders": [45.0, 60.0],
        "customer_lifetime_days":  [90, 300],
        "is_one_time_buyer":       [0, 0],
        "churn_label":             [0, 1],
    })
    # Mirror what lifespan does at startup
    state.features_indexed = state.features.set_index("customer_unique_id")

    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.25, 0.75]])
    fake_model.predict.return_value       = np.array([1])
    state.model         = fake_model
    state.model_version = "models:/churn-prediction/latest"

    yield

    state.features         = None
    state.features_indexed = None
    state.model            = None
    state.model_version    = "unknown"


client = TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["n_customers_in_store"] == 2


# ── /predict ───────────────────────────────────────────────────────────────────

def test_predict_known_customer():
    response = client.post("/predict", json={"customer_id": "cust_001"})
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "cust_001"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in (0, 1)


def test_predict_unknown_customer_returns_404():
    response = client.post("/predict", json={"customer_id": "does_not_exist"})
    assert response.status_code == 404


def test_predict_empty_customer_id_returns_422():
    response = client.post("/predict", json={"customer_id": ""})
    assert response.status_code == 422


# ── /batch-predict ─────────────────────────────────────────────────────────────

def test_batch_predict_all_found():
    response = client.post(
        "/batch-predict",
        json={"customer_ids": ["cust_001", "cust_002"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["n_found"]     == 2
    assert data["n_not_found"] == 0
    assert len(data["results"]) == 2


def test_batch_predict_partial_not_found():
    response = client.post(
        "/batch-predict",
        json={"customer_ids": ["cust_001", "unknown_id"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["n_found"]     == 1
    assert data["n_not_found"] == 1


def test_batch_predict_none_found():
    response = client.post(
        "/batch-predict",
        json={"customer_ids": ["ghost_1", "ghost_2"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["n_found"]     == 0
    assert data["n_not_found"] == 2
    # Unknown IDs are returned with found=False and null scores, not silently dropped
    assert len(data["results"]) == 2
    assert all(r["found"] is False for r in data["results"])
    assert all(r["churn_probability"] is None for r in data["results"])
    assert all(r["churn_prediction"] is None for r in data["results"])


def test_batch_predict_empty_list_returns_422():
    response = client.post("/batch-predict", json={"customer_ids": []})
    assert response.status_code == 422


def test_batch_predict_model_not_loaded_returns_503():
    state.model = None
    response = client.post(
        "/batch-predict",
        json={"customer_ids": ["cust_001"]},
    )
    assert response.status_code == 503