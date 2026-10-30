"""Unit tests for src/logging_config.py and model version resolution."""

import json
import logging
from types import SimpleNamespace

from src.api import main
from src.logging_config import JsonFormatter, request_id_var


def make_record(**extra) -> logging.LogRecord:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "predict", None, None)
    record.__dict__.update(extra)
    return record


def test_json_formatter_flattens_structured_fields():
    line = JsonFormatter().format(make_record(fields={"event": "predict", "latency_ms": 1.5}))
    payload = json.loads(line)
    assert payload["message"] == "predict"
    assert payload["event"] == "predict"
    assert payload["latency_ms"] == 1.5
    assert payload["level"] == "INFO"
    assert payload["timestamp"].endswith("+00:00")


def test_json_formatter_includes_request_id_from_context():
    token = request_id_var.set("req-42")
    try:
        payload = json.loads(JsonFormatter().format(make_record()))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "req-42"


def test_resolve_model_uri_pins_latest_version(monkeypatch):
    client = SimpleNamespace(search_model_versions=lambda *a, **k: [SimpleNamespace(version="3")])
    monkeypatch.setattr(main.mlflow, "MlflowClient", lambda: client)
    assert main.resolve_model_uri("models:/churn-prediction/latest") == (
        "models:/churn-prediction/3",
        "churn-prediction/v3",
    )


def test_resolve_model_uri_keeps_explicit_versions():
    assert main.resolve_model_uri("models:/churn-prediction/2") == (
        "models:/churn-prediction/2",
        "models:/churn-prediction/2",
    )
