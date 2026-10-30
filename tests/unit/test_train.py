"""
Unit tests for src/models/train.py on a synthetic, deliberately learnable dataset.
The end-to-end test logs to a throw-away SQLite MLflow backend in tmp_path.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURE_COLUMNS, MODEL_FEATURES
from src.models.train import (
    build_baseline,
    build_model,
    cross_validate_auc,
    evaluate,
    load_data,
    split_data,
    train,
)


@pytest.fixture
def features_df() -> pd.DataFrame:
    """400 customers, ~85% churners; retained customers have low recency."""
    rng = np.random.default_rng(0)
    n = 400
    churn = (rng.random(n) < 0.85).astype(int)
    recency = np.where(churn == 1, rng.integers(60, 400, n), rng.integers(0, 80, n))
    frequency = rng.integers(2, 6, n)
    monetary = rng.gamma(2.0, 80.0, n).round(2)
    lifetime = rng.integers(0, 400, n)
    return pd.DataFrame(
        {
            "customer_unique_id": [f"c{i}" for i in range(n)],
            "recency_days": recency,
            "frequency": frequency,
            "monetary": monetary,
            "monetary_log": np.log1p(monetary),
            "avg_order_value": monetary / frequency,
            "days_per_order": lifetime / frequency,
            "avg_days_between_orders": lifetime / (frequency - 1),
            "customer_lifetime_days": lifetime,
            "is_one_time_buyer": 0,
            "churn_label": churn,
        }
    )[FEATURE_COLUMNS]


@pytest.fixture
def features_path(tmp_path, features_df):
    path = tmp_path / "features.parquet"
    features_df.to_parquet(path, index=False)
    return path


def test_load_data_rejects_missing_columns(tmp_path, features_df):
    path = tmp_path / "broken.parquet"
    features_df.drop(columns=["monetary"]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="monetary"):
        load_data(path)


def test_load_data_rejects_single_class(tmp_path, features_df):
    path = tmp_path / "one_class.parquet"
    features_df.assign(churn_label=1).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="single class"):
        load_data(path)


def test_split_is_stratified_and_uses_model_features_only(features_df):
    X_train, X_test, y_train, y_test = split_data(features_df)
    assert list(X_train.columns) == MODEL_FEATURES
    assert abs(y_train.mean() - y_test.mean()) < 0.02


@pytest.mark.parametrize("factory", [build_model, build_baseline])
def test_cross_validate_auc_returns_one_score_per_fold(features_df, factory):
    X_train, _, y_train, _ = split_data(features_df)
    scores = cross_validate_auc(factory(), X_train, y_train)
    assert len(scores) == 25
    assert ((scores >= 0) & (scores <= 1)).all()


def test_evaluate_returns_bounded_metrics(features_df):
    X_train, X_test, y_train, y_test = split_data(features_df)
    model = build_model().fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    for name in ("roc_auc", "pr_auc_churn", "pr_auc_retained", "brier", "precision", "recall"):
        assert 0.0 <= metrics[name] <= 1.0
    assert metrics["churn_rate"] + metrics["retained_rate"] == pytest.approx(1.0)
    # The synthetic signal is strong: the model must beat random ranking clearly
    assert metrics["roc_auc"] > 0.9


def test_train_logs_and_registers_model(tmp_path, features_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    metrics = train(
        features_path=features_path,
        tracking_uri=tracking_uri,
        experiment_name="test-churn",
        registered_model_name="test-churn-model",
    )

    assert {"roc_auc", "cv_roc_auc_mean", "cv_roc_auc_std", "baseline_cv_roc_auc_mean"} <= set(
        metrics
    )

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions("name='test-churn-model'")
    assert len(versions) == 1
    run = client.get_run(versions[0].run_id)
    assert run.data.params["features"] == ",".join(MODEL_FEATURES)
    assert run.data.metrics["roc_auc"] == pytest.approx(metrics["roc_auc"])

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.xgboost.load_model("models:/test-churn-model/latest")
    assert model.predict_proba(pd.read_parquet(features_path)[MODEL_FEATURES]).shape[1] == 2
