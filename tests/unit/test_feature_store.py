"""Unit tests for src/api/feature_store.py."""

import numpy as np
import pandas as pd
import pytest

from src.api.feature_store import CustomerStatus, FeatureStore, FeatureStoreError
from src.features.build import MODEL_FEATURES


def make_features(n: int = 100) -> pd.DataFrame:
    df = pd.DataFrame(
        np.arange(n * len(MODEL_FEATURES), dtype=float).reshape(n, -1), columns=MODEL_FEATURES
    )
    df.insert(0, "customer_unique_id", [f"c{i}" for i in range(n)])
    return df


def test_lookup_returns_rows_in_requested_order():
    df = make_features()
    store = FeatureStore(df)
    X = store.get_features(["c5", "c1"])
    assert list(X.columns) == MODEL_FEATURES
    np.testing.assert_array_equal(X.to_numpy(), df.loc[[5, 1], MODEL_FEATURES].to_numpy())


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1.0])
def test_invalid_rows_are_quarantined(bad_value):
    df = make_features()
    df.loc[3, "monetary"] = bad_value
    store = FeatureStore(df)
    assert store.quarantined == {"c3"}
    assert store.status("c3") is CustomerStatus.INVALID_FEATURES
    assert store.status("c4") is CustomerStatus.OK
    assert store.status("missing") is CustomerStatus.NOT_FOUND


def test_systemic_corruption_refuses_to_start():
    df = make_features()
    df.loc[:9, "recency_days"] = np.nan  # 10% of rows
    with pytest.raises(FeatureStoreError, match="failed validation"):
        FeatureStore(df)


def test_missing_model_column_refuses_to_start():
    with pytest.raises(FeatureStoreError, match="monetary"):
        FeatureStore(make_features().drop(columns=["monetary"]))


def test_duplicate_ids_refuse_to_start():
    df = make_features()
    df.loc[1, "customer_unique_id"] = "c0"
    with pytest.raises(FeatureStoreError, match="Duplicate"):
        FeatureStore(df)


def test_from_parquet_missing_file(tmp_path):
    with pytest.raises(FeatureStoreError, match="not found"):
        FeatureStore.from_parquet(tmp_path / "missing.parquet")
