"""
Unit tests for src/features/build.py
Each test is independent and uses only in-memory DataFrames — no DB, no disk.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    add_avg_order_value,
    add_days_per_order,
    add_monetary_log,
    add_one_time_buyer_flag,
    handle_missing_values,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def base_df():
    """Minimal valid DataFrame that mirrors rfm_features.parquet schema."""
    return pd.DataFrame({
        "customer_unique_id":       ["c1", "c2", "c3"],
        "recency_days":             [10, 200, 50],
        "frequency":                [3, 1, 5],
        "monetary":                 [150.0, 0.0, 400.0],
        "customer_lifetime_days":   [90, 0, 300],
        "avg_days_between_orders":  [45.0, None, 60.0],
        "churn_label":              [0, 1, 0],
    })


# ── Test: monetary_log ─────────────────────────────────────────────────────────

def test_add_monetary_log_values(base_df):
    """log1p(0) = 0, log1p(150) > 0 — verifies the transform is applied correctly."""
    result = add_monetary_log(base_df)
    assert "monetary_log" in result.columns
    assert result.loc[result["monetary"] == 0.0, "monetary_log"].iloc[0] == 0.0
    assert (result["monetary_log"] >= 0).all()


def test_add_monetary_log_no_mutation(base_df):
    """Pure function — original DataFrame must not be modified."""
    original_cols = list(base_df.columns)
    add_monetary_log(base_df)
    assert list(base_df.columns) == original_cols


# ── Test: avg_order_value ──────────────────────────────────────────────────────

def test_add_avg_order_value_correct(base_df):
    """monetary / frequency must match for all rows."""
    result = add_avg_order_value(base_df)
    expected = base_df["monetary"] / base_df["frequency"]
    pd.testing.assert_series_equal(
        result["avg_order_value"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# ── Test: days_per_order ───────────────────────────────────────────────────────

def test_days_per_order_one_time_buyer(base_df):
    """One-time buyers (frequency=1) must get days_per_order=0, not a division error."""
    result = add_days_per_order(base_df)
    one_time = result[result["frequency"] == 1]
    assert (one_time["days_per_order"] == 0.0).all()


def test_days_per_order_multi_buyer(base_df):
    """Multi-order buyers: customer_lifetime_days / frequency."""
    result = add_days_per_order(base_df)
    multi = result[result["frequency"] > 1]
    expected = multi["customer_lifetime_days"] / multi["frequency"]
    pd.testing.assert_series_equal(
        multi["days_per_order"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# ── Test: one_time_buyer flag ──────────────────────────────────────────────────

def test_one_time_buyer_flag(base_df):
    """is_one_time_buyer must be 1 only where frequency == 1."""
    result = add_one_time_buyer_flag(base_df)
    assert result.loc[result["frequency"] == 1, "is_one_time_buyer"].iloc[0] == 1
    assert (result.loc[result["frequency"] > 1, "is_one_time_buyer"] == 0).all()


# ── Test: handle_missing_values ────────────────────────────────────────────────

def test_handle_missing_fills_avg_days(base_df):
    """avg_days_between_orders NaN (one-time buyer) must be filled with 0."""
    result = handle_missing_values(base_df)
    assert result["avg_days_between_orders"].isna().sum() == 0
    assert result.loc[result["frequency"] == 1, "avg_days_between_orders"].iloc[0] == 0.0


def test_handle_missing_raises_on_critical_null(base_df):
    """Unexpected NaN in a critical column must raise ValueError, not silently pass."""
    bad_df = base_df.copy()
    bad_df.loc[0, "monetary"] = None
    with pytest.raises(ValueError, match="monetary"):
        handle_missing_values(bad_df)


def test_engineered_features_rejects_negative(base_df):
    """Le model_validator doit rejeter une valeur négative dans les features."""
    from src.features.build import EngineeredFeatures
    with pytest.raises(Exception):
        EngineeredFeatures(
            customer_unique_id="c1",
            recency_days=-1,        # valeur négative — doit être rejetée
            frequency=3,
            monetary=150.0,
            monetary_log=5.01,
            avg_order_value=50.0,
            days_per_order=30.0,
            avg_days_between_orders=45.0,
            customer_lifetime_days=90,
            is_one_time_buyer=0,
            churn_label=0,
        )


def test_validate_schema_passes_on_valid_df(base_df):
    """validate_schema ne doit pas lever d'erreur sur un DataFrame valide et complet."""
    from src.features.build import validate_schema
    import numpy as np

    df = base_df.copy()
    df = df.assign(
        monetary_log=np.log1p(df["monetary"]),
        avg_order_value=df["monetary"] / df["frequency"],
        days_per_order=np.where(df["frequency"] > 1, df["customer_lifetime_days"] / df["frequency"], 0.0),
        is_one_time_buyer=(df["frequency"] == 1).astype(int),
        avg_days_between_orders=df["avg_days_between_orders"].fillna(0.0),
    )
    # Ne doit pas lever d'exception
    validate_schema(df, sample_size=3)


def test_build_features_output_shape(tmp_path):
    """build_features doit lire un parquet, transformer, et sauvegarder le résultat."""
    import numpy as np
    from src.features.build import build_features, FEATURE_COLUMNS

    # Créer un rfm_features.parquet minimal en entrée
    input_df = pd.DataFrame({
        "customer_unique_id":       ["c1", "c2", "c3"],
        "recency_days":             [10, 200, 50],
        "frequency":                [3, 1, 5],
        "monetary":                 [150.0, 0.0, 400.0],
        "customer_lifetime_days":   [90, 0, 300],
        "avg_days_between_orders":  [45.0, 0.0, 60.0],
        "churn_label":              [0, 1, 0],
    })
    input_path = tmp_path / "rfm_features.parquet"
    output_path = tmp_path / "features.parquet"
    input_df.to_parquet(input_path, index=False)

    result = build_features(input_path=input_path, output_path=output_path)

    # Les colonnes finales doivent correspondre exactement à FEATURE_COLUMNS
    assert list(result.columns) == FEATURE_COLUMNS
    # Le fichier de sortie doit exister sur le disque
    assert output_path.exists()
    # Aucune ligne ne doit être perdue
    assert len(result) == 3