"""
Unit tests for the database-free parts of src/features/sql_features.py.
The query itself is covered by tests/integration/test_rfm_sql.py.
"""

import pandas as pd
import pytest

from src.features.sql_features import (
    SQL_PATH,
    RFMValidationError,
    load_query,
    validate_rfm,
)


@pytest.fixture
def valid_rfm() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_unique_id": ["a", "b", "c"],
            "recency_days": [10, 0, 300],
            "frequency": [2, 3, 5],
            "monetary": [100.0, 250.5, 0.0],
            "customer_lifetime_days": [30, 0, 400],
            "avg_days_between_orders": [30.0, 0.0, 100.0],
            "churn_label": [0, 1, 1],
        }
    )


def test_load_query_reads_bound_parameter():
    query = load_query(SQL_PATH)
    assert ":churn_days" in query


def test_load_query_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_query(tmp_path / "missing.sql")


def test_validate_rfm_accepts_valid_table(valid_rfm):
    validate_rfm(valid_rfm)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("customer_unique_id", "a", "duplicate"),
        ("monetary", None, "null values in 'monetary'"),
        ("churn_label", 2, "churn_label"),
        ("frequency", 1, "frequency must be >= 2"),
        ("monetary", -5.0, "monetary must be >= 0"),
        ("recency_days", -1, "recency_days must be >= 0"),
    ],
)
def test_validate_rfm_rejects_bad_rows(valid_rfm, column, value, message):
    bad = valid_rfm.copy()
    bad.loc[1, column] = value
    with pytest.raises(RFMValidationError, match=message):
        validate_rfm(bad)


def test_validate_rfm_reports_every_failure(valid_rfm):
    bad = valid_rfm.copy()
    bad.loc[0, "monetary"] = -1.0
    bad.loc[0, "recency_days"] = -1
    with pytest.raises(RFMValidationError) as exc_info:
        validate_rfm(bad)
    assert "monetary" in str(exc_info.value)
    assert "recency_days" in str(exc_info.value)
