"""
Integration tests for src/features/rfm.sql on a hand-built fixture
(tests/integration/seed_rfm.sql) where every expected value is known.
"""

import pandas as pd
import pytest

from src.features.build import RawCustomerFeatures
from src.features.sql_features import build_rfm_dataset, compute_rfm, validate_rfm

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def rfm(pg_engine) -> pd.DataFrame:
    return compute_rfm(pg_engine, churn_days=180).set_index("customer_unique_id")


def test_only_repeat_customers_with_feature_history_are_kept(rfm):
    assert sorted(rfm.index) == ["u_churned", "u_multi_id", "u_retained"]


def test_churn_labels(rfm):
    assert rfm.loc["u_retained", "churn_label"] == 0
    assert rfm.loc["u_churned", "churn_label"] == 1
    assert rfm.loc["u_multi_id", "churn_label"] == 1


def test_label_window_orders_do_not_leak_into_features(rfm):
    # u_retained's 2018-05-01 order (after cutoff) must not count
    assert rfm.loc["u_retained", "frequency"] == 2
    # recency is measured from the cutoff (2018-02-02) to the last order (2017-12-01)
    assert rfm.loc["u_retained", "recency_days"] == 63


def test_split_payments_are_summed_once_per_order(rfm):
    # Regression test for incident 04: 2 identical payment rows of 50.00 on one
    # order were collapsed by SELECT DISTINCT, and 3 items multiplied rows.
    assert rfm.loc["u_retained", "monetary"] == pytest.approx(80.0 + 50.0 + 50.0)
    assert rfm.loc["u_churned", "monetary"] == pytest.approx(40.0 + 30.0 + 20.0 + 60.0)


def test_gap_between_orders_ignores_payment_rows(rfm):
    # Orders on 2017-06-01, 2017-07-01, 2017-09-01 → gaps of 30 and 62 days.
    # Before the fix, the 2-payment order produced an extra 0-day gap.
    assert rfm.loc["u_churned", "avg_days_between_orders"] == pytest.approx(46.0)
    assert rfm.loc["u_churned", "customer_lifetime_days"] == 92


def test_customer_ids_are_aggregated_by_unique_id(rfm):
    assert rfm.loc["u_multi_id", "frequency"] == 2
    assert rfm.loc["u_multi_id", "monetary"] == pytest.approx(60.0)


def test_churn_window_is_a_bound_parameter(pg_engine):
    # With a 60-day window the cutoff moves to 2018-06-02: u_retained's
    # 2018-05-01 order now falls in the feature window and no one is active after.
    df = compute_rfm(pg_engine, churn_days=60).set_index("customer_unique_id")
    assert df.loc["u_retained", "frequency"] == 3
    assert df["churn_label"].eq(1).all()


def test_output_matches_downstream_contract(rfm):
    validate_rfm(rfm.reset_index())
    for record in rfm.reset_index().to_dict("records"):
        RawCustomerFeatures(**record)


def test_build_rfm_dataset_writes_parquet(pg_engine, tmp_path):
    output = tmp_path / "rfm_features.parquet"
    df = build_rfm_dataset(output_path=output, engine=pg_engine, churn_days=180)
    assert output.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(output), df)
