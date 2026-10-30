# Features

SQL and Python pipeline producing the model-ready feature table.

## Pipeline position

```
PostgreSQL ── rfm.sql ──► sql_features.py ──► rfm_features.parquet ──► build.py ──► features.parquet
               (SQL)      (quality checks)                             (contracts +
                                                                        engineering)
```

```bash
python -m src.features.sql_features   # or: make features (runs both steps)
python -m src.features.build
```

## Files

| File | Role |
|------|------|
| `rfm.sql` | RFM aggregates and churn label, on two disjoint time windows |
| `sql_features.py` | Runs `rfm.sql` with `:churn_days` as a bound parameter, checks data quality, writes `rfm_features.parquet` |
| `build.py` | Validates input rows (strict Pydantic), derives features, validates output, writes `features.parquet`; defines `MODEL_FEATURES`, shared with training and serving |

## Time windows

```
 first order                    cutoff_dt                         ref_dt
     │◄──────── feature window ───────►│◄──── label window (CHURN_DAYS) ────►│
                                       │                                    │
                         features: orders < cutoff          label = 1 if no delivered
                                                            order in [cutoff, ref]
```

`ref_dt` is the latest delivered order date (2018-08-29); with the default
`CHURN_DAYS=180`, `cutoff_dt` is 2018-03-02. Only customers with at least two
delivered orders before the cutoff are kept.

## Features

| Feature | Source | Description |
|---------|--------|-------------|
| `recency_days` | SQL | Days between the last order and the cutoff |
| `frequency` | SQL | Delivered orders in the feature window |
| `monetary` | SQL | Total paid (sum of payments, aggregated per order first) |
| `customer_lifetime_days` | SQL | Days between first and last order |
| `avg_days_between_orders` | SQL | Mean gap between consecutive orders (`LAG`) |
| `monetary_log` | Python | `log1p(monetary)` — reduces right skew |
| `avg_order_value` | Python | `monetary / frequency` |
| `days_per_order` | Python | `customer_lifetime_days / frequency` |
| `is_one_time_buyer` | Python | Kept in the table, **not** a model input: constant (0) after the repeat-customer filter |
| `churn_label` | SQL | Target: 1 = churned |

Output: `data/processed/features.parquet`, 1,632 rows × 11 columns.

## Data quality gates

Each step fails loudly instead of passing bad data downstream:

| Step | Checks |
|------|--------|
| `sql_features.validate_rfm` | one row per customer, no nulls in core columns, binary label, `frequency >= 2`, non-negative values — all failures reported together |
| `build.handle_missing_values` | only `avg_days_between_orders` may be imputed; any other null is an upstream bug |
| `build.validate_schema` | every input row against `RawCustomerFeatures` (strict types), a sample of output rows against `EngineeredFeatures` |

The query itself is covered by integration tests on a hand-built fixture
(`tests/integration/`), including regression tests for the join fan-out found
in [incident 04](../../docs/incidents/04-sql-join-fan-out.md).
