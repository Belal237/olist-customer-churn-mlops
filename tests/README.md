## Tests

Pytest test suite.

Run the test suite:

```bash
pytest -v
```

Run with coverage report:

```bash
pytest -v --cov=src/features --cov-report=term-missing
```


## Current coverage

| Module | Coverage |
|--------|----------|
| `src/features/build.py` | 92% |
| `src/data/sql_features.py` | — (requires PostgreSQL) |

## Structure
tests/
├── init.py
└── unit/
├── init.py
└── test_build.py    # 11 tests — feature engineering pipeline


## What is tested

- `add_monetary_log` — log1p transform on monetary column
- `add_avg_order_value` — monetary / frequency
- `add_days_per_order` — customer_lifetime_days / frequency, 0 for one-time buyers
- `add_one_time_buyer_flag` — binary flag for frequency == 1
- `handle_missing_values` — NaN imputation and fail-fast on critical columns
- Pydantic schema validation — EngineeredFeatures rejects invalid rows

## Target coverage (end of P1)

70%+ on `src/features/` and `src/api/` — measured with pytest-cov.
