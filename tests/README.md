# Tests

```bash
make test               # everything; integration tests skip if PostgreSQL is down
make test-unit          # unit tests only, no external dependency
make test-integration   # rfm.sql against PostgreSQL (make up first)
make coverage           # with coverage report, fails under 70%
```

## Layout

| File | Tests | Scope |
|------|------:|-------|
| `unit/test_build.py` | 12 | Feature transformations, imputation, strict input/output schemas |
| `unit/test_sql_features.py` | 10 | Data-quality rules of `validate_rfm`, SQL file loading |
| `unit/test_train.py` | 7 | Split, CV, metrics, end-to-end training + registration on a throw-away MLflow backend |
| `unit/test_feature_store.py` | 8 | Row validation, quarantine, startup failures, lookups |
| `unit/test_api.py` | 21 | Endpoints, error schema, request ids, structured logs, batch ordering |
| `unit/test_logging_config.py` | 4 | JSON formatter, model version resolution |
| `integration/test_rfm_sql.py` | 9 | `rfm.sql` on a hand-built PostgreSQL fixture |

## Conventions

- **Unit tests** use in-memory data, `tmp_path` and fakes — no network, no
  database, no real model.
- **Integration tests** are marked `@pytest.mark.integration`. They run in a
  randomly named schema that is dropped afterwards, so they never touch the
  Olist tables of a development database. Locally they skip when PostgreSQL is
  unreachable; in CI (`CI=true`) they fail instead.
- The SQL fixture (`integration/seed_rfm.sql`) documents every customer
  scenario and its expected label, so a failing assertion points to a precise
  business rule.
- Bugs get a regression test that fails on the old code before the fix is
  merged (see `test_split_payments_are_summed_once_per_order`).
