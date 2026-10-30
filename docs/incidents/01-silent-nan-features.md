# 01 — Silent NaN features

**Type**: simulated · **Severity**: SEV2 · **Status**: resolved in `3897aaa`

## Summary

Three features (`recency_days`, `monetary`, `avg_order_value`) of one customer
were overwritten with `NaN` in `features.parquet`, without touching the code.
The API kept answering `200 OK` with a plausible churn probability.

## Impact

Every corrupted customer would receive a wrong score with no error, no log and
no metric. Downstream consumers (retention campaigns, and later the support
agent) cannot tell a corrupted score from a valid one.

| Feature row | HTTP | `churn_probability` |
|-------------|------|---------------------|
| clean       | 200  | 0.98615 |
| corrupted   | 200  | 0.98770 |

## Detection

Not detected by the system. Found only by comparing responses before and after
the injection.

## Root cause

- XGBoost treats `NaN` as a *missing value* and routes it through the learned
  default branch of each tree. The model never fails on `NaN`, by design.
- The API validated the **request** (Pydantic) but not the **data it read**:
  the feature store was loaded with `pd.read_parquet` and served as-is.
- The `EngineeredFeatures` schema existed in the pipeline, but nothing
  re-checked the parquet file at serving time.

## Resolution

- `src/api/feature_store.py` validates every row at startup: each model input
  must be finite and non-negative.
- Invalid rows are **quarantined**: `/predict` returns
  `422 invalid_features` and `/batch-predict` returns `status: invalid_features`
  with a null score. The model is never called for them.
- If more than 5% of rows are invalid, the corruption is systemic and the API
  **refuses to start** rather than serving a partially broken store.
- Quarantined ids are logged at startup (`feature_store_rows_quarantined`) and
  exposed through `GET /health` (`n_quarantined`).

After the fix, the same injection gives:

```json
{"error": "invalid_features",
 "message": "Features for customer '0042…066c' failed validation; no score is served until the feature store is rebuilt.",
 "request_id": "03954bdc2f2e4585882cd637629abddc"}
```

Covered by `tests/unit/test_feature_store.py` and
`test_predict_quarantined_customer_is_never_scored`.

## Follow-ups

- P3: alert when `n_quarantined > 0` (Grafana / health dashboard).
- P3: data drift checks (Evidently) catch distribution shifts that stay within
  valid ranges, which this check cannot see.

## Lessons

Validate data at every trust boundary, not only user input. A model that
"handles" missing values makes corruption silent: the absence of errors is
not evidence of correctness.
