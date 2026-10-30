# Incident post-mortems

Failure scenarios deliberately injected into the system at the end of Phase 1,
then diagnosed and fixed. Each post-mortem follows the same blameless format:
summary, impact, detection, root cause, resolution, follow-ups.

| # | Incident | Layer | Fix |
|---|----------|-------|-----|
| 01 | [Silent NaN features](01-silent-nan-features.md) | API / data quality | Feature store validation + quarantine |
| 02 | [Unexplained API latency](02-unexplained-latency.md) | API / observability | Request ids, stage timings, vectorized batch |
| 03 | [Unreadable validation errors and plain-text 500s](03-validation-errors.md) | API / contracts | Unified error schema, strict payloads |
| 04 | [SQL join fan-out](04-sql-join-fan-out.md) | Feature pipeline | Pre-aggregated payments + regression tests |

Incidents 01 to 03 were reproduced against the real model and feature store
with FastAPI's `TestClient`; incident 04 was found while validating
intermediate row counts in PostgreSQL. All numbers below are measured, not
estimated.

## Template

```markdown
# NN — Title
**Type**: simulated | real · **Severity**: SEV1–SEV3 · **Status**: resolved

## Summary
## Impact
## Detection
## Root cause
## Resolution
## Follow-ups
## Lessons
```
