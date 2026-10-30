# 02 — Unexplained API latency

**Type**: simulated · **Severity**: SEV3 · **Status**: resolved in `3897aaa`

## Summary

A random `time.sleep(0.15–0.30 s)` was injected into 10% of model calls.
`/predict` p95 latency jumped from **0.84 ms to 206 ms**, but the logs only
contained the total `latency_ms`: there was no way to tell which step was slow
or to follow one request across log lines.

## Impact

- p95 latency ×245 for affected requests.
- Mean time to diagnose: unbounded. Nothing in the logs pointed to the
  inference step rather than the feature lookup, serialization or the network.

## Detection

Latency was visible in the response body and the `predict` log line, but only
as a single number per request. Log lines were prefixed with a plain-text
timestamp (`2026-… INFO {json}`), so they could not be parsed as JSON by a log
aggregator either.

## Root cause

Missing instrumentation: no per-stage timings, no request id, no access log.

Profiling the request path while investigating also showed two real
inefficiencies, independent of the injected delay:

| Finding | Measurement |
|---------|-------------|
| `/predict` called both `predict()` and `predict_proba()` | 2 × ~0.40 ms per request |
| `/batch-predict` called the model once per customer | 1000 ids → **466 ms** |
| `pandas` boolean filter vs. `.loc` on an index (1.6k rows) | 0.12 ms vs. 0.21 ms — the index was *not* faster at this size |

## Resolution

- **JSON logs**: one object per line with `timestamp`, `level`, `logger`,
  `request_id` and event fields (`src/logging_config.py`).
- **Request id**: `X-Request-ID` is accepted or generated, returned on every
  response and attached to every log line of the request.
- **Access log**: one `http_request` event per call with method, path, status
  and duration.
- **Stage timings**: `predict` / `batch_predict` events log `lookup_ms`,
  `inference_ms` and `latency_ms`.
- **Performance**: features served from a numpy matrix + dict index;
  churn prediction derived from the probability (one model call);
  batch scored in a single vectorized call.

With the same injected delay, the slow stage is now obvious in a single line:

```json
{"event": "predict", "request_id": "410780d4…", "lookup_ms": 0.044,
 "inference_ms": 156.985, "latency_ms": 157.029, "model_version": "churn-prediction/v3"}
```

| Metric (real model, 200 requests) | Before | After |
|-----------------------------------|--------|-------|
| `/predict` p50 | 0.65 ms | 0.46 ms |
| `/predict` p95 | 0.84 ms | 0.57 ms |
| `/batch-predict`, 1000 ids | 466 ms | 2.4 ms |

## Follow-ups

- P2: export latency histograms (p50/p95/p99) to the monitoring stack instead
  of computing them from logs.
- P4: distributed tracing (OpenTelemetry / LangSmith) once the agent calls
  several services.

## Lessons

Instrument before you need it. A single total latency tells you *that*
something is slow, never *where*. And measure before optimizing: the
"obvious" fix (a pandas index) was slower than the code it replaced.
