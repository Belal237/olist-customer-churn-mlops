# 03 — Unreadable validation errors and plain-text 500s

**Type**: simulated · **Severity**: SEV3 · **Status**: resolved in `3897aaa`

## Summary

Malformed requests and an internal failure were sent to the API to see what an
on-call engineer — or a client developer — would get back at 2 a.m.

| Request | Response before the fix |
|---------|-------------------------|
| `{"customer_id": 123}` | `422 {"detail":[{"type":"string_type","loc":["body","customer_id"],"msg":"Input should be a valid string","input":123}]}` |
| `{"customerId": "…"}` (typo) | `422 … "loc":["body","customer_id"],"msg":"Field required" …` — the typo itself is never mentioned |
| `{bad json` | `422 … "loc":["body",1],"msg":"JSON decode error" …` |
| Model raises `ValueError` | `500 text/plain "Internal Server Error"` |

## Impact

- Errors were technically correct but written for the framework, not for a
  human: nested `loc` arrays, no summary, no hint about the actual typo.
- The 500 was plain text, with no JSON body and no identifier: impossible to
  correlate a client report with the server logs.

## Root cause

- FastAPI's default `RequestValidationError` handler returns raw Pydantic
  errors.
- Unknown fields were silently ignored (Pydantic default `extra="ignore"`),
  so a misspelt field surfaced as a confusing "Field required".
- No handler for unexpected exceptions, and no request id to quote.

## Resolution

- **One error schema** for every non-2xx response (`ErrorResponse`):
  `error` (stable machine code), `message` (one-line human summary),
  `details` (per-field `field` / `issue` / `received`) and `request_id`.
- **Strict payloads**: `extra="forbid"` on request models, so typos are
  reported as `Unknown field`.
- **Unhandled exceptions** are caught in the request middleware, logged with
  their traceback *and* request id, and returned as a JSON 500 that does not
  leak internal messages.

```json
{"error": "validation_error",
 "message": "Invalid request — customer_id: Field required; customerId: Unknown field",
 "request_id": "7b97d2e51d764f0c8d9d88569e794117",
 "details": [
   {"field": "customer_id", "issue": "Field required"},
   {"field": "customerId", "issue": "Unknown field", "received": "0042…066c"}]}
```

Covered by `test_predict_invalid_payload_returns_readable_422`,
`test_malformed_json_returns_readable_422` and
`test_unhandled_error_returns_json_500_with_request_id`.

## Follow-ups

- P2: count `request_rejected` events per field to spot misbehaving clients.

## Lessons

An error message is part of the API contract. It should say what is wrong,
where, and give an id that links the client's view to the server's logs.
