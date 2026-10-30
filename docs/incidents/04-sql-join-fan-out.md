# 04 — SQL join fan-out in rfm.sql

**Type**: real (found during the P1 validation pass) · **Severity**: SEV2 ·
**Status**: resolved in `f976876`

## Summary

`rfm.sql` joined `orders`, `order_items` and `order_payments` at different
grains, then relied on `SELECT DISTINCT` to remove the duplicates. The query
ran without errors and passed every existing check, but two features were
wrong for part of the customers.

## Impact

Measured on the full Olist dataset with `churn_days = 180` (1,632 repeat
customers):

| Feature | Customers affected | Effect |
|---------|--------------------|--------|
| `avg_days_between_orders` | 76 (4.7%) | biased low (spurious 0-day gaps) — mean error +2.2 days once fixed |
| `monetary` | 10 | understated (identical payment rows collapsed) |

`frequency` was unaffected because it used `COUNT(DISTINCT order_id)`, which
is precisely why the issue stayed hidden: the most visible metric looked right.

## Detection

Comparing intermediate row counts while reviewing the query:

```sql
SELECT COUNT(*) AS joined_rows, COUNT(DISTINCT o.order_id) AS orders
FROM orders o
JOIN order_items oi    ON o.order_id = oi.order_id
JOIN order_payments op ON o.order_id = op.order_id
WHERE o.order_status = 'delivered';
-- joined_rows = 115,035   orders = 96,477
```

Then counting the patterns `DISTINCT` could not handle:

- 2,961 orders have several payment rows (vouchers, split payments): with
  different amounts, `DISTINCT` keeps one row per payment.
- 308 orders have several **identical** payment rows: `DISTINCT` collapses them
  into one, dropping money.

## Root cause

`delivered_before_cutoff` mixed three grains — order, item, payment — in one
`SELECT DISTINCT o.order_id, …, op.payment_value`. Downstream:

- `LAG(order_date)` ran over one row per *payment*, so a 2-payment order added
  a 0-day gap to the customer's average.
- `SUM(payment_value)` ran over *distinct* payment values, losing duplicates.

The `order_items` join added nothing (no item column was used) but multiplied
rows further.

## Resolution

- New `order_payment_totals` CTE: payments are summed **per order before the
  join**, so every downstream CTE is strictly one row per order.
- `order_items` join removed.
- `LAG` ordered by `(order_date, order_id)` so same-day orders are
  deterministic.
- Grain rule documented at the top of `rfm.sql`.

Regression tests (`tests/integration/test_rfm_sql.py`) run the real query on a
hand-built fixture with split payments, identical payment rows and multi-item
orders. Both regression tests **fail on the previous query** and pass on the
fixed one; CI runs them against a PostgreSQL service on every push.

## Follow-ups

- P2: add a row-count assertion per CTE in the orchestrated pipeline
  (orders in = orders out).

## Lessons

A query that runs is not a query that is correct. Every join should state its
grain, and `DISTINCT` used to "fix" duplicates is a smell: it usually hides a
join at the wrong grain. Validate with counts at each step.
