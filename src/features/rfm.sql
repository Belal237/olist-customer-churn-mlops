-- RFM features + churn label for Olist repeat customers.
--
-- Two disjoint time windows prevent label leakage:
--   * features : delivered orders strictly BEFORE cutoff_dt
--   * label    : presence/absence of a delivered order in [cutoff_dt, ref_dt]
-- where ref_dt is the latest delivered order date and
-- cutoff_dt = ref_dt - :churn_days.
--
-- Grain rule: every CTE below is one row per order (or per customer).
-- order_payments has several rows per order (vouchers, split payments), so it
-- is aggregated BEFORE joining. Joining it raw duplicates orders and skews
-- LAG()-based gaps — see docs/incidents/04-sql-join-fan-out.md.

WITH

reference_dates AS (
    SELECT
        MAX(order_purchase_timestamp::DATE)                             AS ref_dt,
        MAX(order_purchase_timestamp::DATE) - CAST(:churn_days AS INT)  AS cutoff_dt
    FROM orders
    WHERE order_status = 'delivered'
),

-- One row per order: total amount actually paid
order_payment_totals AS (
    SELECT
        order_id,
        SUM(payment_value) AS order_value
    FROM order_payments
    GROUP BY order_id
),

-- One row per delivered order placed BEFORE the cutoff (feature window)
delivered_before_cutoff AS (
    SELECT
        o.order_id,
        c.customer_unique_id,
        o.order_purchase_timestamp::DATE AS order_date,
        p.order_value
    FROM orders o
    JOIN customers c             ON o.customer_id = c.customer_id
    JOIN order_payment_totals p  ON o.order_id    = p.order_id
    CROSS JOIN reference_dates rd
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp::DATE < rd.cutoff_dt
),

-- Gap with the previous order of the same customer (order_id breaks same-day ties)
orders_with_lag AS (
    SELECT
        customer_unique_id,
        order_id,
        order_date,
        order_value,
        order_date - LAG(order_date) OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_date, order_id
        ) AS days_since_previous_order
    FROM delivered_before_cutoff
),

-- RFM aggregates on the feature window
customer_rfm_metrics AS (
    SELECT
        d.customer_unique_id,
        rd.cutoff_dt - MAX(d.order_date)                       AS recency_days,
        COUNT(*)                                               AS frequency,
        ROUND(SUM(d.order_value)::NUMERIC, 2)                  AS monetary,
        MAX(d.order_date) - MIN(d.order_date)                  AS customer_lifetime_days,
        ROUND(AVG(d.days_since_previous_order)::NUMERIC, 1)    AS avg_days_between_orders
    FROM orders_with_lag d
    CROSS JOIN reference_dates rd
    GROUP BY d.customer_unique_id, rd.cutoff_dt
),

-- Customers with at least one delivered order in the label window
active_after_cutoff AS (
    SELECT DISTINCT c.customer_unique_id
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    CROSS JOIN reference_dates rd
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp::DATE >= rd.cutoff_dt
),

-- Label: churned = active before the cutoff AND absent from the label window
customer_churn_labels AS (
    SELECT
        r.customer_unique_id,
        r.recency_days,
        r.frequency,
        r.monetary,
        r.customer_lifetime_days,
        COALESCE(r.avg_days_between_orders, 0) AS avg_days_between_orders,
        CASE WHEN a.customer_unique_id IS NULL THEN 1 ELSE 0 END AS churn_label
    FROM customer_rfm_metrics r
    LEFT JOIN active_after_cutoff a ON r.customer_unique_id = a.customer_unique_id
)

SELECT *
FROM customer_churn_labels
WHERE frequency >= 2  -- churn is only meaningful for repeat customers
ORDER BY customer_unique_id;
