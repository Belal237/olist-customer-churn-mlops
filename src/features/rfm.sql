-- ============================================================
-- RFM Features + Churn Label
-- Dataset  : Olist Brazilian E-Commerce
-- Scope    : 'delivered' orders only
-- Ref date : MAX(order_purchase_timestamp) — never use NOW()
-- Churn    : no purchase in the last :churn_days before ref date
-- Monetary : payment_value (actual amount paid, includes discounts)
-- ============================================================

WITH

-- Step 1: Fixed reference date
-- Using MAX() ensures full reproducibility — the label never changes
-- depending on when the pipeline runs.
reference_date AS (
    SELECT MAX(order_purchase_timestamp::DATE) AS ref_dt
    FROM orders
    WHERE order_status = 'delivered'
),

-- Step 2: Delivered orders with actual payment value
-- order_payments.payment_value is used instead of price + freight_value
-- because it reflects what the customer actually paid (after discounts/vouchers).
delivered_orders AS (
    SELECT
        o.order_id,
        c.customer_unique_id,
        o.order_purchase_timestamp::DATE AS order_date,
        op.payment_value
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    JOIN order_payments op
        ON o.order_id = op.order_id
    WHERE o.order_status = 'delivered'
),

-- Step 3: Days between consecutive orders per customer (LAG)
-- Window functions cannot be nested inside aggregates —
-- they must be computed in a separate CTE first.
orders_with_lag AS (
    SELECT
        customer_unique_id,
        order_id,
        order_date,
        payment_value,
        (
            order_date
            - LAG(order_date) OVER (
                PARTITION BY customer_unique_id
                ORDER BY order_date
            )
        ) AS days_since_previous_order
    FROM delivered_orders
),

-- Step 4: RFM aggregates per unique customer
customer_rfm_metrics AS (
    SELECT
        d.customer_unique_id,

        -- Recency: days since last purchase
        (rd.ref_dt - MAX(d.order_date))         AS recency_days,

        -- Frequency: number of distinct orders
        COUNT(DISTINCT d.order_id)              AS frequency,

        -- Monetary: total actual spend
        ROUND(SUM(d.payment_value)::NUMERIC, 2) AS monetary,

        -- Customer lifetime: span between first and last order
        (MAX(d.order_date) - MIN(d.order_date)) AS customer_lifetime_days,

        -- Avg gap between orders (NULL for one-time buyers → filled in Step 5)
        ROUND(AVG(d.days_since_previous_order)::NUMERIC, 1)
                                                AS avg_days_between_orders

    FROM orders_with_lag d
    CROSS JOIN reference_date rd
    GROUP BY d.customer_unique_id
),

-- Step 5: Churn label + NULL handling
-- :churn_days is a bound parameter passed from Python — never hardcoded.
-- >= is used (not >) so that exactly :churn_days days counts as churned.
-- avg_days_between_orders is 0 for one-time buyers (no gap to compute).
customer_churn_labels AS (
    SELECT
        customer_unique_id,
        recency_days,
        frequency,
        monetary,
        customer_lifetime_days,
        COALESCE(avg_days_between_orders, 0) AS avg_days_between_orders,
        CASE
            WHEN recency_days >= :churn_days THEN 1
            ELSE 0
        END                                  AS churn_label
    FROM customer_rfm_metrics
)

SELECT *
FROM customer_churn_labels
ORDER BY recency_days ASC;