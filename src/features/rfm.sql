-- ============================================================
-- RFM Features + Churn Label
-- Dataset  : Olist Brazilian E-Commerce
-- Scope    : 'delivered' orders only
-- Ref date : MAX(order_purchase_timestamp) = 2018-08-29
-- Churn    : no purchase in the last 90 days before ref date
-- ============================================================

WITH
-- Step 1: Fixed reference date — never use NOW()
reference_date AS (
    SELECT MAX(order_purchase_timestamp) AS ref_date
    FROM orders
    WHERE order_status = 'delivered'
),

-- Step 2: Delivered orders with total value per order
delivered_orders AS (
    SELECT
        o.order_id,
        c.customer_unique_id,
        o.order_purchase_timestamp,
        SUM(oi.price + oi.freight_value) AS order_value
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    JOIN order_items oi
        ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY
        o.order_id,
        c.customer_unique_id,
        o.order_purchase_timestamp
),

-- Step 3: Compute days between consecutive orders per customer
-- Window functions must be isolated — cannot be nested inside aggregates
orders_with_lag AS (
    SELECT
        customer_unique_id,
        order_id,
        order_purchase_timestamp,
        order_value,
        EXTRACT(DAY FROM (
            order_purchase_timestamp - LAG(order_purchase_timestamp)
            OVER (
                PARTITION BY customer_unique_id
                ORDER BY order_purchase_timestamp
            )
        )) AS days_since_previous_order
    FROM delivered_orders
),

-- Step 4: RFM aggregates per unique customer
customer_rfm_metrics AS (
    SELECT
        d.customer_unique_id,
        r.ref_date,

        -- Recency: days since last purchase
        CAST(EXTRACT(DAY FROM (
            r.ref_date - MAX(d.order_purchase_timestamp)
        )) AS INTEGER) AS recency_days,

        -- Frequency: number of distinct orders
        COUNT(DISTINCT d.order_id) AS frequency,

        -- Monetary: total spend
        ROUND(CAST(SUM(d.order_value) AS NUMERIC), 2) AS monetary,

        -- Avg days between orders — now safe to aggregate
        ROUND(CAST(AVG(d.days_since_previous_order) AS NUMERIC), 1)
            AS avg_days_between_orders,

        MAX(d.order_purchase_timestamp) AS last_order_date

    FROM orders_with_lag d
    CROSS JOIN reference_date r
    GROUP BY d.customer_unique_id, r.ref_date
),

-- Step 5: Add churn label + handle NULLs
customer_churn_labels AS (
    SELECT
        customer_unique_id,
        recency_days,
        frequency,
        monetary,
        COALESCE(avg_days_between_orders, 0) AS avg_days_between_orders,
        last_order_date,
        ref_date,
        CASE WHEN recency_days > 90 THEN 1 ELSE 0 END AS churn_label
    FROM customer_rfm_metrics
)

SELECT *
FROM customer_churn_labels
ORDER BY recency_days ASC;