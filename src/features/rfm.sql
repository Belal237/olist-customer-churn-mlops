WITH

-- Cutoff : point de séparation features / label
-- On prend ref_date - churn_days comme date de coupure
reference_dates AS (
    SELECT
        MAX(order_purchase_timestamp::DATE) AS ref_dt,
        MAX(order_purchase_timestamp::DATE) - CAST(:churn_days AS INT) AS cutoff_dt
    FROM orders
    WHERE order_status = 'delivered'
),

-- Delivered orders AVANT le cutoff → sert à calculer les features
delivered_before_cutoff AS (
    SELECT
        o.order_id,
        c.customer_unique_id,
        o.order_purchase_timestamp::DATE AS order_date,
        op.payment_value
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_payments op ON o.order_id = op.order_id
    CROSS JOIN reference_dates rd
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp::DATE < rd.cutoff_dt  -- ← AVANT le cutoff
),

-- LAG sur la fenêtre features uniquement
orders_with_lag AS (
    SELECT
        customer_unique_id,
        order_id,
        order_date,
        payment_value,
        order_date - LAG(order_date) OVER (
            PARTITION BY customer_unique_id ORDER BY order_date
        ) AS days_since_previous_order
    FROM delivered_before_cutoff
),

-- RFM calculé sur la fenêtre features
customer_rfm_metrics AS (
    SELECT
        d.customer_unique_id,
        (rd.cutoff_dt - MAX(d.order_date))              AS recency_days,
        COUNT(DISTINCT d.order_id)                      AS frequency,
        ROUND(SUM(d.payment_value)::NUMERIC, 2)         AS monetary,
        (MAX(d.order_date) - MIN(d.order_date))         AS customer_lifetime_days,
        ROUND(AVG(d.days_since_previous_order)::NUMERIC, 1) AS avg_days_between_orders
    FROM orders_with_lag d
    CROSS JOIN reference_dates rd
    GROUP BY d.customer_unique_id, rd.cutoff_dt
),

-- Clients actifs APRES le cutoff → ils ne sont PAS churners
active_after_cutoff AS (
    SELECT DISTINCT c.customer_unique_id
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    CROSS JOIN reference_dates rd
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp::DATE >= rd.cutoff_dt  -- ← APRES le cutoff
),

-- Label : churner = existait avant le cutoff ET absent après
customer_churn_labels AS (
    SELECT
        r.customer_unique_id,
        r.recency_days,
        r.frequency,
        r.monetary,
        r.customer_lifetime_days,
        COALESCE(r.avg_days_between_orders, 0) AS avg_days_between_orders,
        CASE
            WHEN a.customer_unique_id IS NULL THEN 1  -- absent après cutoff → churner
            ELSE 0
        END AS churn_label
    FROM customer_rfm_metrics r
    LEFT JOIN active_after_cutoff a ON r.customer_unique_id = a.customer_unique_id
)

SELECT * FROM customer_churn_labels
WHERE frequency >= 2  -- churn only meaningful for repeat customers
ORDER BY recency_days ASC;