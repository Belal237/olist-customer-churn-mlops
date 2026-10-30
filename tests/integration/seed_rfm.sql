-- Minimal Olist-shaped fixture for rfm.sql. Only the columns read by the query
-- are created. Executed inside a throw-away schema (see conftest.py).
--
-- Reference date (latest delivered order) = 2018-08-01
-- With churn_days = 180 → cutoff = 2018-02-02
--
-- Customer scenarios:
--   u_retained   2 orders before cutoff + 1 after       → label 0
--                the 2017-12-01 order has 3 items and 2 identical payment rows
--   u_churned    3 orders before cutoff, none after     → label 1
--                the 2017-07-01 order has 2 payment rows of different amounts
--   u_multi_id   2 orders under 2 different customer_id → aggregated, label 1
--   u_one_time   1 order before cutoff                  → excluded (frequency < 2)
--   u_canceled   1 delivered + 1 canceled order         → excluded (frequency < 2)
--   u_recent     1 order after cutoff only              → excluded (no feature window)

CREATE TABLE customers (
    customer_id        TEXT PRIMARY KEY,
    customer_unique_id TEXT NOT NULL
);

CREATE TABLE orders (
    order_id                 TEXT PRIMARY KEY,
    customer_id              TEXT NOT NULL,
    order_status             TEXT NOT NULL,
    order_purchase_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE order_items (
    order_id      TEXT NOT NULL,
    order_item_id INTEGER NOT NULL
);

CREATE TABLE order_payments (
    order_id           TEXT NOT NULL,
    payment_sequential INTEGER NOT NULL,
    payment_value      NUMERIC(10, 2) NOT NULL
);

INSERT INTO customers VALUES
    ('c_retained',  'u_retained'),
    ('c_churned',   'u_churned'),
    ('c_multi_1',   'u_multi_id'),
    ('c_multi_2',   'u_multi_id'),
    ('c_one_time',  'u_one_time'),
    ('c_canceled',  'u_canceled'),
    ('c_recent',    'u_recent');

INSERT INTO orders VALUES
    ('o_r1', 'c_retained', 'delivered', '2017-10-01 10:00:00'),
    ('o_r2', 'c_retained', 'delivered', '2017-12-01 10:00:00'),
    ('o_r3', 'c_retained', 'delivered', '2018-05-01 10:00:00'),
    ('o_c1', 'c_churned',  'delivered', '2017-06-01 10:00:00'),
    ('o_c2', 'c_churned',  'delivered', '2017-07-01 10:00:00'),
    ('o_c3', 'c_churned',  'delivered', '2017-09-01 10:00:00'),
    ('o_m1', 'c_multi_1',  'delivered', '2017-03-01 10:00:00'),
    ('o_m2', 'c_multi_2',  'delivered', '2017-04-01 10:00:00'),
    ('o_o1', 'c_one_time', 'delivered', '2017-11-15 10:00:00'),
    ('o_x1', 'c_canceled', 'delivered', '2017-08-01 10:00:00'),
    ('o_x2', 'c_canceled', 'canceled',  '2017-09-01 10:00:00'),
    ('o_n1', 'c_recent',   'delivered', '2018-08-01 10:00:00');

INSERT INTO order_items VALUES
    ('o_r1', 1),
    ('o_r2', 1), ('o_r2', 2), ('o_r2', 3),
    ('o_r3', 1),
    ('o_c1', 1), ('o_c2', 1), ('o_c3', 1),
    ('o_m1', 1), ('o_m2', 1),
    ('o_o1', 1),
    ('o_x1', 1), ('o_x2', 1),
    ('o_n1', 1);

INSERT INTO order_payments VALUES
    ('o_r1', 1,  80.00),
    ('o_r2', 1,  50.00), ('o_r2', 2, 50.00),
    ('o_r3', 1,  70.00),
    ('o_c1', 1,  40.00),
    ('o_c2', 1,  30.00), ('o_c2', 2, 20.00),
    ('o_c3', 1,  60.00),
    ('o_m1', 1,  25.00),
    ('o_m2', 1,  35.00),
    ('o_o1', 1,  99.00),
    ('o_x1', 1,  10.00),
    ('o_x2', 1,  10.00),
    ('o_n1', 1, 500.00);
