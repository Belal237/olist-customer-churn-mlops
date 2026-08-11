-- Minimal schema and seed data for CI integration test
BEGIN;

CREATE TABLE IF NOT EXISTS customers (
  customer_id INTEGER PRIMARY KEY,
  customer_unique_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  order_purchase_timestamp TIMESTAMP NOT NULL,
  order_status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
  order_item_id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS order_payments (
  payment_id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL,
  payment_value NUMERIC NOT NULL
);

-- Insert two customers
INSERT INTO customers (customer_id, customer_unique_id) VALUES (1, 'cust_001') ON CONFLICT DO NOTHING;
INSERT INTO customers (customer_id, customer_unique_id) VALUES (2, 'cust_002') ON CONFLICT DO NOTHING;

-- Use fixed dates: one order before cutoff, one after cutoff
-- Max order date will be the later date below
INSERT INTO orders (order_id, customer_id, order_purchase_timestamp, order_status) VALUES
  (10, 1, current_date - INTERVAL '200 days', 'delivered') ON CONFLICT DO NOTHING,
  (11, 2, current_date - INTERVAL '10 days', 'delivered') ON CONFLICT DO NOTHING;

INSERT INTO order_items (order_item_id, order_id) VALUES (100, 10) ON CONFLICT DO NOTHING;
INSERT INTO order_items (order_item_id, order_id) VALUES (101, 11) ON CONFLICT DO NOTHING;

INSERT INTO order_payments (payment_id, order_id, payment_value) VALUES (1000, 10, 100.0) ON CONFLICT DO NOTHING;
INSERT INTO order_payments (payment_id, order_id, payment_value) VALUES (1001, 11, 200.0) ON CONFLICT DO NOTHING;

COMMIT;
