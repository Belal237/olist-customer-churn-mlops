# Data

Loads the raw Olist CSV files into PostgreSQL.

## Pipeline position

```
data/raw/*.csv  →  load.py  →  PostgreSQL (olist_db)  →  src/features/
```

## Setup

```bash
# Download the dataset (requires a Kaggle API token in ~/.kaggle/kaggle.json)
kaggle datasets download olistbr/brazilian-ecommerce --unzip -p data/raw

# Start PostgreSQL, then load every table
make up
python -m src.data.load
```

`load.py` drops and recreates each table, bulk-loads it with `COPY` (the
geolocation table has 1M rows) and creates indexes on the join keys used by
`rfm.sql`. Running it again is safe and gives the same result.

## Tables

| Table | Rows | Description |
|-------|-----:|-------------|
| `orders` | 99,441 | Order status and timestamps |
| `customers` | 99,441 | One row per (order-level) `customer_id`, linked to a `customer_unique_id` |
| `order_items` | 112,650 | Products and sellers per order |
| `order_payments` | 103,886 | One row per payment — an order can have several |
| `order_reviews` | 99,224 | Review scores and comments |
| `products` | 32,951 | Product categories and dimensions |
| `sellers` | 3,095 | Seller locations |
| `geolocation` | 1,000,163 | Zip-code coordinates |
| `product_category` | 71 | Portuguese → English category names |

Two schema details matter for feature engineering:

- **`customer_id` is per order.** The same person gets a new `customer_id` on
  each order; `customer_unique_id` identifies the customer. All features are
  aggregated by `customer_unique_id`.
- **`order_payments` is per payment, not per order.** Vouchers and split
  payments create several rows for one order; it must be aggregated before any
  join (see [incident 04](../../docs/incidents/04-sql-join-fan-out.md)).

## Data policy

Raw and processed data stay out of Git (`data/raw/`, `data/processed/` are
ignored). Only code and small test fixtures are versioned.
