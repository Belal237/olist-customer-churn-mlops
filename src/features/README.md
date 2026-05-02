# Features

SQL and Python pipeline to generate model-ready features from raw Olist data.

## Pipeline position
PostgreSQL (raw Olist tables)
└── rfm.sql (via sql_features.py)
└── rfm_features.parquet
└── build.py
└── features.parquet → train.py

## Files

| File | Role |
|------|------|
| `rfm.sql` | SQL query — RFM aggregates + churn label with cutoff date |
| `sql_features.py` | Executes rfm.sql against PostgreSQL, validates and saves rfm_features.parquet |
| `build.py` | Python transformations, Pydantic validation, saves features.parquet |

## Features produced

| Feature | Description |
|---------|-------------|
| `recency_days` | Days since last order before cutoff date |
| `frequency` | Number of distinct delivered orders |
| `monetary` | Total actual spend (payment_value) |
| `monetary_log` | log1p(monetary) — reduces right skew |
| `avg_order_value` | monetary / frequency |
| `days_per_order` | customer_lifetime_days / frequency |
| `avg_days_between_orders` | Average gap between consecutive orders |
| `customer_lifetime_days` | Days between first and last order |
| `is_one_time_buyer` | 1 if frequency == 1 |
| `churn_label` | 1 if no purchase after cutoff date, 0 otherwise |

## Key decisions

**Cutoff date approach** — features and label are computed on separate time windows
to prevent data leakage:
- Features: orders BEFORE cutoff_date (ref_date - churn_days)
- Label: presence/absence of orders AFTER cutoff_date

**frequency >= 2 filter** — churn is only meaningful for repeat customers.
One-time buyers (~97% of Olist customers) are excluded from the training set.

**churn_days = 90** — optimal class distribution on repeat customers.
180 days produced 98.8% churn rate with no learnable signal.

## Run

```bash
# Step 1 — extract RFM features from PostgreSQL
python src/features/sql_features.py

# Step 2 — build engineered features
python src/features/build.py
```

## Output

`data/processed/features.parquet` — 2,226 rows × 11 columns, ready for training.