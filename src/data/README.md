# Data

Data loading, validation and SQL feature extraction for the Olist churn system.

## Pipeline position
Kaggle (raw CSV files)
└── load.py → PostgreSQL (olist_db)
└── sql_features.py (rfm.sql)
└── data/processed/rfm_features.parquet
└── src/features/build.py

## Files

| File | Role |
|------|------|
| `load.py` | Loads raw Olist CSV files into PostgreSQL |
| `sql_features.py` | Executes rfm.sql, validates output, saves rfm_features.parquet |

## Dataset

**Olist Brazilian E-Commerce** — https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

| Table | Description |
|-------|-------------|
| orders | 99k orders with status and timestamps |
| customers | Customer unique IDs and zip codes |
| order_items | Products and sellers per order |
| order_payments | Actual payment values per order |
| order_reviews | Customer satisfaction scores |
| products | Product categories and dimensions |
| sellers | Seller locations |
| geolocation | Brazilian zip code coordinates |
| product_category_name_translation | Category name translations |

## Setup

```bash
# Download dataset via Kaggle API
kaggle datasets download olistbr/brazilian-ecommerce

# Start PostgreSQL
docker compose up -d

# Load CSV files into PostgreSQL
python src/data/load.py
```

## Raw data

Raw CSV files are stored in `data/raw/` and excluded from Git via `.gitignore`.
Never commit raw data to the repository.

## Processed data

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `data/processed/rfm_features.parquet` | RFM aggregates + churn label | 2,226 | 7 |
| `data/processed/features.parquet` | Engineered features ready for training | 2,226 | 11 |