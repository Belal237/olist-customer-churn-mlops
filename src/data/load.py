"""
Load the raw Olist CSV files into PostgreSQL.

Each table is dropped and recreated, then bulk-loaded with COPY, which is an
order of magnitude faster than row-by-row INSERTs on the 1M+ row geolocation
table. Indexes on the join keys used by rfm.sql are created afterwards.

Pipeline position: data/raw/*.csv → [this file] → PostgreSQL → sql_features.py

Usage:
    python -m src.data.load
"""

import logging

import psycopg2
from psycopg2 import sql

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEMAS = {
    "customers": {
        "file": "olist_customers_dataset.csv",
        "columns": [
            ("customer_id", "TEXT PRIMARY KEY"),
            ("customer_unique_id", "TEXT"),
            ("customer_zip_code_prefix", "TEXT"),
            ("customer_city", "TEXT"),
            ("customer_state", "TEXT"),
        ],
    },
    "orders": {
        "file": "olist_orders_dataset.csv",
        "columns": [
            ("order_id", "TEXT PRIMARY KEY"),
            ("customer_id", "TEXT"),
            ("order_status", "TEXT"),
            ("order_purchase_timestamp", "TIMESTAMP"),
            ("order_approved_at", "TIMESTAMP"),
            ("order_delivered_carrier_date", "TIMESTAMP"),
            ("order_delivered_customer_date", "TIMESTAMP"),
            ("order_estimated_delivery_date", "TIMESTAMP"),
        ],
    },
    "order_items": {
        "file": "olist_order_items_dataset.csv",
        "columns": [
            ("order_id", "TEXT"),
            ("order_item_id", "INTEGER"),
            ("product_id", "TEXT"),
            ("seller_id", "TEXT"),
            ("shipping_limit_date", "TIMESTAMP"),
            ("price", "NUMERIC(10,2)"),
            ("freight_value", "NUMERIC(10,2)"),
        ],
    },
    "order_payments": {
        "file": "olist_order_payments_dataset.csv",
        "columns": [
            ("order_id", "TEXT"),
            ("payment_sequential", "INTEGER"),
            ("payment_type", "TEXT"),
            ("payment_installments", "INTEGER"),
            ("payment_value", "NUMERIC(10,2)"),
        ],
    },
    "order_reviews": {
        "file": "olist_order_reviews_dataset.csv",
        "columns": [
            ("review_id", "TEXT"),
            ("order_id", "TEXT"),
            ("review_score", "SMALLINT"),
            ("review_comment_title", "TEXT"),
            ("review_comment_message", "TEXT"),
            ("review_creation_date", "TIMESTAMP"),
            ("review_answer_timestamp", "TIMESTAMP"),
        ],
    },
    "products": {
        "file": "olist_products_dataset.csv",
        "columns": [
            ("product_id", "TEXT PRIMARY KEY"),
            ("product_category_name", "TEXT"),
            ("product_name_lenght", "INTEGER"),
            ("product_description_lenght", "INTEGER"),
            ("product_photos_qty", "INTEGER"),
            ("product_weight_g", "INTEGER"),
            ("product_length_cm", "INTEGER"),
            ("product_height_cm", "INTEGER"),
            ("product_width_cm", "INTEGER"),
        ],
    },
    "sellers": {
        "file": "olist_sellers_dataset.csv",
        "columns": [
            ("seller_id", "TEXT PRIMARY KEY"),
            ("seller_zip_code_prefix", "TEXT"),
            ("seller_city", "TEXT"),
            ("seller_state", "TEXT"),
        ],
    },
    "geolocation": {
        "file": "olist_geolocation_dataset.csv",
        "columns": [
            ("geolocation_zip_code_prefix", "TEXT"),
            ("geolocation_lat", "NUMERIC(10,6)"),
            ("geolocation_lng", "NUMERIC(10,6)"),
            ("geolocation_city", "TEXT"),
            ("geolocation_state", "TEXT"),
        ],
    },
    "product_category": {
        "file": "product_category_name_translation.csv",
        "columns": [
            ("product_category_name", "TEXT PRIMARY KEY"),
            ("product_category_name_english", "TEXT"),
        ],
    },
}


# Join keys used by src/features/rfm.sql — created once the data is loaded
INDEXES = {
    "orders": ["customer_id", "order_status"],
    "customers": ["customer_unique_id"],
    "order_items": ["order_id"],
    "order_payments": ["order_id"],
    "order_reviews": ["order_id"],
}


def load_table(conn, table_name: str, schema: dict) -> int:
    """Drop, recreate and bulk-load one table. Returns the loaded row count."""
    csv_path = config.RAW_DATA_DIR / schema["file"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing raw file: {csv_path}. "
            "Download the dataset with: kaggle datasets download olistbr/brazilian-ecommerce"
        )

    table = sql.Identifier(table_name)
    col_defs = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(col_type))
        for col, col_type in schema["columns"]
    )

    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(table))
        cur.execute(sql.SQL("CREATE TABLE {} ({})").format(table, col_defs))
        with open(csv_path, encoding="utf-8") as f:
            cur.copy_expert(
                sql.SQL("COPY {} FROM STDIN WITH CSV HEADER NULL ''").format(table),
                f,
            )
        for column in INDEXES.get(table_name, []):
            cur.execute(
                sql.SQL("CREATE INDEX {} ON {} ({})").format(
                    sql.Identifier(f"idx_{table_name}_{column}"),
                    table,
                    sql.Identifier(column),
                )
            )
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(table))
        n_rows = cur.fetchone()[0]
    conn.commit()
    return n_rows


def main() -> None:
    logger.info(
        "Connecting to PostgreSQL at %s:%s/%s",
        config.POSTGRES_HOST,
        config.POSTGRES_PORT,
        config.POSTGRES_DB,
    )
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
    )
    try:
        for table_name, schema in SCHEMAS.items():
            n_rows = load_table(conn, table_name, schema)
            logger.info("Loaded %-18s %9s rows", table_name, f"{n_rows:,}")
    finally:
        conn.close()
    logger.info("All tables loaded.")


if __name__ == "__main__":
    main()
