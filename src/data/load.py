import psycopg2
import os
from pathlib import Path

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "olist_db"),
    "user": os.getenv("DB_USER", "olist_user"),
    "password": os.getenv("DB_PASSWORD", "olist_pass"),
}

BASE_DIR = Path(__file__).parent.parent.parent

SCHEMAS = {
    "customers": {
        "file": "olist_customers_dataset.csv",
        "columns": [
            ("customer_id", "TEXT PRIMARY KEY"),
            ("customer_unique_id", "TEXT"),
            ("customer_zip_code_prefix", "TEXT"),
            ("customer_city", "TEXT"),
            ("customer_state", "TEXT"),
        ]
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
        ]
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
        ]
    },
    "order_payments": {
        "file": "olist_order_payments_dataset.csv",
        "columns": [
            ("order_id", "TEXT"),
            ("payment_sequential", "INTEGER"),
            ("payment_type", "TEXT"),
            ("payment_installments", "INTEGER"),
            ("payment_value", "NUMERIC(10,2)"),
        ]
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
        ]
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
        ]
    },
    "sellers": {
        "file": "olist_sellers_dataset.csv",
        "columns": [
            ("seller_id", "TEXT PRIMARY KEY"),
            ("seller_zip_code_prefix", "TEXT"),
            ("seller_city", "TEXT"),
            ("seller_state", "TEXT"),
        ]
    },
    "geolocation": {
        "file": "olist_geolocation_dataset.csv",
        "columns": [
            ("geolocation_zip_code_prefix", "TEXT"),
            ("geolocation_lat", "NUMERIC(10,6)"),
            ("geolocation_lng", "NUMERIC(10,6)"),
            ("geolocation_city", "TEXT"),
            ("geolocation_state", "TEXT"),
        ]
    },
    "product_category": {
        "file": "product_category_name_translation.csv",
        "columns": [
            ("product_category_name", "TEXT PRIMARY KEY"),
            ("product_category_name_english", "TEXT"),
        ]
    },
}


def load_table(conn, table_name: str, schema: dict) -> None:
    csv_path = BASE_DIR / "src" / "data" / schema["file"]
    col_defs = ", ".join(f'"{col}" {col_type}' for col, col_type in schema["columns"])

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        cur.execute(f"CREATE TABLE {table_name} ({col_defs})")
        with open(csv_path, "r", encoding="utf-8") as f:
            next(f)  # skip header
            cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV NULL ''", f)
        conn.commit()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        n = cur.fetchone()[0]
        print(f"  {table_name}: {n} lignes")


def main():
    print("Connexion à PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("Chargement des tables :")
    for table_name, schema in SCHEMAS.items():
        load_table(conn, table_name, schema)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()