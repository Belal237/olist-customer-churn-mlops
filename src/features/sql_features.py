"""
RFM feature extraction using PostgreSQL (SQLAlchemy + psycopg2).
Reads the RFM query from src/features/rfm.sql and executes it against
the olist_db database, then saves the result as a parquet file.

Pipeline position: RAW DATA → [this file] → rfm_features.parquet → build.py
"""

import os
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent.parent
SQL_PATH       = BASE_DIR / "src" / "features" / "rfm.sql"
DEFAULT_OUTPUT = BASE_DIR / "data" / "processed" / "rfm_features.parquet"

# ── Database configuration ─────────────────────────────────────────────────────
DB_USER     = os.getenv("POSTGRES_USER",     "olist_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "olist_pass")
DB_HOST     = os.getenv("POSTGRES_HOST",     "localhost")
DB_PORT     = os.getenv("POSTGRES_PORT",     "5432")
DB_NAME     = os.getenv("POSTGRES_DB",       "olist_db")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── Churn threshold ────────────────────────────────────────────────────────────
# 180 days chosen based on dataset distribution analysis:
# - median recency = 218 days
# - 90-day threshold  → 80.1% churn rate (too imbalanced for ML)
# - 180-day threshold → 58.9% churn rate (acceptable class balance)
CHURN_DAYS: int = 90


def get_engine():
    """
    Create and return a SQLAlchemy engine connected to PostgreSQL.
    pool_pre_ping=True verifies the connection is alive before each query.
    """
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    logger.info("Connected to PostgreSQL: %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)
    return engine


def load_query(sql_path: Path) -> str:
    """
    Load a SQL query from an external .sql file.
    Keeping SQL in a dedicated file allows syntax highlighting,
    version control diffs, and reuse outside Python.
    """
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")
    return sql_path.read_text(encoding="utf-8")


def compute_rfm(engine) -> pd.DataFrame:
    """
    Execute the RFM query against PostgreSQL.
    :churn_days is passed as a bound parameter — never interpolated via f-string.
    This is the correct SQLAlchemy pattern and prevents SQL injection.
    """
    query = load_query(SQL_PATH)
    logger.info("Running RFM query (churn_days=%d)...", CHURN_DAYS)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(query),
            conn,
            params={"churn_days": CHURN_DAYS},
        )
    logger.info("Query returned %s rows x %s columns", f"{len(df):,}", df.shape[1])
    return df


def validate_rfm(df: pd.DataFrame) -> None:
    """
    Assert data quality before saving to disk.
    Silent bad data is worse than a loud failure — fail fast and explicitly.
    """
    # One row per customer — verifies GROUP BY correctness
    n_unique = df["customer_unique_id"].nunique()
    assert n_unique == len(df), (
        f"Duplicate customer_unique_id detected — "
        f"{len(df) - n_unique} duplicates found. Check GROUP BY logic."
    )

    # No nulls in core features
    key_cols = ["recency_days", "frequency", "monetary", "churn_label"]
    for col in key_cols:
        n_nulls = df[col].isna().sum()
        assert n_nulls == 0, f"Null values in '{col}': {n_nulls} rows affected"

    # Churn label must be strictly binary
    assert set(df["churn_label"].unique()).issubset({0, 1}), (
        "churn_label contains values other than 0 and 1"
    )

    # Value range sanity checks
    assert (df["frequency"]    >= 1).all(), "frequency must be >= 1"
    assert (df["monetary"]     >= 0).all(), "monetary must be >= 0"
    assert (df["recency_days"] >= 0).all(), "recency_days must be >= 0"

    churn_rate = df["churn_label"].mean()
    logger.info(
        "Churn rate: %.1f%% (%s churned / %s total)",
        churn_rate * 100,
        f"{df['churn_label'].sum():,}",
        f"{len(df):,}",
    )
    logger.info("All validation checks passed.")


def build_rfm_dataset(output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """
    Full pipeline:
        1. Connect to PostgreSQL
        2. Load and execute rfm.sql with bound parameters
        3. Validate data quality
        4. Save to parquet (preserves types, faster than CSV)

    Returns:
        pd.DataFrame: validated RFM feature table
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    df     = compute_rfm(engine)
    validate_rfm(df)

    df.to_parquet(output_path, index=False)
    logger.info("RFM features saved → %s", output_path)

    return df


if __name__ == "__main__":
    df = build_rfm_dataset()

    print("\n--- Sample (first 10 rows) ---")
    print(df.head(10).to_string(index=False))

    print("\n--- Descriptive statistics ---")
    print(df.describe())

    print("\n--- Class distribution ---")
    print(df["churn_label"].value_counts())