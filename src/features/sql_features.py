"""
RFM feature extraction using PostgreSQL (SQLAlchemy + psycopg2).
Reads the RFM query from src/features/rfm.sql and executes it against
the olist_db database, then saves the result as a parquet file.

Pipeline position: RAW DATA → [this file] → rfm_features.parquet → build.py
"""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from src import config

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
SQL_PATH = Path(__file__).resolve().parent / "rfm.sql"
DEFAULT_OUTPUT = config.RFM_FEATURES_PATH

# Churn window in days — configurable through the CHURN_DAYS env variable.
CHURN_DAYS: int = config.CHURN_DAYS

RFM_DTYPES = {
    "customer_unique_id": "object",
    "recency_days": "int64",
    "frequency": "int64",
    "monetary": "float64",
    "customer_lifetime_days": "int64",
    "avg_days_between_orders": "float64",
    "churn_label": "int64",
}


def get_engine():
    """
    Create and return a SQLAlchemy engine connected to PostgreSQL.
    pool_pre_ping=True verifies the connection is alive before each query.
    """
    engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
    logger.info(
        "Connected to PostgreSQL: %s:%s/%s",
        config.POSTGRES_HOST,
        config.POSTGRES_PORT,
        config.POSTGRES_DB,
    )
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


def compute_rfm(engine, churn_days: int = CHURN_DAYS) -> pd.DataFrame:
    """
    Execute the RFM query against PostgreSQL.
    :churn_days is passed as a bound parameter — never interpolated via f-string.
    This is the correct SQLAlchemy pattern and prevents SQL injection.
    """
    query = load_query(SQL_PATH)
    logger.info("Running RFM query (churn_days=%d)...", churn_days)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"churn_days": churn_days})
    logger.info("Query returned %s rows x %s columns", f"{len(df):,}", df.shape[1])
    # NUMERIC columns come back as Decimal objects — cast to stable dtypes
    return df.astype(RFM_DTYPES)


class RFMValidationError(ValueError):
    """Raised when the RFM table violates a data quality rule."""


def validate_rfm(df: pd.DataFrame) -> None:
    """
    Check data quality before saving to disk.
    Silent bad data is worse than a loud failure — fail fast and explicitly.
    Explicit raises (not asserts) so checks still run under `python -O`.
    """
    errors: list[str] = []

    # One row per customer — verifies GROUP BY correctness
    n_duplicates = len(df) - df["customer_unique_id"].nunique()
    if n_duplicates:
        errors.append(f"{n_duplicates} duplicate customer_unique_id rows (check GROUP BY logic)")

    # No nulls in core features
    for col in ["recency_days", "frequency", "monetary", "churn_label"]:
        n_nulls = int(df[col].isna().sum())
        if n_nulls:
            errors.append(f"{n_nulls} null values in '{col}'")

    # Churn label must be strictly binary
    if not set(df["churn_label"].unique()).issubset({0, 1}):
        errors.append("churn_label contains values other than 0 and 1")

    # Value range sanity checks
    range_rules = {
        "frequency must be >= 2 (repeat customers only)": df["frequency"] >= 2,
        "monetary must be >= 0": df["monetary"] >= 0,
        "recency_days must be >= 0": df["recency_days"] >= 0,
        "customer_lifetime_days must be >= 0": df["customer_lifetime_days"] >= 0,
    }
    errors.extend(rule for rule, ok in range_rules.items() if not ok.all())

    if errors:
        raise RFMValidationError("RFM validation failed: " + "; ".join(errors))

    churn_rate = df["churn_label"].mean()
    logger.info(
        "Churn rate: %.1f%% (%s churned / %s total)",
        churn_rate * 100,
        f"{df['churn_label'].sum():,}",
        f"{len(df):,}",
    )
    logger.info("All validation checks passed.")


def build_rfm_dataset(
    output_path: Path = DEFAULT_OUTPUT,
    engine=None,
    churn_days: int = CHURN_DAYS,
) -> pd.DataFrame:
    """
    Full pipeline:
        1. Connect to PostgreSQL (or reuse the given engine)
        2. Load and execute rfm.sql with bound parameters
        3. Validate data quality
        4. Save to parquet (preserves types, faster than CSV)

    Returns:
        pd.DataFrame: validated RFM feature table
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    engine = engine or get_engine()
    df = compute_rfm(engine, churn_days=churn_days)
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
