"""
Feature engineering pipeline for the Olist churn model.
Reads rfm_features.parquet produced by sql_features.py,
applies Python transformations, validates with Pydantic v2,
and saves the final feature table ready for model training.

Pipeline position: rfm_features.parquet → [this file] → features.parquet → train.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent.parent
DEFAULT_INPUT  = BASE_DIR / "data" / "processed" / "rfm_features.parquet"
DEFAULT_OUTPUT = BASE_DIR / "data" / "processed" / "features.parquet"

# ── Final column order (fed into the model) ────────────────────────────────────
FEATURE_COLUMNS = [
    "customer_unique_id",
    "recency_days",
    "frequency",
    "monetary",
    "monetary_log",
    "avg_order_value",
    "days_per_order",
    "avg_days_between_orders",
    "customer_lifetime_days",
    "is_one_time_buyer",
    "churn_label",
]


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class RawCustomerFeatures(BaseModel):
    """Schema for one row coming out of sql_features.py (rfm_features.parquet)."""
    customer_unique_id:       str
    recency_days:             int   = Field(ge=0)
    frequency:                int   = Field(ge=1)
    monetary:                 float = Field(ge=0.0)
    customer_lifetime_days:   int   = Field(ge=0)
    avg_days_between_orders:  float = Field(ge=0.0)
    churn_label:              int   = Field(ge=0, le=1)

    model_config = {"strict": True}


class EngineeredFeatures(BaseModel):
    """Schema for one row after feature engineering — input to the model."""
    customer_unique_id:      str
    recency_days:            int
    frequency:               int
    monetary:                float
    monetary_log:            float  # log1p(monetary) — reduces right skew
    avg_order_value:         float  # monetary / frequency
    days_per_order:          float  # customer_lifetime_days / frequency
    avg_days_between_orders: float
    customer_lifetime_days:  int
    is_one_time_buyer:       int    # 1 if frequency == 1
    churn_label:             int

    @model_validator(mode="after")
    def check_no_negatives(self) -> "EngineeredFeatures":
        """All numeric features must be non-negative after engineering."""
        for field in [
            "recency_days", "frequency", "monetary",
            "avg_order_value", "days_per_order",
        ]:
            if getattr(self, field) < 0:
                raise ValueError(f"Feature '{field}' must be >= 0")
        return self


# ── Pure transformation functions ──────────────────────────────────────────────
# Each function is pure: takes a DataFrame, returns a new one, never mutates.

def add_monetary_log(df: pd.DataFrame) -> pd.DataFrame:
    """log1p transform on monetary to reduce right skew."""
    df = df.copy()
    df["monetary_log"] = np.log1p(df["monetary"])
    return df


def add_avg_order_value(df: pd.DataFrame) -> pd.DataFrame:
    """Average spend per order = monetary / frequency."""
    df = df.copy()
    df["avg_order_value"] = df["monetary"] / df["frequency"]
    return df


def add_days_per_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average days per order over the customer lifetime.
    One-time buyers have frequency=1 and lifetime=0,
    so days_per_order = 0 for them — no division issue.
    """
    df = df.copy()
    df["days_per_order"] = np.where(
        df["frequency"] > 1,
        df["customer_lifetime_days"] / df["frequency"],
        0.0,
    )
    return df


def add_one_time_buyer_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if the customer placed exactly one order."""
    df = df.copy()
    df["is_one_time_buyer"] = (df["frequency"] == 1).astype(int)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation strategy (documented decisions):

    - avg_days_between_orders: filled with 0 for one-time buyers.
      They have no gap between orders by definition — 0 is semantically correct,
      not an arbitrary fill value.

    - All other core columns: must have no nulls at this stage.
      If they do, the bug is upstream (in rfm.sql or sql_features.py)
      and must be fixed there — not silently patched here.
    """
    df = df.copy()
    df["avg_days_between_orders"] = df["avg_days_between_orders"].fillna(0.0)

    critical_cols = [
        "recency_days", "frequency", "monetary",
        "customer_lifetime_days", "churn_label",
    ]
    for col in critical_cols:
        n_nulls = df[col].isna().sum()
        if n_nulls > 0:
            raise ValueError(
                f"Unexpected NaNs in '{col}': {n_nulls} rows. "
                "Fix this upstream in rfm.sql or sql_features.py."
            )
    return df


# ── Schema validation on a sample ─────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, sample_size: int = 500) -> None:
    """
    Validate a random sample of rows against the EngineeredFeatures schema.
    Catches type errors and business rule violations before saving.
    Sampling (not full scan) keeps validation fast on large datasets.
    """
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    for _, row in sample.iterrows():
        EngineeredFeatures(**row.to_dict())
    logger.info("Pydantic schema validation passed (%d rows sampled).", len(sample))


# ── Full pipeline ──────────────────────────────────────────────────────────────

def build_features(
    input_path:  Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
        1. Load rfm_features.parquet
        2. Handle missing values (fail fast on unexpected nulls)
        3. Apply transformations (all pure functions)
        4. Select and order final columns
        5. Validate schema with Pydantic v2
        6. Save to features.parquet

    Returns:
        pd.DataFrame: engineered features ready for model training
    """
    logger.info("Loading RFM features from: %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Input shape: %s rows x %s columns", f"{len(df):,}", df.shape[1])

    # Transformations applied in order — each step is independent and pure
    df = handle_missing_values(df)
    df = add_monetary_log(df)
    df = add_avg_order_value(df)
    df = add_days_per_order(df)
    df = add_one_time_buyer_flag(df)

    # Keep only expected columns in the correct order
    df = df[FEATURE_COLUMNS]

    # Validate types and business rules on a sample
    validate_schema(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Features saved → %s | shape: %s x %s", output_path, f"{len(df):,}", df.shape[1])

    return df


if __name__ == "__main__":
    df = build_features()

    print("\n--- Sample (first 10 rows) ---")
    print(df.head(10).to_string(index=False))

    print("\n--- Descriptive statistics ---")
    print(df.describe())

    print("\n--- Class distribution ---")
    print(df["churn_label"].value_counts())