"""
Central configuration for the Olist churn system.

Every tunable value is read from the environment (optionally through a local
.env file) so the same code runs unchanged on a laptop, in Docker Compose and
in CI. Modules import constants from here instead of calling os.getenv
themselves, which keeps variable names consistent across the codebase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=False)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", BASE_DIR / "data" / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", BASE_DIR / "data" / "processed"))
RFM_FEATURES_PATH = PROCESSED_DATA_DIR / "rfm_features.parquet"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.parquet"

# ── PostgreSQL ─────────────────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "olist_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "olist_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "olist_pass")

DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ── Churn definition ───────────────────────────────────────────────────────────
# A customer is labelled as churned when they place no delivered order during
# the CHURN_DAYS window that follows the cutoff date. See README, section
# "Data & Modeling Decisions", for why 180 days is the default.
CHURN_DAYS = int(os.getenv("CHURN_DAYS", "180"))

# Probability above which churn_prediction = 1. The probability is the primary
# output: with a ~96% base rate a calibrated model scores most customers above 0.5.
CHURN_DECISION_THRESHOLD = float(os.getenv("CHURN_DECISION_THRESHOLD", "0.5"))

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "churn-prediction")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{REGISTERED_MODEL_NAME}/latest")
