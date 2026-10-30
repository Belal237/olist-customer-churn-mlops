"""
In-memory feature store served by the API.

The feature table is small (one row per repeat customer), so it is loaded once
at startup into a float matrix plus a {customer_id: row} dict. Lookups are a
dict access and a numpy slice — no pandas filtering on the request path.

Rows are validated at load time. XGBoost accepts NaN as "missing", so a
corrupted row still yields a plausible-looking score: invalid rows are
quarantined and never scored (see docs/incidents/01-silent-nan-features.md).
"""

import logging
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build import ID_COLUMN, MODEL_FEATURES
from src.logging_config import log_event

logger = logging.getLogger(__name__)

# Above this share of invalid rows the corruption is systemic: refuse to start.
MAX_QUARANTINE_RATIO = 0.05


class FeatureStoreError(RuntimeError):
    """Raised when the feature table cannot be served at all."""


class CustomerStatus(StrEnum):
    OK = "ok"
    NOT_FOUND = "not_found"
    INVALID_FEATURES = "invalid_features"


class FeatureStore:
    def __init__(self, df: pd.DataFrame, max_quarantine_ratio: float | None = None):
        if max_quarantine_ratio is None:
            max_quarantine_ratio = MAX_QUARANTINE_RATIO
        missing = set(MODEL_FEATURES + [ID_COLUMN]) - set(df.columns)
        if missing:
            raise FeatureStoreError(f"Feature table is missing columns: {sorted(missing)}")
        if df[ID_COLUMN].duplicated().any():
            raise FeatureStoreError(f"Duplicate {ID_COLUMN} values in feature table")

        ids = df[ID_COLUMN].astype(str).to_numpy()
        matrix = df[MODEL_FEATURES].to_numpy(dtype=np.float64)

        # Every model input must be finite and non-negative (see EngineeredFeatures)
        valid = np.isfinite(matrix).all(axis=1) & (np.nan_to_num(matrix, nan=-1.0) >= 0).all(axis=1)
        self.quarantined: frozenset[str] = frozenset(ids[~valid])

        ratio = len(self.quarantined) / max(len(ids), 1)
        if ratio > max_quarantine_ratio:
            raise FeatureStoreError(
                f"{len(self.quarantined)} of {len(ids)} rows ({ratio:.1%}) failed validation, "
                f"above the {max_quarantine_ratio:.0%} limit. Rebuild the features."
            )
        if self.quarantined:
            log_event(
                logger,
                "feature_store_rows_quarantined",
                level=logging.WARNING,
                n_quarantined=len(self.quarantined),
                sample_ids=sorted(self.quarantined)[:10],
            )

        self._matrix = matrix
        self._row_by_id = {cid: i for i, cid in enumerate(ids)}

    @classmethod
    def from_parquet(cls, path: Path) -> "FeatureStore":
        if not Path(path).exists():
            raise FeatureStoreError(f"Feature store not found: {path}")
        return cls(pd.read_parquet(path))

    def __len__(self) -> int:
        return len(self._row_by_id)

    def status(self, customer_id: str) -> CustomerStatus:
        if customer_id not in self._row_by_id:
            return CustomerStatus.NOT_FOUND
        if customer_id in self.quarantined:
            return CustomerStatus.INVALID_FEATURES
        return CustomerStatus.OK

    def get_features(self, customer_ids: list[str]) -> pd.DataFrame:
        """Model input for customers whose status is OK, in the requested order."""
        rows = [self._row_by_id[cid] for cid in customer_ids]
        return pd.DataFrame(self._matrix[rows], columns=MODEL_FEATURES)
