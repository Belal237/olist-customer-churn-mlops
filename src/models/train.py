"""
Training pipeline for the Olist churn model.

Loads features.parquet, trains an XGBoost classifier, evaluates it on a
stratified hold-out set, logs everything to MLflow and registers the model
under config.REGISTERED_MODEL_NAME so the API can load the latest version.

Pipeline position: features.parquet → [this file] → MLflow Registry → src/api/main.py

Usage:
    python -m src.models.train
"""

import logging
import random
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config
from src.features.build import ID_COLUMN, MODEL_FEATURES, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
CV_REPEATS = 5
DECISION_THRESHOLD = 0.5

# Shallow, regularized trees: ~1.6k rows and only ~70 retained customers.
# No scale_pos_weight: re-weighting barely moved ROC-AUC but destroyed
# calibration (Brier 0.113 vs 0.038), and the API serves probabilities.
# See src/models/README.md for the comparison.
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 2,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
}


# ── Steps ──────────────────────────────────────────────────────────────────────


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_data(path: Path) -> pd.DataFrame:
    """Load the feature table and fail fast if the training contract is broken."""
    df = pd.read_parquet(path)
    missing = set(MODEL_FEATURES + [ID_COLUMN, TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Feature table {path} is missing columns: {sorted(missing)}")
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training data contains a single class — check CHURN_DAYS.")
    return df


def split_data(df: pd.DataFrame):
    """Stratified split — keeps the (very skewed) churn rate identical in both sets."""
    X = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)


def build_model() -> XGBClassifier:
    return XGBClassifier(**MODEL_PARAMS, random_state=SEED)


def build_baseline() -> Pipeline:
    """Logistic regression reference: the XGBoost model has to justify its complexity."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def cross_validate_auc(estimator, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Repeated stratified k-fold ROC-AUC on the training set.
    The hold-out set holds only ~17 retained customers, so a single test AUC is
    noisy; the CV spread tells how much a reported number can be trusted.
    """
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=SEED)
    return cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc")


def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """
    Hold-out metrics. With ~96% churners, a calibrated model scores almost everyone
    above 0.5, so precision/recall at that threshold mostly mirror the base rate.
    The ranking and calibration metrics carry the signal:
        - roc_auc / brier: overall ranking and calibration (compare brier with
          churn_rate * retained_rate, the score of always predicting the base rate)
        - pr_auc_retained: average precision on the minority class (retained
          customers), compared with its base rate retained_rate
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc_churn": average_precision_score(y_test, y_proba),
        "pr_auc_retained": average_precision_score(1 - y_test, 1 - y_proba),
        "brier": brier_score_loss(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "churn_rate": float(y_test.mean()),
        "retained_rate": float(1 - y_test.mean()),
    }


# ── Pipeline ───────────────────────────────────────────────────────────────────


def train(
    features_path: Path = config.FEATURES_PATH,
    tracking_uri: str = config.MLFLOW_TRACKING_URI,
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
    registered_model_name: str = config.REGISTERED_MODEL_NAME,
) -> dict[str, float]:
    """Train, evaluate, log and register the churn model. Returns hold-out metrics."""
    set_seed(SEED)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    df = load_data(features_path)
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info(
        "Data: %d rows (train=%d, test=%d) | churn rate %.1f%%",
        len(df),
        len(X_train),
        len(X_test),
        df[TARGET_COLUMN].mean() * 100,
    )

    with mlflow.start_run() as run:
        cv_auc = cross_validate_auc(build_model(), X_train, y_train)
        baseline_auc = cross_validate_auc(build_baseline(), X_train, y_train)
        logger.info("CV ROC-AUC: xgboost %.3f ± %.3f", cv_auc.mean(), cv_auc.std())
        logger.info(
            "CV ROC-AUC: logreg baseline %.3f ± %.3f", baseline_auc.mean(), baseline_auc.std()
        )

        model = build_model()
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics |= {
            "cv_roc_auc_mean": cv_auc.mean(),
            "cv_roc_auc_std": cv_auc.std(),
            "baseline_cv_roc_auc_mean": baseline_auc.mean(),
        }

        mlflow.log_params(
            {
                "model_type": "xgboost",
                **MODEL_PARAMS,
                "decision_threshold": DECISION_THRESHOLD,
                "churn_days": config.CHURN_DAYS,
                "test_size": TEST_SIZE,
                "cv_folds": CV_FOLDS,
                "cv_repeats": CV_REPEATS,
                "n_rows": len(df),
                "features": ",".join(MODEL_FEATURES),
            }
        )
        mlflow.log_metrics(metrics)

        # Native XGBoost format (not pickle): portable across library versions.
        # The signature documents the expected input schema in the registry.
        mlflow.xgboost.log_model(
            model,
            name="model",
            registered_model_name=registered_model_name,
            signature=infer_signature(X_train, model.predict_proba(X_train)[:, 1]),
            input_example=X_train.head(3),
        )

        logger.info("Run %s logged and registered as '%s'.", run.info.run_id, registered_model_name)
        for name, value in metrics.items():
            logger.info("  %-24s %.4f", name, value)

    return metrics


if __name__ == "__main__":
    train()
