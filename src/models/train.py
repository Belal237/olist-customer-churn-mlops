import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

# =========================
# CONFIG
# =========================

SEED = 42
TEST_SIZE = 0.25
TARGET = "churn_label"
ID_COLUMNS = ["customer_unique_id"]

DATA_PATH = "data/processed/features.parquet"


# =========================
# UTILS
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET] + ID_COLUMNS)
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )


def build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False
    )


def evaluate(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    return metrics


# =========================
# MAIN TRAINING
# =========================

def train():

    print("Starting training...")

    # 1. Reproductivity
    set_seed(SEED)

    # 2. Load
    df = load_data(DATA_PATH)
    print(f"Data shape: {df.shape}")

    # 3. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 4. MLflow
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():

        # 5. Model
        model = build_model()

        # 6. Train
        model.fit(X_train, y_train)

        # 7. Evaluate
        metrics = evaluate(model, X_test, y_test)

        # 8. Log params
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.1)

        # 9. Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # 10. Log model
        mlflow.sklearn.log_model(model, "model")

        print("Training complete")
        print(metrics)


# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    train()