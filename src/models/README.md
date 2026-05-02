# Models

XGBoost training pipeline with MLflow experiment tracking and model registry.

## Pipeline position
data/processed/features.parquet
└── train.py
└── MLflow Registry (models:/churn-prediction/latest)
└── src/api/main.py

## Files

| File | Role |
|------|------|
| `train.py` | Full training pipeline — load, split, train, evaluate, log, register |

## Run

```bash
python src/models/train.py
```

## View results in MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://localhost:5000 in your browser.

## Model

**Algorithm**: XGBoost (XGBClassifier)

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_depth | 4 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| scale_pos_weight | 0.019 (computed from training set) |
| eval_metric | logloss |

## Metrics (latest run)

| Metric | Value |
|--------|-------|
| AUC | 0.797 |
| Precision | 0.990 |
| Recall | 0.890 |

## Class imbalance handling

Residual class imbalance (neg/pos ratio = 0.019) handled via `scale_pos_weight`,
computed automatically at training time and logged in MLflow.

## Model Registry

The model is registered in the MLflow Model Registry under `churn-prediction`.
The API always loads `models:/churn-prediction/latest` — no hardcoded run_id needed.
Each new training run creates a new version automatically.