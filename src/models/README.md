# Models

Training pipeline with MLflow tracking and model registry.

## Pipeline position

```
features.parquet  →  train.py  →  MLflow registry (churn-prediction)  →  src/api/
```

```bash
make up                       # MLflow server on http://localhost:5001
python -m src.models.train    # or: make train
```

Each run logs parameters, metrics, the feature list and the model (native
XGBoost format, with input signature and example), then registers a new
version of `churn-prediction`.

## Model card

| | |
|---|---|
| **Task** | Probability that a repeat customer places no delivered order in the next 180 days |
| **Population** | Olist customers with ≥ 2 delivered orders before the cutoff (1,632) |
| **Algorithm** | `XGBClassifier` — 200 trees, depth 2, learning rate 0.05, `min_child_weight` 5, subsample/colsample 0.8 |
| **Inputs** | `MODEL_FEATURES` in `src/features/build.py` (8 RFM-derived features) |
| **Output** | `churn_probability` (calibrated) and `churn_prediction` at `CHURN_DECISION_THRESHOLD` (default 0.5) |
| **Split** | Stratified 75/25 hold-out + repeated stratified 5×5 CV on the training part |
| **Baseline** | Standardized logistic regression, cross-validated on every run |

### Metrics (`churn-prediction/v1`)

| Metric | Value | How to read it |
|--------|-------|----------------|
| `cv_roc_auc_mean` ± `cv_roc_auc_std` | 0.723 ± 0.058 | Main reference number |
| `baseline_cv_roc_auc_mean` | 0.707 | Logistic regression on the same folds |
| `roc_auc` (hold-out) | 0.637 | Noisy: 17 retained customers in the hold-out set |
| `pr_auc_retained` | 0.154 | vs. a base rate (`retained_rate`) of 0.042 |
| `brier` | 0.0391 | vs. 0.0399 for always predicting the base rate |
| `precision` / `recall` at 0.5 | 0.961 / 1.000 | Mirror the 95.8% churn rate, not model quality |

### Configuration choices

Compared on repeated 5×5 CV (churn window 180 days):

| Configuration | ROC-AUC | Brier |
|---------------|--------:|------:|
| Depth 4, 100 trees, `scale_pos_weight` = neg/pos (previous) | 0.680 ± 0.063 | 0.113 |
| Depth 4, 100 trees, no re-weighting | 0.672 ± 0.062 | 0.039 |
| **Depth 2, 200 trees, lr 0.05, no re-weighting (current)** | **0.704 ± 0.066** | **0.038** |
| Logistic regression | 0.712 ± 0.069 | 0.038 |

Re-weighting the classes barely changed the ranking but destroyed calibration.
Since the API serves probabilities, the current configuration was chosen.

### Limitations

- Small minority class (68 retained customers): every metric has a wide
  confidence interval, so CV spread is reported with each run.
- Features cover purchase behaviour only; satisfaction, delivery and product
  signals are planned for later phases.
- Single cutoff date: the model is trained on one snapshot. Rolling cutoffs
  would multiply training examples and are a P2 candidate.

## Registry and serving

The API loads `models:/churn-prediction/latest`, resolves it to the concrete
version at startup (e.g. `churn-prediction/v1`) and reports that version in
`/health` and every prediction log. Restarting the API after a training run
picks up the new version. Staging/production aliases and automated retraining
are part of P3.
