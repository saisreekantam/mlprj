# SignalCluster Classification

Multi-class classification project predicting signal category (`Group_A`, `Group_B`, `Group_C`) from two continuous features: `signal_strength` and `response_level`.

**Best result:** 0.992 Macro F1 Score (MLP Classifier)
**Metric:** Macro F1 Score
**Data:** 1,444 training samples / 362 test samples, moderately imbalanced (46% / 34% / 20%)

## Overview

Over 30 model configurations across 7 algorithm families were tested, from logistic regression baselines to deep ensemble systems combining SVMs, neural networks, and gradient boosting. Macro F1 was used throughout (instead of accuracy or Micro F1) to ensure balanced performance across all three classes regardless of their sample counts.

## Top Results

| Model | Macro F1 | Notes |
|---|---|---|
| **MLP Classifier** | **0.992** | Single hidden layer (100 neurons), grid-search tuned |
| SVM–NN Ensemble | 0.988 | 5 SVMs + 10 NNs + 10 seed models, weighted soft voting |
| SVM (RBF kernel) | 0.987 | C=15, gamma=0.7 |
| Random Forest | 0.987 | 300 trees, balanced class weights |
| Extra Trees / Gradient Boosting | 0.986 | — |
| XGBoost | 0.984 | — |
| LightGBM | 0.982 | — |
| Logistic Regression | 0.981 | L2-regularized baseline |
| AdaBoost | 0.958 | Weakest performer |

## Key Findings

- A simple, well-regularized MLP (single 100-unit layer, alpha=0.0001) outperformed deeper/more complex networks and large ensembles.
- Feature engineering (polynomial terms, ratios, polar coordinates, etc.) gave negligible improvement — non-linear models already capture these interactions.
- Class imbalance was handled via `class_weight='balanced'`, stratified 5-fold CV, and Macro F1 optimization rather than resampling.
- Tree-based ensembles (Random Forest) matched or beat boosting methods (XGBoost, LightGBM) on this small, low-dimensional dataset.

## Recommended Usage

- **Production / best accuracy:** `mlp_classifier.py` — fastest training (~30s), highest score, simplest to maintain.
- **Maximum robustness:** `svm.py` — 25+ model ensemble, slightly lower score but more resilient to distribution shift.

## Pipeline

```python
# Scale features (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Train and predict
model.fit(X_train_scaled, y_train_encoded)
predictions = label_encoder.inverse_transform(model.predict(X_test_scaled))
```

## Full Report

See the full report for detailed EDA, preprocessing rationale, hyperparameter search grids, ensemble architecture, and per-class metrics.
