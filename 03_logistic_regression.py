"""
Logistic Regression Models with Hyperparameter Tuning
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, make_scorer, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("LOGISTIC REGRESSION MODELS")
print("="*80)

# Load processed features
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_raw = data['X_train_raw']
X_train_full = data['X_train_full']
y_train = data['y_train']

# Get scaled versions
X_train_scaled = data['scaled_data']['standard']['train_full']

# Define Macro F1 scorer
def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*80)
print("1. LOGISTIC REGRESSION - MULTINOMIAL (L2)")
print("="*80)

param_grid_l2 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'newton-cg', 'sag'],
    'max_iter': [1000]
}

lr_l2 = LogisticRegression(
    multi_class='multinomial',
    penalty='l2',
    class_weight='balanced',
    random_state=42
)

grid_l2 = GridSearchCV(
    lr_l2, param_grid_l2,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_l2.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_l2.best_params_}")
print(f"Best CV Macro F1 Score: {grid_l2.best_score_:.4f}")

print("\n" + "="*80)
print("2. LOGISTIC REGRESSION - MULTINOMIAL (L1)")
print("="*80)

param_grid_l1 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['saga'],
    'max_iter': [1000, 2000]
}

lr_l1 = LogisticRegression(
    multi_class='multinomial',
    penalty='l1',
    class_weight='balanced',
    random_state=42
)

grid_l1 = GridSearchCV(
    lr_l1, param_grid_l1,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_l1.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_l1.best_params_}")
print(f"Best CV Macro F1 Score: {grid_l1.best_score_:.4f}")

print("\n" + "="*80)
print("3. LOGISTIC REGRESSION - ELASTICNET")
print("="*80)

param_grid_elastic = {
    'C': [0.01, 0.1, 1, 10],
    'l1_ratio': [0.2, 0.5, 0.8],
    'solver': ['saga'],
    'max_iter': [2000]
}

lr_elastic = LogisticRegression(
    multi_class='multinomial',
    penalty='elasticnet',
    class_weight='balanced',
    random_state=42
)

grid_elastic = GridSearchCV(
    lr_elastic, param_grid_elastic,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_elastic.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_elastic.best_params_}")
print(f"Best CV Macro F1 Score: {grid_elastic.best_score_:.4f}")

print("\n" + "="*80)
print("4. LOGISTIC REGRESSION - ONE-VS-REST")
print("="*80)

param_grid_ovr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000]
}

lr_ovr = LogisticRegression(
    multi_class='ovr',
    penalty='l2',
    class_weight='balanced',
    random_state=42
)

grid_ovr = GridSearchCV(
    lr_ovr, param_grid_ovr,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_ovr.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_ovr.best_params_}")
print(f"Best CV Macro F1 Score: {grid_ovr.best_score_:.4f}")

# Compare all models
print("\n" + "="*80)
print("LOGISTIC REGRESSION SUMMARY")
print("="*80)

results = {
    'Multinomial L2': grid_l2.best_score_,
    'Multinomial L1': grid_l1.best_score_,
    'ElasticNet': grid_elastic.best_score_,
    'One-vs-Rest': grid_ovr.best_score_
}

for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:20s}: {score:.4f}")

# Select best model
best_lr_name = max(results, key=results.get)
best_lr_score = results[best_lr_name]

if best_lr_name == 'Multinomial L2':
    best_lr_model = grid_l2.best_estimator_
elif best_lr_name == 'Multinomial L1':
    best_lr_model = grid_l1.best_estimator_
elif best_lr_name == 'ElasticNet':
    best_lr_model = grid_elastic.best_estimator_
else:
    best_lr_model = grid_ovr.best_estimator_

print(f"\n{'='*80}")
print(f"BEST LOGISTIC REGRESSION: {best_lr_name}")
print(f"CV Macro F1 Score: {best_lr_score:.4f}")
print(f"{'='*80}")

# Save best model
model_data = {
    'model': best_lr_model,
    'model_name': best_lr_name,
    'cv_score': best_lr_score,
    'all_results': results,
    'feature_type': 'scaled_full',
    'scaler_type': 'standard'
}

with open('best_logistic_regression.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✓ Saved best_logistic_regression.pkl")
