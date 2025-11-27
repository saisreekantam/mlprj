"""
SVM Models with Multiple Kernels and Hyperparameter Tuning
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import loguniform, uniform
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("SVM MODELS")
print("="*80)

# Load processed features
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_scaled = data['scaled_data']['standard']['train_full']
y_train = data['y_train']

# Define Macro F1 scorer
def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*80)
print("1. LINEAR SVM")
print("="*80)

param_grid_linear = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear']
}

svm_linear = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

grid_linear = GridSearchCV(
    svm_linear, param_grid_linear,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_linear.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_linear.best_params_}")
print(f"Best CV Macro F1 Score: {grid_linear.best_score_:.4f}")

print("\n" + "="*80)
print("2. RBF SVM (Radial Basis Function)")
print("="*80)

param_dist_rbf = {
    'C': loguniform(0.1, 100),
    'gamma': loguniform(0.001, 10),
    'kernel': ['rbf']
}

svm_rbf = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

random_rbf = RandomizedSearchCV(
    svm_rbf, param_dist_rbf,
    n_iter=50,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_rbf.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {random_rbf.best_params_}")
print(f"Best CV Macro F1 Score: {random_rbf.best_score_:.4f}")

print("\n" + "="*80)
print("3. POLYNOMIAL SVM")
print("="*80)

param_grid_poly = {
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4],
    'kernel': ['poly'],
    'coef0': [0, 1]
}

svm_poly = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

grid_poly = GridSearchCV(
    svm_poly, param_grid_poly,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_poly.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_poly.best_params_}")
print(f"Best CV Macro F1 Score: {grid_poly.best_score_:.4f}")

print("\n" + "="*80)
print("4. SIGMOID SVM")
print("="*80)

param_dist_sigmoid = {
    'C': loguniform(0.1, 100),
    'gamma': loguniform(0.001, 1),
    'kernel': ['sigmoid'],
    'coef0': uniform(0, 1)
}

svm_sigmoid = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

random_sigmoid = RandomizedSearchCV(
    svm_sigmoid, param_dist_sigmoid,
    n_iter=30,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_sigmoid.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {random_sigmoid.best_params_}")
print(f"Best CV Macro F1 Score: {random_sigmoid.best_score_:.4f}")

# Compare all SVM models
print("\n" + "="*80)
print("SVM MODELS SUMMARY")
print("="*80)

results = {
    'Linear SVM': grid_linear.best_score_,
    'RBF SVM': random_rbf.best_score_,
    'Polynomial SVM': grid_poly.best_score_,
    'Sigmoid SVM': random_sigmoid.best_score_
}

for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:20s}: {score:.4f}")

# Select best SVM model
best_svm_name = max(results, key=results.get)
best_svm_score = results[best_svm_name]

if best_svm_name == 'Linear SVM':
    best_svm_model = grid_linear.best_estimator_
elif best_svm_name == 'RBF SVM':
    best_svm_model = random_rbf.best_estimator_
elif best_svm_name == 'Polynomial SVM':
    best_svm_model = grid_poly.best_estimator_
else:
    best_svm_model = random_sigmoid.best_estimator_

print(f"\n{'='*80}")
print(f"BEST SVM MODEL: {best_svm_name}")
print(f"CV Macro F1 Score: {best_svm_score:.4f}")
print(f"{'='*80}")

# Save best model
model_data = {
    'model': best_svm_model,
    'model_name': best_svm_name,
    'cv_score': best_svm_score,
    'all_results': results,
    'feature_type': 'scaled_full',
    'scaler_type': 'standard'
}

with open('best_svm.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✓ Saved best_svm.pkl")
