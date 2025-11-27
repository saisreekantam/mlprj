"""
Focused GridSearchCV for SVM and MLP
Parameters chosen based on data characteristics:
- Small dataset (1444 samples)
- 10 engineered features
- 3 classes with imbalance
- Need for good generalization (test scores lower than CV)
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("FOCUSED GRIDSEARCHCV - SVM & MLP")
print("="*80)
print("Data characteristics: 1444 samples, 10 features, 3 classes")
print("Strategy: Conservative parameters for better generalization")

# Load data
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_full = data['X_train_full']
y_train = data['y_train']

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)

# Use 10-fold CV for robust evaluation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

results = {}

# ============================================================================
# 1. SVM - RBF Kernel (Most promising)
# ============================================================================
print("\n" + "="*80)
print("1. SVM with RBF Kernel - Comprehensive Grid")
print("="*80)
print("Testing multiple C and gamma combinations...")

# For small dataset, moderate C and gamma values work best
param_grid_svm_rbf = {
    'C': [0.5, 1, 2, 5, 10, 15, 20, 30, 50],
    'gamma': [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
    'class_weight': ['balanced'],
    'kernel': ['rbf']
}

svm_rbf = SVC(probability=True, random_state=42)
grid_svm_rbf = GridSearchCV(
    svm_rbf, param_grid_svm_rbf, cv=skf, 
    scoring=macro_f1_scorer, n_jobs=-1, verbose=2
)

print("Starting SVM RBF grid search...")
grid_svm_rbf.fit(X_train_scaled, y_train)

print(f"\nBest params: {grid_svm_rbf.best_params_}")
print(f"Best 10-Fold CV F1: {grid_svm_rbf.best_score_:.4f}")

# Get top 5 parameter combinations
cv_results = pd.DataFrame(grid_svm_rbf.cv_results_)
top_5_svm = cv_results.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 5 SVM RBF configurations:")
for idx, row in top_5_svm.iterrows():
    print(f"  F1: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f}) - {row['params']}")

results['SVM_RBF'] = {
    'model': grid_svm_rbf.best_estimator_,
    'score': grid_svm_rbf.best_score_,
    'params': grid_svm_rbf.best_params_,
    'std': cv_results.loc[grid_svm_rbf.best_index_, 'std_test_score'],
    'scaled': True
}

# ============================================================================
# 2. SVM - Linear Kernel (Faster, often good for smaller datasets)
# ============================================================================
print("\n" + "="*80)
print("2. SVM with Linear Kernel")
print("="*80)

param_grid_svm_linear = {
    'C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
    'class_weight': ['balanced'],
    'kernel': ['linear']
}

svm_linear = SVC(probability=True, random_state=42)
grid_svm_linear = GridSearchCV(
    svm_linear, param_grid_svm_linear, cv=skf,
    scoring=macro_f1_scorer, n_jobs=-1, verbose=2
)

print("Starting SVM Linear grid search...")
grid_svm_linear.fit(X_train_scaled, y_train)

print(f"\nBest params: {grid_svm_linear.best_params_}")
print(f"Best 10-Fold CV F1: {grid_svm_linear.best_score_:.4f}")

results['SVM_Linear'] = {
    'model': grid_svm_linear.best_estimator_,
    'score': grid_svm_linear.best_score_,
    'params': grid_svm_linear.best_params_,
    'scaled': True
}

# ============================================================================
# 3. MLP - Small architectures (better for small datasets)
# ============================================================================
print("\n" + "="*80)
print("3. Multi-layer Perceptron (MLP) Neural Network")
print("="*80)
print("Testing shallow and moderate architectures...")

# For small dataset, shallow networks with strong regularization
param_grid_mlp = {
    'hidden_layer_sizes': [
        # Single layer
        (20,), (30,), (40,), (50,), (60,), (70,), (80,),
        # Two layers
        (30, 15), (40, 20), (50, 25), (60, 30), (70, 35),
        # Three layers
        (40, 20, 10), (50, 25, 12), (60, 30, 15)
    ],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5],  # L2 penalty
    'learning_rate_init': [0.0005, 0.001, 0.002],
    'solver': ['adam', 'lbfgs'],
    'early_stopping': [True],
    'validation_fraction': [0.15],
    'max_iter': [1000]
}

mlp = MLPClassifier(random_state=42)
grid_mlp = GridSearchCV(
    mlp, param_grid_mlp, cv=skf,
    scoring=macro_f1_scorer, n_jobs=-1, verbose=2
)

print("Starting MLP grid search...")
grid_mlp.fit(X_train_scaled, y_train)

print(f"\nBest params: {grid_mlp.best_params_}")
print(f"Best 10-Fold CV F1: {grid_mlp.best_score_:.4f}")

# Get top 5 MLP configurations
cv_results_mlp = pd.DataFrame(grid_mlp.cv_results_)
top_5_mlp = cv_results_mlp.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 5 MLP configurations:")
for idx, row in top_5_mlp.iterrows():
    print(f"  F1: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
    print(f"      {row['params']}")

results['MLP'] = {
    'model': grid_mlp.best_estimator_,
    'score': grid_mlp.best_score_,
    'params': grid_mlp.best_params_,
    'std': cv_results_mlp.loc[grid_mlp.best_index_, 'std_test_score'],
    'scaled': True
}

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
for name, data in sorted_results:
    std = data.get('std', 0)
    print(f"{name:15s}: {data['score']:.4f} (+/- {std:.4f})")
    print(f"   Best params: {data['params']}")
    print()

# Save results
with open('svm_mlp_gridsearch.pkl', 'wb') as f:
    pickle.dump({
        'models': results,
        'scaler': scaler,
        'cv_folds': 10
    }, f)

print("✓ Saved svm_mlp_gridsearch.pkl")

# Create summary CSV
summary_data = []
for name, data in sorted_results:
    summary_data.append({
        'Model': name,
        'CV_F1_Score': data['score'],
        'Std_Dev': data.get('std', 0),
        'Best_Params': str(data['params'])
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('svm_mlp_summary.csv', index=False)
print("✓ Saved svm_mlp_summary.csv")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Review the best SVM and MLP models above")
print("2. Generate test predictions with these models")
print("3. Then run grid search for other algorithms if needed")
print("="*80)
