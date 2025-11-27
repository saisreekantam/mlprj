"""
Comprehensive Algorithm Benchmarking
Compares Logistic Regression, Tree-based models, and Boosting algorithms
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform, loguniform
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("COMPREHENSIVE ALGORITHM BENCHMARKING")
print("="*80)

# Load processed features
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

# Prepare datasets
X_train_scaled = data['scaled_data']['standard']['train_full']  # For LR
X_train_unscaled = data['X_train_full']  # For tree-based models
y_train = data['y_train']

# Encode labels for XGBoost/LightGBM (they need numeric labels)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

print(f"\nTraining data shape: {X_train_scaled.shape}")
print(f"Target distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count} samples ({count/len(y_train)*100:.1f}%)")

# Define Macro F1 scorer (for string labels)
def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

# Define Macro F1 scorer (for numeric labels)
def macro_f1_numeric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

macro_f1_scorer = make_scorer(macro_f1_multiclass)
macro_f1_scorer_numeric = make_scorer(macro_f1_numeric)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store all results
all_results = {}

# ============================================================================
# 1. LOGISTIC REGRESSION
# ============================================================================
print("\n" + "="*80)
print("1. LOGISTIC REGRESSION")
print("="*80)

param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000],
    'class_weight': ['balanced', None]
}

lr = LogisticRegression(random_state=42, multi_class='multinomial')

start_time = time.time()
grid_lr = GridSearchCV(
    lr, param_grid_lr,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=2
)

print("\nRunning Grid Search...")
grid_lr.fit(X_train_scaled, y_train)
lr_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"Best parameters: {grid_lr.best_params_}")
print(f"Best CV Macro F1 Score: {grid_lr.best_score_:.6f}")
print(f"Training time: {lr_time:.2f}s")
print(f"{'='*60}")

all_results['Logistic Regression'] = {
    'model': grid_lr.best_estimator_,
    'score': grid_lr.best_score_,
    'params': grid_lr.best_params_,
    'time': lr_time,
    'uses_scaling': True
}

# ============================================================================
# 2. RANDOM FOREST
# ============================================================================
print("\n" + "="*80)
print("2. RANDOM FOREST")
print("="*80)

param_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

start_time = time.time()
random_rf = RandomizedSearchCV(
    rf, param_grid_rf,
    n_iter=50,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("\nRunning Randomized Search...")
random_rf.fit(X_train_unscaled, y_train)
rf_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"Best parameters: {random_rf.best_params_}")
print(f"Best CV Macro F1 Score: {random_rf.best_score_:.6f}")
print(f"Training time: {rf_time:.2f}s")
print(f"{'='*60}")

all_results['Random Forest'] = {
    'model': random_rf.best_estimator_,
    'score': random_rf.best_score_,
    'params': random_rf.best_params_,
    'time': rf_time,
    'uses_scaling': False
}

# ============================================================================
# 3. EXTRA TREES
# ============================================================================
print("\n" + "="*80)
print("3. EXTRA TREES")
print("="*80)

param_grid_et = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}

et = ExtraTreesClassifier(random_state=42, n_jobs=-1)

start_time = time.time()
random_et = RandomizedSearchCV(
    et, param_grid_et,
    n_iter=50,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("\nRunning Randomized Search...")
random_et.fit(X_train_unscaled, y_train)
et_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"Best parameters: {random_et.best_params_}")
print(f"Best CV Macro F1 Score: {random_et.best_score_:.6f}")
print(f"Training time: {et_time:.2f}s")
print(f"{'='*60}")

all_results['Extra Trees'] = {
    'model': random_et.best_estimator_,
    'score': random_et.best_score_,
    'params': random_et.best_params_,
    'time': et_time,
    'uses_scaling': False
}

# ============================================================================
# 4. GRADIENT BOOSTING
# ============================================================================
print("\n" + "="*80)
print("4. GRADIENT BOOSTING")
print("="*80)

param_grid_gb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

gb = GradientBoostingClassifier(random_state=42)

start_time = time.time()
random_gb = RandomizedSearchCV(
    gb, param_grid_gb,
    n_iter=50,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("\nRunning Randomized Search...")
random_gb.fit(X_train_unscaled, y_train)
gb_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"Best parameters: {random_gb.best_params_}")
print(f"Best CV Macro F1 Score: {random_gb.best_score_:.6f}")
print(f"Training time: {gb_time:.2f}s")
print(f"{'='*60}")

all_results['Gradient Boosting'] = {
    'model': random_gb.best_estimator_,
    'score': random_gb.best_score_,
    'params': random_gb.best_params_,
    'time': gb_time,
    'uses_scaling': False
}

# ============================================================================
# 5. ADABOOST
# ============================================================================
print("\n" + "="*80)
print("5. ADABOOST")
print("="*80)

param_grid_ada = {
    'n_estimators': [50, 100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    'algorithm': ['SAMME', 'SAMME.R']
}

ada = AdaBoostClassifier(random_state=42)

start_time = time.time()
grid_ada = GridSearchCV(
    ada, param_grid_ada,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=2
)

print("\nRunning Grid Search...")
grid_ada.fit(X_train_unscaled, y_train)
ada_time = time.time() - start_time

print(f"\n{'='*60}")
print(f"Best parameters: {grid_ada.best_params_}")
print(f"Best CV Macro F1 Score: {grid_ada.best_score_:.6f}")
print(f"Training time: {ada_time:.2f}s")
print(f"{'='*60}")

all_results['AdaBoost'] = {
    'model': grid_ada.best_estimator_,
    'score': grid_ada.best_score_,
    'params': grid_ada.best_params_,
    'time': ada_time,
    'uses_scaling': False
}

# ============================================================================
# 6. XGBOOST (if available)
# ============================================================================
print("\n" + "="*80)
print("6. XGBOOST")
print("="*80)

try:
    import xgboost as xgb
    
    param_grid_xgb = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.3],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
    
    start_time = time.time()
    random_xgb = RandomizedSearchCV(
        xgb_model, param_grid_xgb,
        n_iter=50,
        cv=skf,
        scoring=macro_f1_scorer_numeric,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    print("\nRunning Randomized Search...")
    random_xgb.fit(X_train_unscaled, y_train_encoded)
    xgb_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Best parameters: {random_xgb.best_params_}")
    print(f"Best CV Macro F1 Score: {random_xgb.best_score_:.6f}")
    print(f"Training time: {xgb_time:.2f}s")
    print(f"{'='*60}")
    
    all_results['XGBoost'] = {
        'model': random_xgb.best_estimator_,
        'score': random_xgb.best_score_,
        'params': random_xgb.best_params_,
        'time': xgb_time,
        'uses_scaling': False
    }
except ImportError:
    print("XGBoost not installed. Skipping...")

# ============================================================================
# 7. LIGHTGBM (if available)
# ============================================================================
print("\n" + "="*80)
print("7. LIGHTGBM")
print("="*80)

try:
    import lightgbm as lgb
    
    param_grid_lgb = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [20, 31, 50, 80],
        'max_depth': [-1, 5, 10, 15],
        'min_child_samples': [5, 10, 20, 30],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    lgb_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    
    start_time = time.time()
    random_lgb = RandomizedSearchCV(
        lgb_model, param_grid_lgb,
        n_iter=50,
        cv=skf,
        scoring=macro_f1_scorer_numeric,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    print("\nRunning Randomized Search...")
    random_lgb.fit(X_train_unscaled, y_train_encoded)
    lgb_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Best parameters: {random_lgb.best_params_}")
    print(f"Best CV Macro F1 Score: {random_lgb.best_score_:.6f}")
    print(f"Training time: {lgb_time:.2f}s")
    print(f"{'='*60}")
    
    all_results['LightGBM'] = {
        'model': random_lgb.best_estimator_,
        'score': random_lgb.best_score_,
        'params': random_lgb.best_params_,
        'time': lgb_time,
        'uses_scaling': False
    }
except ImportError:
    print("LightGBM not installed. Skipping...")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE BENCHMARKING RESULTS")
print("="*80)

# Create results DataFrame
results_data = []
for name, result in all_results.items():
    results_data.append({
        'Algorithm': name,
        'CV F1 Score': result['score'],
        'Training Time (s)': result['time'],
        'Uses Scaling': result['uses_scaling']
    })

results_df = pd.DataFrame(results_data)
results_df = results_df.sort_values('CV F1 Score', ascending=False)

print("\n" + "-"*80)
print(f"{'Rank':<6} {'Algorithm':<20} {'CV F1 Score':<15} {'Time (s)':<12} {'Scaling':<10}")
print("-"*80)
for idx, row in results_df.iterrows():
    rank = results_df.index.get_loc(idx) + 1
    print(f"{rank:<6} {row['Algorithm']:<20} {row['CV F1 Score']:<15.6f} "
          f"{row['Training Time (s)']:<12.2f} {str(row['Uses Scaling']):<10}")
print("-"*80)

# Best model
best_algo = results_df.iloc[0]['Algorithm']
best_score = results_df.iloc[0]['CV F1 Score']
best_model = all_results[best_algo]['model']

print(f"\n{'='*80}")
print(f"🏆 BEST ALGORITHM: {best_algo}")
print(f"{'='*80}")
print(f"CV Macro F1 Score: {best_score:.6f}")
print(f"Best Parameters: {all_results[best_algo]['params']}")
print(f"Training Time: {all_results[best_algo]['time']:.2f}s")
print(f"Uses Feature Scaling: {all_results[best_algo]['uses_scaling']}")
print(f"{'='*80}")

# Save all results
benchmark_data = {
    'all_results': all_results,
    'best_algorithm': best_algo,
    'best_score': best_score,
    'best_model': best_model,
    'results_df': results_df
}

with open('algorithm_benchmarking_results.pkl', 'wb') as f:
    pickle.dump(benchmark_data, f)

# Save results CSV
results_df.to_csv('algorithm_benchmarking_summary.csv', index=False)

print("\n✓ Saved algorithm_benchmarking_results.pkl")
print("✓ Saved algorithm_benchmarking_summary.csv")

# Compare with RBF SVM
print("\n" + "="*80)
print("COMPARISON WITH RBF SVM (from previous grid search)")
print("="*80)
rbf_score = 0.986312
print(f"\nRBF SVM (C=10, gamma=0.5): {rbf_score:.6f}")
print(f"Best Tree/LR Algorithm ({best_algo}): {best_score:.6f}")

if best_score > rbf_score:
    print(f"\n🎉 {best_algo} WINS by {(best_score - rbf_score):.6f}!")
elif rbf_score > best_score:
    print(f"\n🎉 RBF SVM WINS by {(rbf_score - best_score):.6f}!")
else:
    print(f"\n🤝 It's a tie!")

print("\nDone!")
