"""
Comprehensive Grid Search for SVM RBF Kernel
Testing various combinations of C and gamma parameters
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("RBF KERNEL GRID SEARCH - COMPREHENSIVE PARAMETER TUNING")
print("="*80)

# Load processed features
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_scaled = data['scaled_data']['standard']['train_full']
y_train = data['y_train']

print(f"\nTraining data shape: {X_train_scaled.shape}")
print(f"Target distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count} samples ({count/len(y_train)*100:.1f}%)")

# Define Macro F1 scorer
def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*80)
print("GRID SEARCH 1: COARSE GRID (Broad Range)")
print("="*80)

# Coarse grid with wider range
param_grid_coarse = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 'scale', 'auto'],
    'kernel': ['rbf']
}

print(f"\nParameter grid:")
print(f"  C: {param_grid_coarse['C']}")
print(f"  gamma: {param_grid_coarse['gamma']}")
print(f"  Total combinations: {len(param_grid_coarse['C']) * len(param_grid_coarse['gamma'])}")

svm_rbf_coarse = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

grid_coarse = GridSearchCV(
    svm_rbf_coarse, 
    param_grid_coarse,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

print("\nRunning Grid Search 1...")
grid_coarse.fit(X_train_scaled, y_train)

print(f"\n{'='*80}")
print(f"RESULTS - COARSE GRID")
print(f"{'='*80}")
print(f"Best parameters: {grid_coarse.best_params_}")
print(f"Best CV Macro F1 Score: {grid_coarse.best_score_:.6f}")

# Show top 10 results
results_df_coarse = pd.DataFrame(grid_coarse.cv_results_)
results_df_coarse = results_df_coarse.sort_values('rank_test_score')
print(f"\nTop 10 parameter combinations:")
print("-"*80)
for idx, row in results_df_coarse.head(10).iterrows():
    print(f"Rank {int(row['rank_test_score'])}: C={row['param_C']}, gamma={row['param_gamma']}, "
          f"Score={row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")

print("\n" + "="*80)
print("GRID SEARCH 2: FINE GRID (Refined Range)")
print("="*80)

# Based on best result, create a fine grid around the best parameters
best_C = grid_coarse.best_params_['C']
best_gamma = grid_coarse.best_params_['gamma']

# Handle 'scale' and 'auto' gamma values
if isinstance(best_gamma, str):
    if best_gamma == 'scale':
        best_gamma = 1 / (X_train_scaled.shape[1] * X_train_scaled.var())
    elif best_gamma == 'auto':
        best_gamma = 1 / X_train_scaled.shape[1]

# Create fine grid around best parameters
param_grid_fine = {
    'C': [best_C * 0.1, best_C * 0.5, best_C, best_C * 2, best_C * 5, best_C * 10],
    'gamma': [best_gamma * 0.1, best_gamma * 0.5, best_gamma, best_gamma * 2, best_gamma * 5, best_gamma * 10],
    'kernel': ['rbf']
}

print(f"\nRefining around best parameters:")
print(f"  Best C from coarse search: {grid_coarse.best_params_['C']}")
print(f"  Best gamma from coarse search: {grid_coarse.best_params_['gamma']}")
print(f"\nFine parameter grid:")
print(f"  C: {[f'{c:.4f}' for c in param_grid_fine['C']]}")
print(f"  gamma: {[f'{g:.6f}' for g in param_grid_fine['gamma']]}")
print(f"  Total combinations: {len(param_grid_fine['C']) * len(param_grid_fine['gamma'])}")

svm_rbf_fine = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

grid_fine = GridSearchCV(
    svm_rbf_fine, 
    param_grid_fine,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

print("\nRunning Grid Search 2...")
grid_fine.fit(X_train_scaled, y_train)

print(f"\n{'='*80}")
print(f"RESULTS - FINE GRID")
print(f"{'='*80}")
print(f"Best parameters: {grid_fine.best_params_}")
print(f"Best CV Macro F1 Score: {grid_fine.best_score_:.6f}")

# Show top 10 results
results_df_fine = pd.DataFrame(grid_fine.cv_results_)
results_df_fine = results_df_fine.sort_values('rank_test_score')
print(f"\nTop 10 parameter combinations:")
print("-"*80)
for idx, row in results_df_fine.head(10).iterrows():
    print(f"Rank {int(row['rank_test_score'])}: C={row['param_C']:.4f}, gamma={row['param_gamma']:.6f}, "
          f"Score={row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")

print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

comparison = {
    'Coarse Grid Search': grid_coarse.best_score_,
    'Fine Grid Search': grid_fine.best_score_
}

print(f"\nCoarse Grid Best Score: {grid_coarse.best_score_:.6f}")
print(f"  Parameters: {grid_coarse.best_params_}")
print(f"\nFine Grid Best Score: {grid_fine.best_score_:.6f}")
print(f"  Parameters: {grid_fine.best_params_}")

# Select overall best
if grid_fine.best_score_ >= grid_coarse.best_score_:
    best_model = grid_fine.best_estimator_
    best_params = grid_fine.best_params_
    best_score = grid_fine.best_score_
    search_type = "Fine Grid"
else:
    best_model = grid_coarse.best_estimator_
    best_params = grid_coarse.best_params_
    best_score = grid_coarse.best_score_
    search_type = "Coarse Grid"

print(f"\n{'='*80}")
print(f"OVERALL BEST MODEL: {search_type}")
print(f"{'='*80}")
print(f"Best parameters: {best_params}")
print(f"Best CV Macro F1 Score: {best_score:.6f}")
print(f"{'='*80}")

# Save best model and all results
model_data = {
    'best_model': best_model,
    'best_params': best_params,
    'best_score': best_score,
    'search_type': search_type,
    'coarse_results': {
        'best_params': grid_coarse.best_params_,
        'best_score': grid_coarse.best_score_,
        'cv_results': results_df_coarse.head(20).to_dict()
    },
    'fine_results': {
        'best_params': grid_fine.best_params_,
        'best_score': grid_fine.best_score_,
        'cv_results': results_df_fine.head(20).to_dict()
    }
}

with open('rbf_grid_search_results.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✓ Saved rbf_grid_search_results.pkl")
print("\nDone!")
