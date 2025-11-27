"""
Comprehensive GridSearchCV for Fast Algorithms (Excluding SVM)
Focus on Logistic Regression, Tree Ensembles, and Neural Networks
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                               GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("COMPREHENSIVE GRIDSEARCHCV - FAST ALGORITHMS")
print("="*80)
print("Skipping SVM (too slow), focusing on other high-performing models")

# Load data
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_full = data['X_train_full']
y_train = data['y_train']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)

# 10-fold CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

grid_results = {}

# 1. LOGISTIC REGRESSION - Comprehensive Grid
print("\n" + "="*80)
print("1. LOGISTIC REGRESSION - Full GridSearch")
print("="*80)

param_grid_lr = {
    'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50],
    'solver': ['lbfgs', 'saga'],
    'penalty': ['l2'],
    'class_weight': ['balanced', None],
    'max_iter': [2000]
}

lr = LogisticRegression(random_state=42)
grid_lr = GridSearchCV(lr, param_grid_lr, cv=skf, scoring=macro_f1_scorer, 
                       n_jobs=-1, verbose=2)
grid_lr.fit(X_train_scaled, y_train)

print(f"\nBest params: {grid_lr.best_params_}")
print(f"Best 10-Fold CV F1: {grid_lr.best_score_:.4f}")
grid_results['Logistic Regression'] = {
    'model': grid_lr.best_estimator_,
    'score': grid_lr.best_score_,
    'params': grid_lr.best_params_,
    'scaled': True
}

# 2. EXTRA TREES - Comprehensive Grid
print("\n" + "="*80)
print("2. EXTRA TREES - Full GridSearch")
print("="*80)

param_grid_et = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}

et = ExtraTreesClassifier(random_state=42, n_jobs=-1)
grid_et = GridSearchCV(et, param_grid_et, cv=skf, scoring=macro_f1_scorer,
                       n_jobs=-1, verbose=2)
grid_et.fit(X_train_full, y_train)

print(f"\nBest params: {grid_et.best_params_}")
print(f"Best 10-Fold CV F1: {grid_et.best_score_:.4f}")
grid_results['Extra Trees'] = {
    'model': grid_et.best_estimator_,
    'score': grid_et.best_score_,
    'params': grid_et.best_params_,
    'scaled': False
}

# 3. RANDOM FOREST - Comprehensive Grid
print("\n" + "="*80)
print("3. RANDOM FOREST - Full GridSearch")
print("="*80)

param_grid_rf = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=skf, scoring=macro_f1_scorer,
                       n_jobs=-1, verbose=2)
grid_rf.fit(X_train_full, y_train)

print(f"\nBest params: {grid_rf.best_params_}")
print(f"Best 10-Fold CV F1: {grid_rf.best_score_:.4f}")
grid_results['Random Forest'] = {
    'model': grid_rf.best_estimator_,
    'score': grid_rf.best_score_,
    'params': grid_rf.best_params_,
    'scaled': False
}

# 4. GRADIENT BOOSTING - Comprehensive Grid
print("\n" + "="*80)
print("4. GRADIENT BOOSTING - Full GridSearch")
print("="*80)

param_grid_gb = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [5, 10, 15],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

gb = GradientBoostingClassifier(random_state=42)
grid_gb = GridSearchCV(gb, param_grid_gb, cv=skf, scoring=macro_f1_scorer,
                       n_jobs=-1, verbose=2)
grid_gb.fit(X_train_full, y_train)

print(f"\nBest params: {grid_gb.best_params_}")
print(f"Best 10-Fold CV F1: {grid_gb.best_score_:.4f}")
grid_results['Gradient Boosting'] = {
    'model': grid_gb.best_estimator_,
    'score': grid_gb.best_score_,
    'params': grid_gb.best_params_,
    'scaled': False
}

# 5. ADABOOST - Comprehensive Grid
print("\n" + "="*80)
print("5. ADABOOST - Full GridSearch")
print("="*80)

param_grid_ada = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    'algorithm': ['SAMME']
}

ada = AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(max_depth=3))
grid_ada = GridSearchCV(ada, param_grid_ada, cv=skf, scoring=macro_f1_scorer,
                        n_jobs=-1, verbose=2)
grid_ada.fit(X_train_full, y_train)

print(f"\nBest params: {grid_ada.best_params_}")
print(f"Best 10-Fold CV F1: {grid_ada.best_score_:.4f}")
grid_results['AdaBoost'] = {
    'model': grid_ada.best_estimator_,
    'score': grid_ada.best_score_,
    'params': grid_ada.best_params_,
    'scaled': False
}

# Summary
print("\n" + "="*80)
print("GRID SEARCH RESULTS SUMMARY")
print("="*80)

sorted_results = sorted(grid_results.items(), key=lambda x: x[1]['score'], reverse=True)
for name, data in sorted_results:
    print(f"{name:25s}: {data['score']:.4f}")
    print(f"   Params: {data['params']}")

# Save results
with open('gridsearch_results.pkl', 'wb') as f:
    pickle.dump({
        'models': grid_results,
        'scaler': scaler
    }, f)

print("\n✓ Saved gridsearch_results.pkl")

# Also save a CSV summary
summary_df = pd.DataFrame([
    {'Model': name, 'CV_F1_Score': data['score'], 'Best_Params': str(data['params'])}
    for name, data in sorted_results
])
summary_df.to_csv('gridsearch_summary.csv', index=False)
print("✓ Saved gridsearch_summary.csv")

print("="*80)
