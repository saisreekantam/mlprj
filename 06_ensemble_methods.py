"""
Ensemble Methods (Tree-based and Boosting) with Hyperparameter Tuning
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("ENSEMBLE METHODS (TREE-BASED & BOOSTING)")
print("="*80)

# Load processed features
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

# Use raw features for tree-based models (they don't need scaling)
X_train = data['X_train_full']
y_train = data['y_train']

# Define Macro F1 scorer
def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*80)
print("1. RANDOM FOREST")
print("="*80)

param_dist_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25, 30, None],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

random_rf = RandomizedSearchCV(
    rf, param_dist_rf,
    n_iter=50,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_rf.fit(X_train, y_train)

print(f"\nBest parameters: {random_rf.best_params_}")
print(f"Best CV Macro F1 Score: {random_rf.best_score_:.4f}")

print("\n" + "="*80)
print("2. EXTRA TREES")
print("="*80)

param_dist_et = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25, 30, None],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]
}

et = ExtraTreesClassifier(random_state=42, n_jobs=-1)

random_et = RandomizedSearchCV(
    et, param_dist_et,
    n_iter=50,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_et.fit(X_train, y_train)

print(f"\nBest parameters: {random_et.best_params_}")
print(f"Best CV Macro F1 Score: {random_et.best_score_:.4f}")

print("\n" + "="*80)
print("3. GRADIENT BOOSTING")
print("="*80)

param_dist_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

gb = GradientBoostingClassifier(random_state=42)

random_gb = RandomizedSearchCV(
    gb, param_dist_gb,
    n_iter=40,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_gb.fit(X_train, y_train)

print(f"\nBest parameters: {random_gb.best_params_}")
print(f"Best CV Macro F1 Score: {random_gb.best_score_:.4f}")

print("\n" + "="*80)
print("4. XGBOOST")
print("="*80)

try:
    import xgboost as xgb
    
    param_dist_xgb = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': randint(1, 7),
        'gamma': uniform(0, 0.5),
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
    
    random_xgb = RandomizedSearchCV(
        xgb_model, param_dist_xgb,
        n_iter=60,
        cv=skf,
        scoring=macro_f1_scorer,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_xgb.fit(X_train, y_train)
    
    print(f"\nBest parameters: {random_xgb.best_params_}")
    print(f"Best CV Macro F1 Score: {random_xgb.best_score_:.4f}")
    xgb_score = random_xgb.best_score_
    xgb_model_best = random_xgb.best_estimator_
except ImportError:
    print("XGBoost not installed. Skipping...")
    xgb_score = 0.0
    xgb_model_best = None

print("\n" + "="*80)
print("5. LIGHTGBM")
print("="*80)

try:
    import lightgbm as lgb
    
    param_dist_lgb = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': randint(20, 100),
        'max_depth': [-1, 5, 10, 15, 20],
        'min_child_samples': randint(5, 50),
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }
    
    lgb_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    
    random_lgb = RandomizedSearchCV(
        lgb_model, param_dist_lgb,
        n_iter=60,
        cv=skf,
        scoring=macro_f1_scorer,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_lgb.fit(X_train, y_train)
    
    print(f"\nBest parameters: {random_lgb.best_params_}")
    print(f"Best CV Macro F1 Score: {random_lgb.best_score_:.4f}")
    lgb_score = random_lgb.best_score_
    lgb_model_best = random_lgb.best_estimator_
except ImportError:
    print("LightGBM not installed. Skipping...")
    lgb_score = 0.0
    lgb_model_best = None

print("\n" + "="*80)
print("6. CATBOOST")
print("="*80)

try:
    from catboost import CatBoostClassifier
    
    param_dist_cat = {
        'iterations': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': uniform(1, 10),
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.7, 0.8, 0.9, 1.0]
    }
    
    cat_model = CatBoostClassifier(random_state=42, verbose=0)
    
    random_cat = RandomizedSearchCV(
        cat_model, param_dist_cat,
        n_iter=50,
        cv=skf,
        scoring=macro_f1_scorer,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_cat.fit(X_train, y_train)
    
    print(f"\nBest parameters: {random_cat.best_params_}")
    print(f"Best CV Macro F1 Score: {random_cat.best_score_:.4f}")
    cat_score = random_cat.best_score_
    cat_model_best = random_cat.best_estimator_
except ImportError:
    print("CatBoost not installed. Skipping...")
    cat_score = 0.0
    cat_model_best = None

# Compare all ensemble models
print("\n" + "="*80)
print("ENSEMBLE METHODS SUMMARY")
print("="*80)

results = {
    'Random Forest': random_rf.best_score_,
    'Extra Trees': random_et.best_score_,
    'Gradient Boosting': random_gb.best_score_,
}

if xgb_score > 0:
    results['XGBoost'] = xgb_score
if lgb_score > 0:
    results['LightGBM'] = lgb_score
if cat_score > 0:
    results['CatBoost'] = cat_score

for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:20s}: {score:.4f}")

# Select best ensemble model
best_ensemble_name = max(results, key=results.get)
best_ensemble_score = results[best_ensemble_name]

if best_ensemble_name == 'Random Forest':
    best_ensemble_model = random_rf.best_estimator_
elif best_ensemble_name == 'Extra Trees':
    best_ensemble_model = random_et.best_estimator_
elif best_ensemble_name == 'Gradient Boosting':
    best_ensemble_model = random_gb.best_estimator_
elif best_ensemble_name == 'XGBoost':
    best_ensemble_model = xgb_model_best
elif best_ensemble_name == 'LightGBM':
    best_ensemble_model = lgb_model_best
elif best_ensemble_name == 'CatBoost':
    best_ensemble_model = cat_model_best

print(f"\n{'='*80}")
print(f"BEST ENSEMBLE MODEL: {best_ensemble_name}")
print(f"CV Macro F1 Score: {best_ensemble_score:.4f}")
print(f"{'='*80}")

# Save best model
model_data = {
    'model': best_ensemble_model,
    'model_name': best_ensemble_name,
    'cv_score': best_ensemble_score,
    'all_results': results,
    'feature_type': 'full',
    'scaler_type': 'none'
}

with open('best_ensemble.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✓ Saved best_ensemble.pkl")
