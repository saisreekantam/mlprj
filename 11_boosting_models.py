"""
Try XGBoost and LightGBM with Installation if Needed
"""
import subprocess
import sys
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import loguniform, randint, uniform

np.random.seed(42)

print("="*80)
print("ADVANCED BOOSTING MODELS - XGBoost & LightGBM")
print("="*80)

# Load data
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_full = data['X_train_full']
y_train = data['y_train']

# Encode labels for XGBoost/LightGBM
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def macro_f1_multiclass(y_true, y_pred):
    # Decode predictions back to original labels
    y_true_decoded = le.inverse_transform(y_true)
    y_pred_decoded = le.inverse_transform(y_pred)
    return f1_score(y_true_decoded, y_pred_decoded, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

best_boosting_models = {}

# Try XGBoost
print("\n1. XGBOOST")
print("="*80)
try:
    import xgboost as xgb
    print("✓ XGBoost available")
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': randint(1, 6),
        'gamma': uniform(0, 0.3),
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': loguniform(0.001, 1),
        'reg_lambda': loguniform(0.001, 1)
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
    
    random_xgb = RandomizedSearchCV(
        xgb_model, param_dist, n_iter=50, cv=skf,
        scoring=macro_f1_scorer, n_jobs=-1, verbose=1, random_state=42
    )
    
    random_xgb.fit(X_train_full, y_train_encoded)
    
    print(f"Best params: {random_xgb.best_params_}")
    print(f"10-Fold CV Macro F1: {random_xgb.best_score_:.4f}")
    best_boosting_models['XGBoost'] = {
        'model': random_xgb.best_estimator_,
        'score': random_xgb.best_score_,
        'scaled': False
    }
    
except ImportError:
    print("✗ XGBoost not installed")
    print("Installing XGBoost...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', '-q'])
        print("✓ XGBoost installed successfully. Please re-run this script.")
    except:
        print("✗ Could not install XGBoost automatically")

# Try LightGBM
print("\n2. LIGHTGBM")
print("="*80)
try:
    import lightgbm as lgb
    print("✓ LightGBM available")
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': randint(20, 60),
        'max_depth': [3, 5, 7, 10],
        'min_child_samples': randint(10, 40),
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': loguniform(0.001, 1),
        'reg_lambda': loguniform(0.001, 1)
    }
    
    lgb_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    
    random_lgb = RandomizedSearchCV(
        lgb_model, param_dist, n_iter=50, cv=skf,
        scoring=macro_f1_scorer, n_jobs=-1, verbose=1, random_state=42
    )
    
    random_lgb.fit(X_train_full, y_train_encoded)
    
    print(f"Best params: {random_lgb.best_params_}")
    print(f"10-Fold CV Macro F1: {random_lgb.best_score_:.4f}")
    best_boosting_models['LightGBM'] = {
        'model': random_lgb.best_estimator_,
        'score': random_lgb.best_score_,
        'scaled': False
    }
    
except ImportError:
    print("✗ LightGBM not installed")
    print("Installing LightGBM...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm', '-q'])
        print("✓ LightGBM installed successfully. Please re-run this script.")
    except:
        print("✗ Could not install LightGBM automatically")

# Save if we got any models
if best_boosting_models:
    with open('boosting_models_v2.pkl', 'wb') as f:
        pickle.dump({
            'models': best_boosting_models,
            'label_encoder': le
        }, f)
    print("\n✓ Saved boosting_models_v2.pkl")
    
    print("\nBoosting Models Summary:")
    for name, data in best_boosting_models.items():
        print(f"{name:15s}: {data['score']:.4f}")

print("="*80)
