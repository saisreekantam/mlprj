"""
Neural Network (MLP) Models with Hyperparameter Tuning
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from scipy.stats import loguniform
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("NEURAL NETWORK (MLP) MODELS")
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
print("NEURAL NETWORK HYPERPARAMETER SEARCH")
print("="*80)

# Comprehensive parameter distribution for MLP
param_dist = {
    'hidden_layer_sizes': [
        (50,), (100,), (150,),
        (50, 25), (100, 50), (150, 75),
        (100, 50, 25), (150, 100, 50),
        (200, 100, 50), (200, 100, 50, 25)
    ],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'lbfgs'],
    'alpha': loguniform(0.0001, 0.1),
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [500, 1000, 2000],
    'early_stopping': [True],
    'validation_fraction': [0.1]
}

mlp = MLPClassifier(random_state=42)

# Randomized search (more efficient for large parameter space)
random_search = RandomizedSearchCV(
    mlp, param_dist,
    n_iter=80,  # Test 80 combinations
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("Starting hyperparameter search (this may take several minutes)...")
random_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best CV Macro F1 Score: {random_search.best_score_:.4f}")

# Get top 5 configurations
results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')
top_5 = results_df.head(5)[['params', 'mean_test_score', 'std_test_score']]

print("\n" + "="*80)
print("TOP 5 NEURAL NETWORK CONFIGURATIONS")
print("="*80)
for idx, row in top_5.iterrows():
    print(f"\nRank {row.name + 1}:")
    print(f"  Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
    print(f"  Params: {row['params']}")

# Train final model with best parameters on full training data
best_mlp = random_search.best_estimator_

print(f"\n{'='*80}")
print(f"BEST NEURAL NETWORK MODEL")
print(f"CV Macro F1 Score: {random_search.best_score_:.4f}")
print(f"Architecture: {best_mlp.hidden_layer_sizes}")
print(f"Activation: {best_mlp.activation}")
print(f"Solver: {best_mlp.solver}")
print(f"Alpha: {best_mlp.alpha:.6f}")
print(f"{'='*80}")

# Save best model
model_data = {
    'model': best_mlp,
    'model_name': 'Neural Network (MLP)',
    'cv_score': random_search.best_score_,
    'best_params': random_search.best_params_,
    'top_5_configs': top_5.to_dict('records'),
    'feature_type': 'scaled_full',
    'scaler_type': 'standard'
}

with open('best_neural_network.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✓ Saved best_neural_network.pkl")
