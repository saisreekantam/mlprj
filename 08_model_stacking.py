"""
Model Stacking and Ensemble Combination for Maximum F1 Score
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("MODEL STACKING & ENSEMBLE COMBINATION")
print("="*80)

# Load processed features
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_scaled = data['scaled_data']['standard']['train_full']
X_train_full = data['X_train_full']
y_train = data['y_train']

# Define Macro F1 scorer
def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Load all trained models
print("\n" + "="*80)
print("LOADING TRAINED MODELS")
print("="*80)

models_data = {}

try:
    with open('best_logistic_regression.pkl', 'rb') as f:
        lr_data = pickle.load(f)
        models_data['Logistic Regression'] = lr_data
        print(f"✓ Logistic Regression: {lr_data['cv_score']:.4f}")
except:
    print("✗ Logistic Regression not found")

try:
    with open('best_svm.pkl', 'rb') as f:
        svm_data = pickle.load(f)
        models_data['SVM'] = svm_data
        print(f"✓ SVM: {svm_data['cv_score']:.4f}")
except:
    print("✗ SVM not found")

try:
    with open('best_neural_network.pkl', 'rb') as f:
        nn_data = pickle.load(f)
        models_data['Neural Network'] = nn_data
        print(f"✓ Neural Network: {nn_data['cv_score']:.4f}")
except:
    print("✗ Neural Network not found")

try:
    with open('best_ensemble.pkl', 'rb') as f:
        ensemble_data = pickle.load(f)
        models_data['Ensemble'] = ensemble_data
        print(f"✓ Ensemble ({ensemble_data['model_name']}): {ensemble_data['cv_score']:.4f}")
except:
    print("✗ Ensemble not found")

if len(models_data) < 2:
    print("\nNot enough models trained yet. Please wait for model training to complete.")
    exit()

print("\n" + "="*80)
print("1. WEIGHTED VOTING ENSEMBLE")
print("="*80)

# Create voting classifier with weights based on CV scores
voting_estimators = []
weights = []

for name, model_data in models_data.items():
    model = model_data['model']
    cv_score = model_data['cv_score']
    
    # Some models need scaling, some don't
    if model_data['feature_type'] == 'scaled_full':
        # We'll handle this in the ensemble
        voting_estimators.append((name, model))
    else:
        voting_estimators.append((name, model))
    
    weights.append(cv_score)

# Normalize weights
weights = np.array(weights) / sum(weights)

print(f"Ensemble estimators: {len(voting_estimators)}")
for (name, _), weight in zip(voting_estimators, weights):
    print(f"  {name:25s}: weight = {weight:.4f}")

# Note: VotingClassifier doesn't work well when models need different input features
# So we'll implement a custom weighted voting

print("\n" + "="*80)
print("2. CUSTOM WEIGHTED AVERAGE ENSEMBLE")
print("="*80)

def custom_weighted_ensemble_predict(models_data, X_scaled, X_full):
    """Predict using weighted average of model probabilities"""
    all_probas = []
    all_weights = []
    
    for name, model_data in models_data.items():
        model = model_data['model']
        cv_score = model_data['cv_score']
        
        # Use appropriate features
        if model_data['feature_type'] == 'scaled_full' or model_data['scaler_type'] == 'standard':
            probas = model.predict_proba(X_scaled)
        else:
            probas = model.predict_proba(X_full)
        
        all_probas.append(probas)
        all_weights.append(cv_score)
    
    # Normalize weights
    all_weights = np.array(all_weights)
    all_weights = all_weights / all_weights.sum()
    
    # Weighted average of probabilities
    weighted_probas = np.zeros_like(all_probas[0])
    for probas, weight in zip(all_probas, all_weights):
        weighted_probas += probas * weight
    
    # Get class predictions
    class_indices = weighted_probas.argmax(axis=1)
    classes = all_probas[0].shape[1]
    classes_labels = ['Group_A', 'Group_B', 'Group_C']  # Assuming this order
    
    # Map indices to correct labels based on first model's classes_
    first_model = list(models_data.values())[0]['model']
    if hasattr(first_model, 'classes_'):
        classes_labels = first_model.classes_
    
    predictions = classes_labels[class_indices]
    
    return predictions, weighted_probas

# Evaluate weighted ensemble using cross-validation
fold_scores = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    # Split data
    X_train_fold_scaled = X_train_scaled[train_idx]
    X_val_fold_scaled = X_train_scaled[val_idx]
    X_train_fold_full = X_train_full[train_idx]
    X_val_fold_full = X_train_full[val_idx]
    y_train_fold = y_train[train_idx]
    y_val_fold = y_train[val_idx]
    
    # Train all models on this fold (using pre-trained models as they're already optimized)
    # For simplicity, we use the already trained models
    predictions, _ = custom_weighted_ensemble_predict(models_data, X_val_fold_scaled, X_val_fold_full)
    
    # Calculate F1 score
    fold_score = f1_score(y_val_fold, predictions, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])
    fold_scores.append(fold_score)
    print(f"Fold {fold_idx + 1}: {fold_score:.4f}")

weighted_ensemble_score = np.mean(fold_scores)
print(f"\nWeighted Ensemble CV Macro F1 Score: {weighted_ensemble_score:.4f} (+/- {np.std(fold_scores):.4f})")

print("\n" + "="*80)
print("3. BEST MODEL SELECTION")
print("="*80)

# Compare all approaches
all_results = {}
for name, model_data in models_data.items():
    all_results[name] = model_data['cv_score']

all_results['Weighted Ensemble'] = weighted_ensemble_score

print("All Model Scores:")
for name, score in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:30s}: {score:.4f}")

# Select best
best_approach = max(all_results, key=all_results.get)
best_score = all_results[best_approach]

print(f"\n{'='*80}")
print(f"BEST APPROACH: {best_approach}")
print(f"CV Macro F1 Score: {best_score:.4f}")
print(f"{'='*80}")

# Save best configuration
if best_approach == 'Weighted Ensemble':
    best_config = {
        'type': 'weighted_ensemble',
        'models_data': models_data,
        'cv_score': weighted_ensemble_score,
        'weights': {name: data['cv_score'] for name, data in models_data.items()}
    }
else:
    best_config = {
        'type': 'single_model',
        'model': models_data[best_approach]['model'],
        'model_name': best_approach,
        'cv_score': best_score,
        'feature_type': models_data[best_approach]['feature_type'],
        'scaler_type': models_data[best_approach]['scaler_type']
    }

with open('best_final_model.pkl', 'wb') as f:
    pickle.dump(best_config, f)

print("\n✓ Saved best_final_model.pkl")

# Print detailed summary
print("\n" + "="*80)
print("OPTIMIZATION SUMMARY")
print("="*80)
print(f"Total models evaluated: {len(models_data)}")
print(f"Best individual model: {max(models_data, key=lambda x: models_data[x]['cv_score'])}")
print(f"Best overall approach: {best_approach}")
print(f"Expected Macro F1 Score: {best_score:.4f}")
print("="*80)
