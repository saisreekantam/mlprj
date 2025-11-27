"""
Create Optimized Ensemble and Generate New Submissions
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("OPTIMIZED ENSEMBLE CREATION")
print("="*80)

# Load data
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_full = data['X_train_full']
X_test_full = data['X_test_full']
y_train = data['y_train']
test_ids = data['test_ids']

# Load optimized models
with open('optimized_models_v2.pkl', 'rb') as f:
    opt_data = pickle.load(f)

models = opt_data['models']
scaler = opt_data['scaler']

X_train_scaled = scaler.transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

print("\nLoaded Optimized Models:")
for name, data in sorted(models.items(), key=lambda x: x[1]['score'], reverse=True):
    print(f"  {name:25s}: {data['score']:.4f}")

# Create weighted ensemble with top models
print("\n" + "="*80)
print("WEIGHTED ENSEMBLE STRATEGY")
print("="*80)

# Select top 5 models
top_models = sorted(models.items(), key=lambda x: x[1]['score'], reverse=True)[:5]

print(f"\nUsing top {len(top_models)} models:")
for name, data in top_models:
    print(f"  {name:25s}: {data['score']:.4f}")

# 10-fold CV to evaluate ensemble
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

ensemble_cv_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    X_val_scaled = X_train_scaled[val_idx]
    X_val_full = X_train_full[val_idx]
    y_val = y_train[val_idx]
    
    # Collect predictions from all models
    all_probas = []
    weights = []
    
    for name, model_data in top_models:
        model = model_data['model']
        score = model_data['score']
        scaled = model_data['scaled']
        
        if scaled:
            probas = model.predict_proba(X_val_scaled)
        else:
            probas = model.predict_proba(X_val_full)
        
        all_probas.append(probas)
        weights.append(score)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Weighted average
    ensemble_probas = np.zeros_like(all_probas[0])
    for probas, weight in zip(all_probas, weights):
        ensemble_probas += probas * weight
    
    # Get predictions
    pred_indices = ensemble_probas.argmax(axis=1)
    class_labels = top_models[0][1]['model'].classes_
    predictions = class_labels[pred_indices]
    
    # Calculate F1
    fold_score = f1_score(y_val, predictions, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])
    ensemble_cv_scores.append(fold_score)
    print(f"Fold {fold_idx + 1}: {fold_score:.4f}")

ensemble_score = np.mean(ensemble_cv_scores)
ensemble_std = np.std(ensemble_cv_scores)

print(f"\nWeighted Ensemble 10-Fold CV: {ensemble_score:.4f} (+/- {ensemble_std:.4f})")

print("\n" + "="*80)
print("GENERATING PREDICTIONS ON TEST SET")
print("="*80)

# Generate test predictions
all_test_probas = []
weights = []

for name, model_data in top_models:
    model = model_data['model']
    score = model_data['score']
    scaled = model_data['scaled']
    
    if scaled:
        test_probas = model.predict_proba(X_test_scaled)
    else:
        test_probas = model.predict_proba(X_test_full)
    
    all_test_probas.append(test_probas)
    weights.append(score)
    print(f"  ✓ {name:25s}: weight = {score:.4f}")

# Normalize weights
weights = np.array(weights) / sum(weights)

# Weighted average
ensemble_test_probas = np.zeros_like(all_test_probas[0])
for probas, weight in zip(all_test_probas, weights):
    ensemble_test_probas += probas * weight

# Get final predictions
pred_indices = ensemble_test_probas.argmax(axis=1)
class_labels = top_models[0][1]['model'].classes_
final_predictions = class_labels[pred_indices]

# Check distribution
unique, counts = np.unique(final_predictions, return_counts=True)
print("\nPredicted class distribution:")
for cls, count in zip(unique, counts):
    print(f"  {cls}: {count} ({count/len(final_predictions)*100:.1f}%)")

# Create submission
submission = pd.DataFrame({
    'sample_id': test_ids,
    'category': final_predictions
})

submission = submission.sort_values('sample_id').reset_index(drop=True)
submission.to_csv('submission_optimized_ensemble.csv', index=False)

print(f"\n✓ Saved: submission_optimized_ensemble.csv")
print(f"   Expected CV Macro F1: {ensemble_score:.4f}")

# Also generate individual submissions for top models
print("\n" + "="*80)
print("GENERATING INDIVIDUAL SUBMISSIONS")
print("="*80)

for name, model_data in top_models:
    model = model_data['model']
    scaled = model_data['scaled']
    score = model_data['score']
    
    if scaled:
        predictions = model.predict(X_test_scaled)
    else:
        predictions = model.predict(X_test_full)
    
    submission = pd.DataFrame({
        'sample_id': test_ids,
        'category': predictions
    })
    
    submission = submission.sort_values('sample_id').reset_index(drop=True)
    
    safe_name = name.replace(' ', '_').lower()
    filename = f'submission_{safe_name}_v2.csv'
    submission.to_csv(filename, index=False)
    
    print(f"  ✓ {filename:45s} (CV: {score:.4f})")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"🏆 BEST: submission_optimized_ensemble.csv (Expected: {ensemble_score:.4f})")
print(f"🥈 Backup: submission_optimized_svm_v2.csv (Expected: {models['Optimized SVM']['score']:.4f})")
print(f"🥉 Alternative: submission_conservative_gb_v2.csv (Expected: {models['Conservative GB']['score']:.4f})")
print("="*80)
