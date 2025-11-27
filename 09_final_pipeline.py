"""
Final Pipeline - Generate Submission File
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("FINAL PREDICTION PIPELINE")
print("="*80)

# Load processed features
print("\nLoading processed features...")
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_scaled = data['scaled_data']['standard']['train_full']
X_test_scaled = data['scaled_data']['standard']['test_full']
X_train_full = data['X_train_full']
X_test_full = data['X_test_full']
y_train = data['y_train']
test_ids = data['test_ids']

print(f"✓ Loaded {len(X_train_scaled)} training samples")
print(f"✓ Loaded {len(X_test_scaled)} test samples")

# Load best model configuration
print("\nLoading best model configuration...")
with open('best_final_model.pkl', 'rb') as f:
    best_config = pickle.load(f)

print(f"✓ Best approach: {best_config.get('type', best_config.get('model_name', 'Unknown'))}")
print(f"✓ Expected CV Macro F1 Score: {best_config['cv_score']:.4f}")

print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

if best_config['type'] == 'weighted_ensemble':
    print("Using weighted ensemble of all models...")
    
    # Custom weighted ensemble prediction
    all_probas = []
    all_weights = []
    
    for name, model_data in best_config['models_data'].items():
        model = model_data['model']
        cv_score = model_data['cv_score']
        
        # Use appropriate features
        if model_data['feature_type'] == 'scaled_full' or model_data['scaler_type'] == 'standard':
            probas = model.predict_proba(X_test_scaled)
        else:
            probas = model.predict_proba(X_test_full)
        
        all_probas.append(probas)
        all_weights.append(cv_score)
        print(f"  ✓ {name:25s}: weight = {cv_score:.4f}")
    
    # Normalize weights
    all_weights = np.array(all_weights)
    all_weights = all_weights / all_weights.sum()
    
    # Weighted average of probabilities
    weighted_probas = np.zeros_like(all_probas[0])
    for probas, weight in zip(all_probas, all_weights):
        weighted_probas += probas * weight
    
    # Get class predictions
    class_indices = weighted_probas.argmax(axis=1)
    
    # Get class labels from first model
    first_model = list(best_config['models_data'].values())[0]['model']
    if hasattr(first_model, 'classes_'):
        class_labels = first_model.classes_
    else:
        class_labels = np.array(['Group_A', 'Group_B', 'Group_C'])
    
    predictions = class_labels[class_indices]
    
else:
    print("Using single best model...")
    model = best_config['model']
    
    # Use appropriate features
    if best_config['feature_type'] == 'scaled_full' or best_config['scaler_type'] == 'standard':
        predictions = model.predict(X_test_scaled)
    else:
        predictions = model.predict(X_test_full)
    
    print(f"  ✓ Model: {best_config['model_name']}")

print("\n✓ Predictions generated")

# Check predicted class distribution
unique, counts = np.unique(predictions, return_counts=True)
print("\nPredicted class distribution:")
for cls, count in zip(unique, counts):
    print(f"  {cls}: {count} ({count/len(predictions)*100:.1f}%)")

print("\n" + "="*80)
print("CREATING SUBMISSION FILE")
print("="*80)

# Create submission DataFrame
submission = pd.DataFrame({
    'sample_id': test_ids,
    'category': predictions
})

# Sort by sample_id (as in sample submission)
submission = submission.sort_values('sample_id').reset_index(drop=True)

# Save submission
submission.to_csv('submission.csv', index=False)
print("✓ Saved submission.csv")

# Verify format
sample_submission = pd.read_csv('sample_submission.csv.csv')
print("\nVerifying submission format...")
print(f"  ✓ Columns match: {list(submission.columns) == list(sample_submission.columns)}")
print(f"  ✓ Number of rows: {len(submission)} (expected: {len(test_ids)})")
print(f"  ✓ Sample IDs present: {len(submission['sample_id'].unique())} unique IDs")
print(f"  ✓ Valid categories: {set(submission['category'].unique()).issubset({'Group_A', 'Group_B', 'Group_C'})}")

# Show sample of submission
print("\n" + "="*80)
print("SUBMISSION PREVIEW")
print("="*80)
print(submission.head(10))

print("\n" + "="*80)
print("SUBMISSION COMPLETE!")
print("="*80)
print(f"File: submission.csv")
print(f"Samples: {len(submission)}")
print(f"Expected Macro F1 Score (CV): {best_config['cv_score']:.4f}")
print("="*80)
