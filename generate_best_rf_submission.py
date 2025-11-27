"""
Generate submission file using best Random Forest model
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("GENERATING SUBMISSION WITH BEST RANDOM FOREST MODEL")
print("="*80)

# Load benchmarking results
with open('algorithm_benchmarking_results.pkl', 'rb') as f:
    benchmark_data = pickle.load(f)

# Load processed features
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_unscaled = data['X_train_full']
X_test_unscaled = data['X_test_full']
y_train = data['y_train']
test_ids = data['test_ids']

print(f"\nBest algorithm: {benchmark_data['best_algorithm']}")
print(f"CV Score: {benchmark_data['best_score']:.6f}")
print(f"Parameters: {benchmark_data['all_results']['Random Forest']['params']}")

print(f"\nTraining data shape: {X_train_unscaled.shape}")
print(f"Test data shape: {X_test_unscaled.shape}")

# Get the best model
best_model = benchmark_data['best_model']

print("\n" + "="*80)
print("Training model on full training set...")
print("="*80)

# Model is already trained during grid search, just use it
print("✓ Model ready (trained during grid search)")

# Make predictions
print("\nMaking predictions on test set...")
predictions = best_model.predict(X_test_unscaled)
print("✓ Predictions generated!")

# Get prediction probabilities
try:
    probabilities = best_model.predict_proba(X_test_unscaled)
    print("\nPrediction confidence (average probability of predicted class):")
    pred_confidence = []
    for i, pred in enumerate(predictions):
        pred_idx = list(best_model.classes_).index(pred)
        pred_confidence.append(probabilities[i][pred_idx])
    print(f"  Mean confidence: {np.mean(pred_confidence):.4f}")
    print(f"  Std confidence: {np.std(pred_confidence):.4f}")
except:
    print("\nProbabilities not available for this model")

print("\nPrediction distribution:")
unique, counts = np.unique(predictions, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {label}: {count} samples ({count/len(predictions)*100:.1f}%)")

# Create submission file
submission = pd.DataFrame({
    'sample_id': test_ids,
    'category': predictions
})

# Save submission
output_file = 'submission_best_random_forest.csv'
submission.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print(f"✓ Submission file created: {output_file}")
print(f"{'='*80}")
print(f"\nSubmission details:")
print(f"  Total predictions: {len(submission)}")
print(f"  Columns: {list(submission.columns)}")
print(f"\nFirst few predictions:")
print(submission.head(10))

print("\nDone!")
