"""
Generate submission file using optimal RBF SVM parameters
Parameters: C=10, gamma=0.5 (from grid search)
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("GENERATING SUBMISSION WITH OPTIMAL RBF PARAMETERS")
print("="*80)

# Load processed features
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_scaled = data['scaled_data']['standard']['train_full']
X_test_scaled = data['scaled_data']['standard']['test_full']
y_train = data['y_train']
test_ids = data['test_ids']

print(f"\nTraining data shape: {X_train_scaled.shape}")
print(f"Test data shape: {X_test_scaled.shape}")

# Create and train model with optimal parameters
print("\n" + "="*80)
print("Training SVM with optimal parameters:")
print("  C = 10")
print("  gamma = 0.5")
print("  kernel = rbf")
print("="*80)

optimal_svm = SVC(
    C=10,
    gamma=0.5,
    kernel='rbf',
    class_weight='balanced',
    probability=True,
    random_state=42
)

print("\nTraining model on full training set...")
optimal_svm.fit(X_train_scaled, y_train)
print("✓ Model trained successfully!")

# Make predictions
print("\nMaking predictions on test set...")
predictions = optimal_svm.predict(X_test_scaled)
print("✓ Predictions generated!")

# Get prediction probabilities
probabilities = optimal_svm.predict_proba(X_test_scaled)
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
output_file = 'submission_rbf_optimal_c10_gamma0.5.csv'
submission.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print(f"✓ Submission file created: {output_file}")
print(f"{'='*80}")
print(f"\nSubmission details:")
print(f"  Total predictions: {len(submission)}")
print(f"  Columns: {list(submission.columns)}")
print(f"\nFirst few predictions:")
print(submission.head(10))

# Save the trained model
model_package = {
    'model': optimal_svm,
    'parameters': {'C': 10, 'gamma': 0.5, 'kernel': 'rbf'},
    'cv_score': 0.986312,  # From grid search
    'feature_type': 'scaled_full',
    'scaler_type': 'standard',
    'random_state': 42
}

with open('optimal_rbf_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"\n✓ Model saved to: optimal_rbf_model.pkl")
print("\nDone!")
