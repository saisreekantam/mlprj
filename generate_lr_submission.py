"""
Quick Submission Generator - Using Best Logistic Regression Model
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("GENERATING LOGISTIC REGRESSION SUBMISSION")
print("="*80)

# Load processed features
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_test_scaled = data['scaled_data']['standard']['test_full']
test_ids = data['test_ids']

# Load best logistic regression model
with open('best_logistic_regression.pkl', 'rb') as f:
    lr_data = pickle.load(f)

model = lr_data['model']
cv_score = lr_data['cv_score']
model_name = lr_data['model_name']

print(f"Model: {model_name}")
print(f"CV Macro F1 Score: {cv_score:.4f}")

# Generate predictions
print("\nGenerating predictions...")
predictions = model.predict(X_test_scaled)
print("✓ Predictions generated")

# Check predicted class distribution
unique, counts = np.unique(predictions, return_counts=True)
print("\nPredicted class distribution:")
for cls, count in zip(unique, counts):
    print(f"  {cls}: {count} ({count/len(predictions)*100:.1f}%)")

# Create submission
submission = pd.DataFrame({
    'sample_id': test_ids,
    'category': predictions
})

submission = submission.sort_values('sample_id').reset_index(drop=True)
submission.to_csv('submission_logistic_regression.csv', index=False)

print("\n" + "="*80)
print("SUBMISSION CREATED!")
print("="*80)
print(f"File: submission_logistic_regression.csv")
print(f"Samples: {len(submission)}")
print(f"Expected Macro F1 Score: {cv_score:.4f}")
print("="*80)

print("\nPreview:")
print(submission.head(10))
