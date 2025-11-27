"""
MLP Classifier - Complete Working Version
This script replicates the MLP approach from svm.py with all preprocessing steps included
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("MLP CLASSIFIER - COMPLETE IMPLEMENTATION")
print("="*80)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

start_time = time.time()

# Load data
print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Separate features and target
X_train = train_df.drop(columns=['sample_id', 'category'])
y_train = train_df['category']
X_test = test_df.drop(columns=['sample_id'])
sample_ids = test_df['sample_id']

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Class distribution:\n{y_train.value_counts()}\n")

# Scale features
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize LabelEncoder and fit it to y_train
print("Encoding labels...")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

print(f"Original classes: {label_encoder.classes_}")
print(f"Encoded classes: {np.unique(y_train_encoded)}\n")

# Define a hyperparameter grid for MLPClassifier
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'max_iter': [200, 300]
}

print("="*80)
print("GRID SEARCH FOR MLP CLASSIFIER")
print("="*80)
print(f"Parameter grid:")
for key, value in param_grid_mlp.items():
    print(f"  {key}: {value}")
print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid_mlp.values()])}")
print()

# Instantiate an MLPClassifier model with a random_state for reproducibility
mlp_model = MLPClassifier(random_state=42)

# Initialize GridSearchCV
grid_search_mlp = GridSearchCV(
    estimator=mlp_model,
    param_grid=param_grid_mlp,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

# Fit grid_search_mlp to the scaled training data and encoded training labels
print("Starting Grid Search (this may take a while)...")
grid_start = time.time()
grid_search_mlp.fit(X_train_scaled, y_train_encoded)
grid_time = time.time() - grid_start

# Print the best parameters and corresponding best cross-validation score
print(f"\n{'='*80}")
print("GRID SEARCH RESULTS")
print("="*80)
print(f"Best parameters for MLPClassifier: {grid_search_mlp.best_params_}")
print(f"Best cross-validation score (F1 Weighted): {grid_search_mlp.best_score_:.4f}")
print(f"Time taken for grid search: {grid_time:.2f} seconds ({grid_time/60:.2f} minutes)")

# Get the best estimator
best_mlp_model = grid_search_mlp.best_estimator_

# Evaluate the model on the training data
print(f"\n{'='*80}")
print("TRAINING SET EVALUATION")
print("="*80)

# Make predictions on the scaled training data
y_train_pred_encoded = best_mlp_model.predict(X_train_scaled)
y_train_pred_labels = label_encoder.inverse_transform(y_train_pred_encoded)

# Generate and print the classification report
print("\nClassification Report (Training Data):")
print(classification_report(y_train, y_train_pred_labels))

# Generate and print the confusion matrix
print("\nConfusion Matrix (Training Data):")
print(confusion_matrix(y_train_encoded, y_train_pred_encoded))

# Make predictions on the scaled test data
print(f"\n{'='*80}")
print("GENERATING PREDICTIONS FOR TEST SET")
print("="*80)

mlp_predictions_encoded = best_mlp_model.predict(X_test_scaled)

# Inverse transform predictions to get original class labels
mlp_predictions_labels = label_encoder.inverse_transform(mlp_predictions_encoded)

# Create a submission DataFrame and save it to a CSV file
mlp_submission_df = pd.DataFrame({
    'sample_id': sample_ids, 
    'category': mlp_predictions_labels
})

submission_filename = 'mlp_submission.csv'
mlp_submission_df.to_csv(submission_filename, index=False)

print(f"\n✓ Submission file saved: {submission_filename}")

# Display prediction distribution
print("\nPrediction distribution:")
print(mlp_submission_df['category'].value_counts())

# Save model and scaler
model_data = {
    'model': best_mlp_model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'best_params': grid_search_mlp.best_params_,
    'cv_score': grid_search_mlp.best_score_,
    'grid_search_results': grid_search_mlp.cv_results_
}

with open('mlp_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Saved mlp_model.pkl")

# Final timing summary
total_time = time.time() - start_time
print("\n" + "="*80)
print("TIMING SUMMARY")
print("="*80)
print(f"Grid Search: {grid_time:.2f} seconds ({grid_time/60:.2f} minutes)")
print(f"Total time:  {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("\nMLPClassifier submission file created successfully!")
print("="*80)
