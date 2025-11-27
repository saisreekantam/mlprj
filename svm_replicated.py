"""
SVM Models Replication - Complete Working Version
This script replicates the SVM approach with all preprocessing steps included
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report
from scipy.stats import loguniform, uniform
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("SVM MODELS - COMPLETE REPLICATION")
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

# Define Macro F1 scorer
def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store all results
all_results = {}
all_models = {}

print("\n" + "="*80)
print("1. LINEAR SVM")
print("="*80)

linear_start = time.time()

param_grid_linear = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear']
}

svm_linear = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

grid_linear = GridSearchCV(
    svm_linear, param_grid_linear,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_linear.fit(X_train_scaled, y_train)

linear_time = time.time() - linear_start
print(f"\nBest parameters: {grid_linear.best_params_}")
print(f"Best CV Macro F1 Score: {grid_linear.best_score_:.4f}")
print(f"Time taken: {linear_time:.2f} seconds")

all_results['Linear SVM'] = grid_linear.best_score_
all_models['Linear SVM'] = grid_linear.best_estimator_

print("\n" + "="*80)
print("2. RBF SVM (Radial Basis Function)")
print("="*80)

rbf_start = time.time()

param_dist_rbf = {
    'C': loguniform(0.1, 100),
    'gamma': loguniform(0.001, 10),
    'kernel': ['rbf']
}

svm_rbf = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

random_rbf = RandomizedSearchCV(
    svm_rbf, param_dist_rbf,
    n_iter=50,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_rbf.fit(X_train_scaled, y_train)

rbf_time = time.time() - rbf_start
print(f"\nBest parameters: {random_rbf.best_params_}")
print(f"Best CV Macro F1 Score: {random_rbf.best_score_:.4f}")
print(f"Time taken: {rbf_time:.2f} seconds")

all_results['RBF SVM'] = random_rbf.best_score_
all_models['RBF SVM'] = random_rbf.best_estimator_

print("\n" + "="*80)
print("3. POLYNOMIAL SVM")
print("="*80)

poly_start = time.time()

param_grid_poly = {
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4],
    'kernel': ['poly'],
    'coef0': [0, 1]
}

svm_poly = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

grid_poly = GridSearchCV(
    svm_poly, param_grid_poly,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_poly.fit(X_train_scaled, y_train)

poly_time = time.time() - poly_start
print(f"\nBest parameters: {grid_poly.best_params_}")
print(f"Best CV Macro F1 Score: {grid_poly.best_score_:.4f}")
print(f"Time taken: {poly_time:.2f} seconds")

all_results['Polynomial SVM'] = grid_poly.best_score_
all_models['Polynomial SVM'] = grid_poly.best_estimator_

print("\n" + "="*80)
print("4. SIGMOID SVM")
print("="*80)

sigmoid_start = time.time()

param_dist_sigmoid = {
    'C': loguniform(0.1, 100),
    'gamma': loguniform(0.001, 1),
    'kernel': ['sigmoid'],
    'coef0': uniform(0, 1)
}

svm_sigmoid = SVC(
    class_weight='balanced',
    probability=True,
    random_state=42
)

random_sigmoid = RandomizedSearchCV(
    svm_sigmoid, param_dist_sigmoid,
    n_iter=30,
    cv=skf,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_sigmoid.fit(X_train_scaled, y_train)

sigmoid_time = time.time() - sigmoid_start
print(f"\nBest parameters: {random_sigmoid.best_params_}")
print(f"Best CV Macro F1 Score: {random_sigmoid.best_score_:.4f}")
print(f"Time taken: {sigmoid_time:.2f} seconds")

all_results['Sigmoid SVM'] = random_sigmoid.best_score_
all_models['Sigmoid SVM'] = random_sigmoid.best_estimator_

# Compare all SVM models
print("\n" + "="*80)
print("SVM MODELS SUMMARY")
print("="*80)

for name, score in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:20s}: {score:.4f}")

# Select best SVM model
best_svm_name = max(all_results, key=all_results.get)
best_svm_score = all_results[best_svm_name]
best_svm_model = all_models[best_svm_name]

print(f"\n{'='*80}")
print(f"BEST SVM MODEL: {best_svm_name}")
print(f"CV Macro F1 Score: {best_svm_score:.4f}")
print(f"{'='*80}")

# Make predictions on training data for evaluation
print("\n" + "="*80)
print("TRAINING SET EVALUATION")
print("="*80)

y_train_pred = best_svm_model.predict(X_train_scaled)
print(classification_report(y_train, y_train_pred, 
                          labels=['Group_A', 'Group_B', 'Group_C']))

# Make predictions on test data and create submission file
print("\n" + "="*80)
print("GENERATING PREDICTIONS FOR TEST SET")
print("="*80)

test_predictions = best_svm_model.predict(X_test_scaled)
submission_df = pd.DataFrame({
    'sample_id': sample_ids,
    'category': test_predictions
})

submission_filename = 'svm_replicated_submission.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"\n✓ Submission file saved: {submission_filename}")

# Display prediction distribution
print("\nPrediction distribution:")
print(submission_df['category'].value_counts())

# Save model and scaler
model_data = {
    'model': best_svm_model,
    'scaler': scaler,
    'model_name': best_svm_name,
    'cv_score': best_svm_score,
    'all_results': all_results,
    'feature_type': 'scaled_full',
    'scaler_type': 'standard'
}

with open('svm_replicated_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Saved svm_replicated_model.pkl")

# Final timing summary
total_time = time.time() - start_time
print("\n" + "="*80)
print("TIMING SUMMARY")
print("="*80)
print(f"Linear SVM:     {linear_time:.2f} seconds")
print(f"RBF SVM:        {rbf_time:.2f} seconds")
print(f"Polynomial SVM: {poly_time:.2f} seconds")
print(f"Sigmoid SVM:    {sigmoid_time:.2f} seconds")
print(f"Total time:     {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
