# Submissions Directory Organization

This directory contains all submission files organized by model type.

## Directory Structure

### 📁 `data/`
- `train.csv` - Training dataset
- `test.csv` - Test dataset  
- `sample_submission.csv.csv` - Sample submission format

### 📁 `submissions/svm/`
SVM model submissions with various kernels and parameters:
- `submission_grid_svmlinear.csv` - Linear kernel
- `submission_grid_svmrbf.csv` - RBF kernel
- `submission_svm.csv` - Basic SVM
- `submission_optimized_svm_v2.csv` - Optimized SVM v2
- `submission_test_*.csv` - Various test configurations

### 📁 `submissions/ensemble/`
Ensemble model submissions:
- `submission_grid_ensemble.csv` - Grid search ensemble
- `submission_optimized_ensemble.csv` - Optimized ensemble
- `submission_bagged_et_v2.csv` - Bagged Extra Trees v2
- `submission_conservative_gb_v2.csv` - Conservative Gradient Boosting v2
- `submission_extra_trees.csv` - Extra Trees

### 📁 `submissions/neural_networks/`
Neural network submissions:
- `submission_grid_mlp.csv` - Grid search MLP
- `submission_nn.csv` - Neural Network

### 📁 `submissions/logistic_regression/`
Logistic regression submissions:
- `submission_logistic_regression.csv` - Basic logistic regression
- `submission_lr.csv` - LR
- `submission_optimized_lr_v2.csv` - Optimized LR v2

### 📁 `submissions/other/`
Other model submissions:
- `submission.csv` - Generic submission
- `submission_knn_v2.csv` - KNN v2
- `svm_mlp_summary.csv` - SVM/MLP comparison

## Latest Submission

**File:** `submission_rbf_optimal_c10_gamma0.5.csv` (in root directory)
- **Model:** SVM with RBF kernel
- **Parameters:** C=10, gamma=0.5
- **CV Score:** 0.986312 (Macro F1)
- **Date:** Generated from comprehensive grid search
