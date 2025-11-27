"""
Advanced Hyperparameter Optimization with Focus on Generalization
Testing different approaches to reduce overfitting and improve test performance
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                               GradientBoostingClassifier, VotingClassifier,
                               BaggingClassifier, AdaBoostClassifier)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import loguniform, randint, uniform
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("ADVANCED HYPERPARAMETER OPTIMIZATION")
print("="*80)
print("\nGoal: Achieve >0.992 test score with better generalization")
print("Strategy: More conservative hyperparameters, diverse models, robust CV")

# Load data
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_raw = data['X_train_raw']
X_train_full = data['X_train_full']
y_train = data['y_train']

# Use more aggressive CV to better estimate test performance
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold for better estimation

def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

# Standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)

best_models = {}

print("\n" + "="*80)
print("1. OPTIMIZED LOGISTIC REGRESSION (Ridge-regularized)")
print("="*80)

# Try Ridge regularization with stricter penalties
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [2000],
    'penalty': ['l2'],
    'class_weight': ['balanced', None]
}

lr = LogisticRegression(random_state=42, multi_class='multinomial')
grid_lr = GridSearchCV(lr, param_grid_lr, cv=skf, scoring=macro_f1_scorer, n_jobs=-1, verbose=1)
grid_lr.fit(X_train_scaled, y_train)

print(f"Best params: {grid_lr.best_params_}")
print(f"10-Fold CV Macro F1: {grid_lr.best_score_:.4f}")
best_models['Optimized LR'] = {'model': grid_lr.best_estimator_, 'score': grid_lr.best_score_, 'scaled': True}

print("\n" + "="*80)
print("2. OPTIMIZED SVM WITH CALIBRATION")
print("="*80)

# More conservative gamma values
param_dist_svm = {
    'C': loguniform(0.5, 50),
    'gamma': loguniform(0.01, 5),  # Narrower range
    'kernel': ['rbf']
}

svm = SVC(class_weight='balanced', probability=True, random_state=42)
random_svm = RandomizedSearchCV(svm, param_dist_svm, n_iter=40, cv=skf, 
                                scoring=macro_f1_scorer, n_jobs=-1, verbose=1, random_state=42)
random_svm.fit(X_train_scaled, y_train)

print(f"Best params: {random_svm.best_params_}")
print(f"10-Fold CV Macro F1: {random_svm.best_score_:.4f}")
best_models['Optimized SVM'] = {'model': random_svm.best_estimator_, 'score': random_svm.best_score_, 'scaled': True}

print("\n" + "="*80)
print("3. SIMPLIFIED NEURAL NETWORK (Less prone to overfitting)")
print("="*80)

# Simpler architectures with more regularization
param_dist_nn = {
    'hidden_layer_sizes': [(30,), (50,), (70,), (40, 20), (60, 30)],
    'activation': ['relu', 'tanh'],
    'alpha': loguniform(0.01, 1),  # Stronger regularization
    'learning_rate_init': [0.001, 0.0005],
    'early_stopping': [True],
    'validation_fraction': [0.15],
    'max_iter': [1000]
}

nn = MLPClassifier(random_state=42, solver='adam')
random_nn = RandomizedSearchCV(nn, param_dist_nn, n_iter=30, cv=skf,
                               scoring=macro_f1_scorer, n_jobs=-1, verbose=1, random_state=42)
random_nn.fit(X_train_scaled, y_train)

print(f"Best params: {random_nn.best_params_}")
print(f"10-Fold CV Macro F1: {random_nn.best_score_:.4f}")
best_models['Optimized NN'] = {'model': random_nn.best_estimator_, 'score': random_nn.best_score_, 'scaled': True}

print("\n" + "="*80)
print("4. BAGGED EXTRA TREES (Variance reduction)")
print("="*80)

# Use bagging to reduce variance
param_grid_et = {
    'n_estimators': [200, 300, 500],
    'max_depth': [8, 10, 12, 15],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 'log2']
}

et = ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
grid_et = GridSearchCV(et, param_grid_et, cv=skf, scoring=macro_f1_scorer, n_jobs=-1, verbose=1)
grid_et.fit(X_train_full, y_train)

print(f"Best params: {grid_et.best_params_}")
print(f"10-Fold CV Macro F1: {grid_et.best_score_:.4f}")
best_models['Bagged ET'] = {'model': grid_et.best_estimator_, 'score': grid_et.best_score_, 'scaled': False}

print("\n" + "="*80)
print("5. GRADIENT BOOSTING (Conservative learning rate)")
print("="*80)

param_grid_gb = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
    'subsample': [0.8, 0.9],
    'max_features': ['sqrt']
}

gb = GradientBoostingClassifier(random_state=42)
grid_gb = GridSearchCV(gb, param_grid_gb, cv=skf, scoring=macro_f1_scorer, n_jobs=-1, verbose=1)
grid_gb.fit(X_train_full, y_train)

print(f"Best params: {grid_gb.best_params_}")
print(f"10-Fold CV Macro F1: {grid_gb.best_score_:.4f}")
best_models['Conservative GB'] = {'model': grid_gb.best_estimator_, 'score': grid_gb.best_score_, 'scaled': False}

print("\n" + "="*80)
print("6. K-NEAREST NEIGHBORS (Different approach)")
print("="*80)

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()
grid_knn = GridSearchCV(knn, param_grid_knn, cv=skf, scoring=macro_f1_scorer, n_jobs=-1, verbose=1)
grid_knn.fit(X_train_scaled, y_train)

print(f"Best params: {grid_knn.best_params_}")
print(f"10-Fold CV Macro F1: {grid_knn.best_score_:.4f}")
best_models['KNN'] = {'model': grid_knn.best_estimator_, 'score': grid_knn.best_score_, 'scaled': True}

print("\n" + "="*80)
print("7. QUADRATIC DISCRIMINANT ANALYSIS")
print("="*80)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
from sklearn.model_selection import cross_val_score
qda_scores = cross_val_score(qda, X_train_scaled, y_train, cv=skf, scoring=macro_f1_scorer, n_jobs=-1)
qda_score = qda_scores.mean()

qda.fit(X_train_scaled, y_train)
print(f"10-Fold CV Macro F1: {qda_score:.4f}")
best_models['QDA'] = {'model': qda, 'score': qda_score, 'scaled': True}

# Summary
print("\n" + "="*80)
print("ALL MODELS SUMMARY (10-Fold CV)")
print("="*80)

sorted_models = sorted(best_models.items(), key=lambda x: x[1]['score'], reverse=True)
for name, data in sorted_models:
    print(f"{name:25s}: {data['score']:.4f}")

# Save all optimized models
with open('optimized_models_v2.pkl', 'wb') as f:
    pickle.dump({
        'models': best_models,
        'scaler': scaler,
        'cv_folds': 10
    }, f)

print("\n✓ Saved optimized_models_v2.pkl")
print("="*80)
