"""
Test Friend's SVM Parameters vs Our Best
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("COMPARING SVM CONFIGURATIONS")
print("="*80)

# Load data
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_full = data['X_train_full']
y_train = data['y_train']
X_test_full = data['X_test_full']
test_ids = data['test_ids']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

# 10-fold CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def macro_f1_multiclass(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', labels=['Group_A', 'Group_B', 'Group_C'])

macro_f1_scorer = make_scorer(macro_f1_multiclass)

# Test configurations
configs = [
    {
        'name': "Friend's Parameters",
        'params': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf', 'class_weight': 'balanced'},
        'color': '🔵'
    },
    {
        'name': 'Our GridSearch Best',
        'params': {'C': 15, 'gamma': 0.7, 'kernel': 'rbf', 'class_weight': 'balanced'},
        'color': '🟢'
    },
    {
        'name': "Friend's (no balance)",
        'params': {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'},
        'color': '🟡'
    },
    {
        'name': 'High C + gamma=auto',
        'params': {'C': 100, 'gamma': 'auto', 'kernel': 'rbf', 'class_weight': 'balanced'},
        'color': '🟣'
    },
    {
        'name': 'Alternative: C=50, gamma=scale',
        'params': {'C': 50, 'gamma': 'scale', 'kernel': 'rbf', 'class_weight': 'balanced'},
        'color': '🟠'
    }
]

results = []

for config in configs:
    print(f"\n{config['color']} Testing: {config['name']}")
    print(f"   Parameters: {config['params']}")
    
    svm = SVC(probability=True, random_state=42, **config['params'])
    
    # 10-fold CV
    cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=skf, 
                                 scoring=macro_f1_scorer, n_jobs=-1)
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    print(f"   10-Fold CV: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"   Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    # Train on full data and generate predictions
    svm.fit(X_train_scaled, y_train)
    predictions = svm.predict(X_test_scaled)
    
    # Save submission
    submission = pd.DataFrame({
        'sample_id': test_ids,
        'category': predictions
    })
    submission = submission.sort_values('sample_id').reset_index(drop=True)
    
    safe_name = config['name'].replace(' ', '_').replace("'", '').lower().replace('(', '').replace(')', '')
    filename = f'submission_test_{safe_name}.csv'
    submission.to_csv(filename, index=False)
    
    results.append({
        'Config': config['name'],
        'CV_F1': mean_score,
        'Std': std_score,
        'Params': str(config['params']),
        'File': filename
    })
    
    print(f"   ✓ Saved: {filename}")

print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

results_df = pd.DataFrame(results).sort_values('CV_F1', ascending=False)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("VERDICT")
print("="*80)

best = results_df.iloc[0]
print(f"🏆 WINNER: {best['Config']}")
print(f"   CV F1 Score: {best['CV_F1']:.4f} (+/- {best['Std']:.4f})")
print(f"   Submission: {best['File']}")

if "Friend" in best['Config']:
    print("\n   ✅ Your friend's parameters ARE better!")
else:
    print("\n   ✅ Our GridSearch parameters are confirmed best!")

# Calculate gamma='scale' value
from sklearn.svm import SVC
temp_svm = SVC(gamma='scale')
temp_svm.fit(X_train_scaled, y_train)
actual_gamma_scale = temp_svm._gamma

print(f"\n💡 Note: gamma='scale' = {actual_gamma_scale:.4f} for this dataset")
print(f"   (Formula: 1 / (n_features * X.var()) = 1 / ({X_train_scaled.shape[1]} * {X_train_scaled.var():.4f}))")

print("="*80)
