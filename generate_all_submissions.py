"""
Generate Submission Files for All Models
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("GENERATING SUBMISSIONS FOR ALL MODELS")
print("="*80)

# Load processed features
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_test_scaled = data['scaled_data']['standard']['test_full']
X_test_full = data['X_test_full']
test_ids = data['test_ids']

submissions = []

# 1. Logistic Regression
print("\n1. Logistic Regression...")
with open('best_logistic_regression.pkl', 'rb') as f:
    lr_data = pickle.load(f)
    
predictions = lr_data['model'].predict(X_test_scaled)
submission = pd.DataFrame({'sample_id': test_ids, 'category': predictions})
submission = submission.sort_values('sample_id').reset_index(drop=True)
submission.to_csv('submission_lr.csv', index=False)
submissions.append(('Logistic Regression', lr_data['cv_score'], 'submission_lr.csv'))
print(f"   ✓ CV F1: {lr_data['cv_score']:.4f} → submission_lr.csv")

# 2. SVM
print("\n2. SVM (RBF Kernel)...")
with open('best_svm.pkl', 'rb') as f:
    svm_data = pickle.load(f)
    
predictions = svm_data['model'].predict(X_test_scaled)
submission = pd.DataFrame({'sample_id': test_ids, 'category': predictions})
submission = submission.sort_values('sample_id').reset_index(drop=True)
submission.to_csv('submission_svm.csv', index=False)
submissions.append(('SVM (RBF)', svm_data['cv_score'], 'submission_svm.csv'))
print(f"   ✓ CV F1: {svm_data['cv_score']:.4f} → submission_svm.csv")

# 3. Neural Network
print("\n3. Neural Network (MLP)...")
with open('best_neural_network.pkl', 'rb') as f:
    nn_data = pickle.load(f)
    
predictions = nn_data['model'].predict(X_test_scaled)
submission = pd.DataFrame({'sample_id': test_ids, 'category': predictions})
submission = submission.sort_values('sample_id').reset_index(drop=True)
submission.to_csv('submission_nn.csv', index=False)
submissions.append(('Neural Network', nn_data['cv_score'], 'submission_nn.csv'))
print(f"   ✓ CV F1: {nn_data['cv_score']:.4f} → submission_nn.csv")

# 4. Ensemble (Extra Trees)
print("\n4. Ensemble (Extra Trees)...")
with open('best_ensemble.pkl', 'rb') as f:
    ens_data = pickle.load(f)
    
predictions = ens_data['model'].predict(X_test_full)  # Tree models use unscaled features
submission = pd.DataFrame({'sample_id': test_ids, 'category': predictions})
submission = submission.sort_values('sample_id').reset_index(drop=True)
submission.to_csv('submission_extra_trees.csv', index=False)
submissions.append(('Extra Trees', ens_data['cv_score'], 'submission_extra_trees.csv'))
print(f"   ✓ CV F1: {ens_data['cv_score']:.4f} → submission_extra_trees.csv")

# 5. Weighted Ensemble (already created as submission.csv)
print("\n5. Weighted Ensemble (All Models)...")
with open('best_final_model.pkl', 'rb') as f:
    final_data = pickle.load(f)
submissions.append(('Weighted Ensemble', final_data['cv_score'], 'submission.csv'))
print(f"   ✓ CV F1: {final_data['cv_score']:.4f} → submission.csv (already created)")

print("\n" + "="*80)
print("SUMMARY OF ALL SUBMISSIONS")
print("="*80)

# Create summary DataFrame
summary_df = pd.DataFrame(submissions, columns=['Model', 'CV Macro F1', 'Filename'])
summary_df = summary_df.sort_values('CV Macro F1', ascending=False).reset_index(drop=True)
summary_df['Rank'] = range(1, len(summary_df) + 1)
summary_df = summary_df[['Rank', 'Model', 'CV Macro F1', 'Filename']]

print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print(f"🥇 BEST: {summary_df.iloc[0]['Filename']} (Expected F1: {summary_df.iloc[0]['CV Macro F1']:.4f})")
print(f"🥈 2nd:  {summary_df.iloc[1]['Filename']} (Expected F1: {summary_df.iloc[1]['CV Macro F1']:.4f})")
print(f"🥉 3rd:  {summary_df.iloc[2]['Filename']} (Expected F1: {summary_df.iloc[2]['CV Macro F1']:.4f})")

print("\n💡 Tip: Try submitting the top 2-3 to see which performs best on test set!")
print("="*80)

# Verify all files exist
print("\nVerifying all submission files...")
import os
for _, row in summary_df.iterrows():
    filename = row['Filename']
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        print(f"✓ {filename}: {len(df)} predictions")
    else:
        print(f"✗ {filename}: NOT FOUND")

print("\n" + "="*80)
print("ALL SUBMISSIONS GENERATED!")
print("="*80)
