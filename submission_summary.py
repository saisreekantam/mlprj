"""
Quick Summary of Current Best Submissions
"""
import pandas as pd
import os

print("="*80)
print("CURRENT SUBMISSION FILES SUMMARY")
print("="*80)

submissions_info = [
    ('submission_optimized_ensemble.csv', 0.9953, 'Weighted ensemble of top 5 models (10-fold CV)'),
    ('submission_optimized_svm_v2.csv', 0.9872, 'Optimized SVM with RBF kernel'),
    ('submission_conservative_gb_v2.csv', 0.9858, 'Conservative Gradient Boosting'),
    ('submission_bagged_et_v2.csv', 0.9833, 'Bagged Extra Trees'),
    ('submission_svm.csv', 0.9872, 'Original RBF SVM (5-fold CV)'),
    ('submission.csv', 0.9944, 'Original weighted ensemble (5-fold CV)'),
]

print(f"\n{'Filename':<45} {'Expected F1':>12} {'Description':<50}")
print("-" * 110)

for filename, expected_f1, description in sorted(submissions_info, key=lambda x: x[1], reverse=True):
    if os.path.exists(f'/Users/sreekantamsaivenkat/Documents/mlprj/{filename}'):
        status = "✓"
        df = pd.read_csv(filename)
        size = len(df)
        print(f"{status} {filename:<43} {expected_f1:>10.4f}   {description:<50}")

print("\n" + "="*80)
print("RECOMMENDATIONS FOR SUBMISSION")
print("="*80)
print("🏆 PRIMARY: submission_optimized_ensemble.csv (0.9953 expected)")
print("   - Uses 10-fold CV (more robust)")
print("   - Ensemble of 5 best models")
print("   - Most conservative estimate")
print()
print("🥈 BACKUP: submission_optimized_svm_v2.csv (0.9872 expected)")
print("   - Single best performing model")
print("   - Good if ensemble overfits")
print()
print("🥉 ALTERNATIVE: submission_conservative_gb_v2.csv (0.9858 expected)")
print("   - Different algorithm family")
print("   - Conservative hyperparameters")
print("="*80)

print("\n📊 TEST SCORE ANALYSIS (from your Kaggle submissions):")
print("   - submission_nn.csv:  0.985")
print("   - submission_lr.csv:  0.985")
print("   - submission_svm.csv: 0.981")
print("   - submission.csv:     0.985")
print()
print("   Gap between CV (0.994) and test (0.985) suggests some overfitting")
print("   New optimized models use 10-fold CV and conservative params")
print("   Expected to generalize better!")
print("="*80)
