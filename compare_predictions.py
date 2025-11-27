"""
Compare Predictions Across All Models
"""
import pandas as pd
import numpy as np

print("="*80)
print("COMPARING PREDICTIONS ACROSS ALL MODELS")
print("="*80)

# Load all submissions
submissions = {
    'Weighted Ensemble (0.9944)': pd.read_csv('submission.csv'),
    'SVM (0.9872)': pd.read_csv('submission_svm.csv'),
    'Extra Trees (0.9868)': pd.read_csv('submission_extra_trees.csv'),
    'Neural Network (0.9849)': pd.read_csv('submission_nn.csv'),
    'Logistic Reg (0.9814)': pd.read_csv('submission_lr.csv')
}

# Merge all predictions
merged = submissions['Weighted Ensemble (0.9944)'][['sample_id']].copy()
for name, df in submissions.items():
    merged[name] = df['category']

# Check agreement
print("\n1. PREDICTION AGREEMENT ANALYSIS")
print("="*80)

# Count how many models agree on each prediction
agreement_counts = []
for idx, row in merged.iterrows():
    predictions = row[1:].values  # Skip sample_id
    unique_preds = len(set(predictions))
    agreement_counts.append(5 - unique_preds + 1)  # How many agree

merged['Agreement'] = agreement_counts

print(f"Unanimous (all 5 models agree): {sum(merged['Agreement'] == 5)} samples ({sum(merged['Agreement'] == 5)/len(merged)*100:.1f}%)")
print(f"4 models agree: {sum(merged['Agreement'] == 4)} samples ({sum(merged['Agreement'] == 4)/len(merged)*100:.1f}%)")
print(f"3 models agree: {sum(merged['Agreement'] == 3)} samples ({sum(merged['Agreement'] == 3)/len(merged)*100:.1f}%)")
print(f"No consensus (all different): {sum(merged['Agreement'] == 1)} samples ({sum(merged['Agreement'] == 1)/len(merged)*100:.1f}%)")

# Show some examples where models disagree
print("\n2. EXAMPLES OF MODEL DISAGREEMENT")
print("="*80)
disagreements = merged[merged['Agreement'] < 5].head(10)
if len(disagreements) > 0:
    print(disagreements.to_string(index=False))
else:
    print("All models agree on all predictions!")

# Class distribution comparison
print("\n3. PREDICTED CLASS DISTRIBUTIONS")
print("="*80)
print(f"{'Model':<30} {'Group_A':<12} {'Group_B':<12} {'Group_C':<12}")
print("-"*80)

for name, df in submissions.items():
    counts = df['category'].value_counts()
    a = counts.get('Group_A', 0)
    b = counts.get('Group_B', 0)
    c = counts.get('Group_C', 0)
    print(f"{name:<30} {a:<4} ({a/len(df)*100:>5.1f}%) {b:<4} ({b/len(df)*100:>5.1f}%) {c:<4} ({c/len(df)*100:>5.1f}%)")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print("📊 Primary submission: submission.csv (Weighted Ensemble, 0.9944 CV F1)")
print("🔄 Backup option: submission_svm.csv (SVM RBF, 0.9872 CV F1)")
print("\nBoth are excellent options. The ensemble is more robust but SVM is simpler.")
print("="*80)

# Save comparison
merged.to_csv('prediction_comparison.csv', index=False)
print("\n✓ Detailed comparison saved to: prediction_comparison.csv")
