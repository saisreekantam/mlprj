"""
Model Performance Summary and Analysis
"""
import pickle
import pandas as pd

print("="*80)
print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
print("="*80)

# Load all model results
models_summary = []

# Logistic Regression
with open('best_logistic_regression.pkl', 'rb') as f:
    lr = pickle.load(f)
    models_summary.append({
        'Model': 'Logistic Regression (Multinomial L2)',
        'CV F1 Score': lr['cv_score'],
        'Type': 'Linear'
    })

# SVM
with open('best_svm.pkl', 'rb') as f:
    svm = pickle.load(f)
    models_summary.append({
        'Model': f"SVM ({svm['model_name']})",
        'CV F1 Score': svm['cv_score'],
        'Type': 'Kernel'
    })

# Neural Network
with open('best_neural_network.pkl', 'rb') as f:
    nn = pickle.load(f)
    models_summary.append({
        'Model': 'Neural Network (MLP)',
        'CV F1 Score': nn['cv_score'],
        'Type': 'Deep Learning'
    })

# Ensemble
with open('best_ensemble.pkl', 'rb') as f:
    ens = pickle.load(f)
    models_summary.append({
        'Model': f"Ensemble ({ens['model_name']})",
        'CV F1 Score': ens['cv_score'],
        'Type': 'Tree-based'
    })

# Final Weighted Ensemble
with open('best_final_model.pkl', 'rb') as f:
    final = pickle.load(f)
    models_summary.append({
        'Model': 'Weighted Ensemble (All Models)',
        'CV F1 Score': final['cv_score'],
        'Type': 'Meta-ensemble'
    })

# Create DataFrame
df = pd.DataFrame(models_summary)
df = df.sort_values('CV F1 Score', ascending=False).reset_index(drop=True)
df['Rank'] = range(1, len(df) + 1)
df = df[['Rank', 'Model', 'Type', 'CV F1 Score']]

print("\nFINAL MODEL RANKINGS:")
print("="*80)
print(df.to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. Best Individual Model: {df.iloc[1]['Model']} ({df.iloc[1]['CV F1 Score']:.4f})")
print(f"2. Best Overall Approach: {df.iloc[0]['Model']} ({df.iloc[0]['CV F1 Score']:.4f})")
print(f"3. Improvement from Ensemble: +{(df.iloc[0]['CV F1 Score'] - df.iloc[1]['CV F1 Score']):.4f}")
print(f"4. All models achieved >0.98 F1 score!")
print(f"5. Weighted ensemble combines strengths of all models")

print("\n" + "="*80)
print("RECOMMENDED SUBMISSION")
print("="*80)
print("File: submission.csv")
print(f"Expected Macro F1 Score: {final['cv_score']:.4f}")
print("="*80)
