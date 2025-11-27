"""
Generate Submissions from SVM/MLP GridSearch Results
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("GENERATING SUBMISSIONS FROM SVM/MLP GRIDSEARCH")
print("="*80)

# Load processed data
with open('processed_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_test_full = data['X_test_full']
test_ids = data['test_ids']

# Load GridSearch results
try:
    with open('svm_mlp_gridsearch.pkl', 'rb') as f:
        grid_data = pickle.load(f)
    
    models = grid_data['models']
    scaler = grid_data['scaler']
    
    X_test_scaled = scaler.transform(X_test_full)
    
    print("\nLoaded models from GridSearch:")
    for name, model_data in sorted(models.items(), key=lambda x: x[1]['score'], reverse=True):
        print(f"  {name:15s}: {model_data['score']:.4f}")
    
    print("\n" + "="*80)
    print("GENERATING INDIVIDUAL SUBMISSIONS")
    print("="*80)
    
    submission_files = []
    
    for name, model_data in models.items():
        model = model_data['model']
        score = model_data['score']
        scaled = model_data['scaled']
        
        # Generate predictions
        if scaled:
            predictions = model.predict(X_test_scaled)
        else:
            predictions = model.predict(X_test_full)
        
        # Create submission
        submission = pd.DataFrame({
            'sample_id': test_ids,
            'category': predictions
        })
        submission = submission.sort_values('sample_id').reset_index(drop=True)
        
        # Save to file
        safe_name = name.replace('_', '').lower()
        filename = f'submission_grid_{safe_name}.csv'
        submission.to_csv(filename, index=False)
        
        submission_files.append((filename, score, name))
        print(f"✓ {filename:<40} (CV F1: {score:.4f})")
    
    # Create ensemble of all models
    print("\n" + "="*80)
    print("CREATING WEIGHTED ENSEMBLE")
    print("="*80)
    
    all_probas = []
    weights = []
    
    for name, model_data in models.items():
        model = model_data['model']
        score = model_data['score']
        scaled = model_data['scaled']
        
        if scaled:
            probas = model.predict_proba(X_test_scaled)
        else:
            probas = model.predict_proba(X_test_full)
        
        all_probas.append(probas)
        weights.append(score)
        print(f"  {name:15s}: weight = {score:.4f}")
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Weighted average
    ensemble_probas = np.zeros_like(all_probas[0])
    for probas, weight in zip(all_probas, weights):
        ensemble_probas += probas * weight
    
    # Get predictions
    pred_indices = ensemble_probas.argmax(axis=1)
    class_labels = list(models.values())[0]['model'].classes_
    final_predictions = class_labels[pred_indices]
    
    # Create ensemble submission
    submission = pd.DataFrame({
        'sample_id': test_ids,
        'category': final_predictions
    })
    submission = submission.sort_values('sample_id').reset_index(drop=True)
    submission.to_csv('submission_grid_ensemble.csv', index=False)
    
    ensemble_score = np.mean([m['score'] for m in models.values()])
    print(f"\n✓ submission_grid_ensemble.csv (Expected: {ensemble_score:.4f})")
    
    # Summary
    print("\n" + "="*80)
    print("SUBMISSION FILES CREATED")
    print("="*80)
    
    all_submissions = sorted(submission_files, key=lambda x: x[1], reverse=True)
    all_submissions.append(('submission_grid_ensemble.csv', ensemble_score, 'Weighted Ensemble'))
    
    for filename, score, model_name in all_submissions:
        print(f"  {filename:<40} {score:.4f} - {model_name}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    best_file, best_score, best_name = all_submissions[0]
    print(f"🏆 Try: {best_file}")
    print(f"   Model: {best_name}")
    print(f"   Expected CV F1: {best_score:.4f}")
    print("="*80)
    
except FileNotFoundError:
    print("\n✗ svm_mlp_gridsearch.pkl not found yet")
    print("   GridSearch is still running. Please wait for completion.")
except Exception as e:
    print(f"\n✗ Error: {e}")
