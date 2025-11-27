"""
Feature Engineering for SignalCluster Classification
Generates polynomial features, interaction terms, and domain-specific features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
import pickle

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("FEATURE ENGINEERING PIPELINE")
print("="*80)

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"\nOriginal features: {train_df.shape[1] - 2} (excluding sample_id and category)")

# Extract features and target
X_train = train_df[['signal_strength', 'response_level']].values
y_train = train_df['category'].values
X_test = test_df[['signal_strength', 'response_level']].values

# Store sample IDs
train_ids = train_df['sample_id'].values
test_ids = test_df['sample_id'].values

print("\n" + "="*80)
print("1. BASIC FEATURES")
print("="*80)
feature_names = ['signal_strength', 'response_level']
print(f"Features: {feature_names}")

# Create DataFrame for all engineered features
train_features = pd.DataFrame(X_train, columns=feature_names)
test_features = pd.DataFrame(X_test, columns=feature_names)

print("\n" + "="*80)
print("2. DOMAIN-SPECIFIC FEATURES")
print("="*80)

# Distance from origin
train_features['distance_from_origin'] = np.sqrt(
    train_features['signal_strength']**2 + train_features['response_level']**2
)
test_features['distance_from_origin'] = np.sqrt(
    test_features['signal_strength']**2 + test_features['response_level']**2
)
print("✓ Added: distance_from_origin")

# Ratio features (with epsilon to avoid division by zero)
epsilon = 1e-8
train_features['signal_response_ratio'] = train_features['signal_strength'] / (train_features['response_level'] + epsilon)
test_features['signal_response_ratio'] = test_features['signal_strength'] / (test_features['response_level'] + epsilon)
print("✓ Added: signal_response_ratio")

# Angle features
train_features['angle'] = np.arctan2(train_features['response_level'], train_features['signal_strength'])
test_features['angle'] = np.arctan2(test_features['response_level'], test_features['signal_strength'])
print("✓ Added: angle (arctan2)")

# Sum and difference
train_features['sum_features'] = train_features['signal_strength'] + train_features['response_level']
test_features['sum_features'] = test_features['signal_strength'] + test_features['response_level']
print("✓ Added: sum_features")

train_features['diff_features'] = train_features['signal_strength'] - train_features['response_level']
test_features['diff_features'] = test_features['signal_strength'] - test_features['response_level']
print("✓ Added: diff_features")

print("\n" + "="*80)
print("3. POLYNOMIAL FEATURES")
print("="*80)

# Create polynomial features (degree 2 and 3)
poly2 = PolynomialFeatures(degree=2, include_bias=False)
poly3 = PolynomialFeatures(degree=3, include_bias=False)

# Use only original 2 features for polynomial expansion
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)

X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)

poly2_feature_names = poly2.get_feature_names_out(['signal_strength', 'response_level'])
poly3_feature_names = poly3.get_feature_names_out(['signal_strength', 'response_level'])

print(f"✓ Degree 2 polynomial features: {len(poly2_feature_names)}")
print(f"✓ Degree 3 polynomial features: {len(poly3_feature_names)}")

# Add polynomial features to our main feature sets
train_poly2_df = pd.DataFrame(X_train_poly2, columns=poly2_feature_names)
test_poly2_df = pd.DataFrame(X_test_poly2, columns=poly2_feature_names)

train_poly3_df = pd.DataFrame(X_train_poly3, columns=poly3_feature_names)
test_poly3_df = pd.DataFrame(X_test_poly3, columns=poly3_feature_names)

# Combine all features
print("\n" + "="*80)
print("4. FEATURE COMBINATIONS")
print("="*80)

# Base features + domain features
train_full = train_features.copy()
test_full = test_features.copy()
print(f"✓ Base + Domain features: {train_full.shape[1]} features")

# Add polynomial degree 2
for col in poly2_feature_names:
    if col not in train_full.columns:  # Avoid duplicates
        train_full[f'poly2_{col}'] = train_poly2_df[col]
        test_full[f'poly2_{col}'] = test_poly2_df[col]

print(f"✓ With Polynomial degree 2: {train_full.shape[1]} features")

print("\n" + "="*80)
print("5. FEATURE SCALING")
print("="*80)

# Create different scaled versions
scalers = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler()
}

scaled_data = {}

for scaler_name, scaler in scalers.items():
    # Scale basic features only (for simple models)
    X_train_scaled_basic = scaler.fit_transform(X_train)
    X_test_scaled_basic = scaler.transform(X_test)
    
    # Scale full features (for complex models)
    X_train_scaled_full = scaler.fit_transform(train_full)
    X_test_scaled_full = scaler.transform(test_full)
    
    scaled_data[scaler_name] = {
        'scaler': scaler,
        'train_basic': X_train_scaled_basic,
        'test_basic': X_test_scaled_basic,
        'train_full': X_train_scaled_full,
        'test_full': X_test_scaled_full
    }
    
    print(f"✓ {scaler_name.upper()} scaler fitted")

print("\n" + "="*80)
print("6. SAVING PROCESSED DATA")
print("="*80)

# Save all data
data_package = {
    'train_ids': train_ids,
    'test_ids': test_ids,
    'y_train': y_train,
    'X_train_raw': X_train,
    'X_test_raw': X_test,
    'X_train_full': train_full.values,
    'X_test_full': test_full.values,
    'X_train_poly3': X_train_poly3,
    'X_test_poly3': X_test_poly3,
    'feature_names_full': train_full.columns.tolist(),
    'feature_names_poly3': poly3_feature_names.tolist(),
    'scaled_data': scaled_data,
    'random_state': 42
}

with open('processed_features.pkl', 'wb') as f:
    pickle.dump(data_package, f)

print("✓ Saved processed_features.pkl")
print(f"  - Raw features: {X_train.shape[1]}")
print(f"  - Full features: {train_full.shape[1]}")
print(f"  - Poly3 features: {X_train_poly3.shape[1]}")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")
print(f"  - Classes: {len(np.unique(y_train))}")

print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE")
print("="*80)
print(f"\nTotal feature sets created:")
print(f"  1. Raw (2 features)")
print(f"  2. Full engineered ({train_full.shape[1]} features)")
print(f"  3. Polynomial degree 3 ({X_train_poly3.shape[1]} features)")
print(f"  4. Each with 3 scaling methods (Standard, MinMax, Robust)")
print("="*80)
