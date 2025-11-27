"""
Data Exploration and Analysis for SignalCluster Classification
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"\nTraining data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

print("\n" + "="*80)
print("TRAINING DATA INFO")
print("="*80)
print(train_df.info())
print("\nFirst few rows:")
print(train_df.head(10))

print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print(train_df.describe())

print("\n" + "="*80)
print("CLASS DISTRIBUTION")
print("="*80)
class_counts = train_df['category'].value_counts()
print(class_counts)
print("\nClass proportions:")
print(train_df['category'].value_counts(normalize=True))

print("\n" + "="*80)
print("MISSING VALUES CHECK")
print("="*80)
print("Training data missing values:")
print(train_df.isnull().sum())
print("\nTest data missing values:")
print(test_df.isnull().sum())

print("\n" + "="*80)
print("FEATURE STATISTICS BY CLASS")
print("="*80)
print(train_df.groupby('category')[['signal_strength', 'response_level']].describe())

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Scatter plot of features colored by class
ax1 = axes[0, 0]
for category in train_df['category'].unique():
    mask = train_df['category'] == category
    ax1.scatter(train_df.loc[mask, 'signal_strength'], 
                train_df.loc[mask, 'response_level'],
                label=category, alpha=0.6, s=20)
ax1.set_xlabel('Signal Strength')
ax1.set_ylabel('Response Level')
ax1.set_title('Feature Distribution by Category')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Signal strength distribution
ax2 = axes[0, 1]
for category in train_df['category'].unique():
    train_df[train_df['category'] == category]['signal_strength'].hist(
        bins=30, alpha=0.5, label=category, ax=ax2)
ax2.set_xlabel('Signal Strength')
ax2.set_ylabel('Frequency')
ax2.set_title('Signal Strength Distribution by Category')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Response level distribution
ax3 = axes[0, 2]
for category in train_df['category'].unique():
    train_df[train_df['category'] == category]['response_level'].hist(
        bins=30, alpha=0.5, label=category, ax=ax3)
ax3.set_xlabel('Response Level')
ax3.set_ylabel('Frequency')
ax3.set_title('Response Level Distribution by Category')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Box plots for signal_strength
ax4 = axes[1, 0]
train_df.boxplot(column='signal_strength', by='category', ax=ax4)
ax4.set_title('Signal Strength by Category')
ax4.set_xlabel('Category')
ax4.set_ylabel('Signal Strength')
plt.sca(ax4)
plt.xticks(rotation=0)

# 5. Box plots for response_level
ax5 = axes[1, 1]
train_df.boxplot(column='response_level', by='category', ax=ax5)
ax5.set_title('Response Level by Category')
ax5.set_xlabel('Category')
ax5.set_ylabel('Response Level')
plt.sca(ax5)
plt.xticks(rotation=0)

# 6. Class distribution bar plot
ax6 = axes[1, 2]
class_counts.plot(kind='bar', ax=ax6, color=['skyblue', 'lightcoral', 'lightgreen'])
ax6.set_title('Class Distribution')
ax6.set_xlabel('Category')
ax6.set_ylabel('Count')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=150, bbox_inches='tight')
print("\n" + "="*80)
print("Visualization saved as 'data_exploration.png'")
print("="*80)

# Correlation analysis
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)
# Encode categories for correlation
le = LabelEncoder()
train_df_encoded = train_df.copy()
train_df_encoded['category_encoded'] = le.fit_transform(train_df['category'])
correlation_matrix = train_df_encoded[['signal_strength', 'response_level', 'category_encoded']].corr()
print(correlation_matrix)

# Create correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("Correlation heatmap saved as 'correlation_heatmap.png'")

print("\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)
print("1. Number of features: 2 (signal_strength, response_level)")
print("2. Number of classes:", train_df['category'].nunique())
print("3. Classes:", list(train_df['category'].unique()))
print("4. Class imbalance ratio:", f"{class_counts.max() / class_counts.min():.2f}:1")
print("5. Feature ranges:")
print(f"   - Signal Strength: [{train_df['signal_strength'].min():.2f}, {train_df['signal_strength'].max():.2f}]")
print(f"   - Response Level: [{train_df['response_level'].min():.2f}, {train_df['response_level'].max():.2f}]")
print("="*80)
