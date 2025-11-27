"""
Advanced Multi-Class Classification with Sophisticated SVM-NN Ensemble Strategies
Dataset: Signal characteristics classification into Group_A, Group_B, Group_C
Focus: Multiple ensemble techniques combining SVMs and Neural Networks
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("ADVANCED ENSEMBLE: Sophisticated SVM-NN Combination Strategies")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING WITH MULTIPLE SCALERS
# ============================================================================
print("\n[1] Loading and Preprocessing Data with Multiple Scaling Strategies...")

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Categories: {train_df['category'].unique()}")
print(f"Category distribution:\n{train_df['category'].value_counts()}")

# Prepare features and labels
X_train_full = train_df[['signal_strength', 'response_level']].values
y_train_full = train_df['category'].values
X_test = test_df[['signal_strength', 'response_level']].values
test_ids = test_df['sample_id'].values

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_full)
n_classes = len(label_encoder.classes_)
print(f"\nNumber of classes: {n_classes}")
print(f"Class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Multiple scalers for diversity
scaler_standard = StandardScaler()
scaler_robust = RobustScaler()
scaler_minmax = MinMaxScaler()

X_train_standard = scaler_standard.fit_transform(X_train_full)
X_test_standard = scaler_standard.transform(X_test)

X_train_robust = scaler_robust.fit_transform(X_train_full)
X_test_robust = scaler_robust.transform(X_test)

X_train_minmax = scaler_minmax.fit_transform(X_train_full)
X_test_minmax = scaler_minmax.transform(X_test)

# Split for validation
X_train_s, X_val_s, y_train, y_val = train_test_split(
    X_train_standard, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)
X_train_r, X_val_r, _, _ = train_test_split(
    X_train_robust, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)
X_train_m, X_val_m, _, _ = train_test_split(
    X_train_minmax, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print(f"\nTraining set: {X_train_s.shape}")
print(f"Validation set: {X_val_s.shape}")

# ============================================================================
# 2. DIVERSE SVM MODELS WITH MULTIPLE CONFIGURATIONS
# ============================================================================
print("\n" + "="*80)
print("[2] Training Diverse SVM Models")
print("="*80)

svm_models_list = []

# 2.1 Linear SVMs with different C values
print("\n[2.1] Linear SVMs with different regularization...")
for c_value in [0.1, 1.0, 10.0]:
    svm = SVC(kernel='linear', C=c_value, random_state=42, probability=True)
    svm.fit(X_train_s, y_train)
    acc = accuracy_score(y_val, svm.predict(X_val_s))
    svm_models_list.append(('Linear_C' + str(c_value), svm, acc, 'standard'))
    print(f"  Linear SVM (C={c_value}): {acc:.4f}")

# 2.2 RBF SVMs with different parameters
print("\n[2.2] RBF SVMs with different parameters...")
for c_value in [1.0, 10.0, 100.0]:
    for gamma_value in ['scale', 0.01, 0.1]:
        svm = SVC(kernel='rbf', C=c_value, gamma=gamma_value, random_state=42, probability=True)
        svm.fit(X_train_s, y_train)
        acc = accuracy_score(y_val, svm.predict(X_val_s))
        svm_models_list.append((f'RBF_C{c_value}_g{gamma_value}', svm, acc, 'standard'))
        print(f"  RBF SVM (C={c_value}, gamma={gamma_value}): {acc:.4f}")

# 2.3 Polynomial SVMs with different degrees
print("\n[2.3] Polynomial SVMs with different degrees...")
for degree in [2, 3, 4]:
    for c_value in [1.0, 10.0]:
        svm = SVC(kernel='poly', degree=degree, C=c_value, random_state=42, probability=True)
        svm.fit(X_train_s, y_train)
        acc = accuracy_score(y_val, svm.predict(X_val_s))
        svm_models_list.append((f'Poly_d{degree}_C{c_value}', svm, acc, 'standard'))
        print(f"  Polynomial SVM (degree={degree}, C={c_value}): {acc:.4f}")

# 2.4 SVMs with Robust Scaler
print("\n[2.4] SVMs with Robust Scaler...")
svm_robust = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42, probability=True)
svm_robust.fit(X_train_r, y_train)
acc_robust = accuracy_score(y_val, svm_robust.predict(X_val_r))
svm_models_list.append(('RBF_Robust', svm_robust, acc_robust, 'robust'))
print(f"  RBF SVM with Robust Scaler: {acc_robust:.4f}")

# 2.5 SVMs with MinMax Scaler
print("\n[2.5] SVMs with MinMax Scaler...")
svm_minmax = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42, probability=True)
svm_minmax.fit(X_train_m, y_train)
acc_minmax = accuracy_score(y_val, svm_minmax.predict(X_val_m))
svm_models_list.append(('RBF_MinMax', svm_minmax, acc_minmax, 'minmax'))
print(f"  RBF SVM with MinMax Scaler: {acc_minmax:.4f}")

print(f"\nTotal SVM models trained: {len(svm_models_list)}")

# Select top 5 SVM models
svm_models_list.sort(key=lambda x: x[2], reverse=True)
top_5_svms = svm_models_list[:5]
print(f"\nTop 5 SVM Models:")
for name, model, acc, scaler_type in top_5_svms:
    print(f"  • {name:30s}: {acc:.4f} (scaler: {scaler_type})")

# ============================================================================
# 3. DIVERSE NEURAL NETWORK ARCHITECTURES
# ============================================================================
print("\n" + "="*80)
print("[3] Training Diverse Neural Network Architectures")
print("="*80)

y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_val_cat = to_categorical(y_val, num_classes=n_classes)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7, verbose=0)

nn_models_list = []

# 3.1 Attention-based Neural Network
print("\n[3.1] Attention-based Neural Network...")

def attention_block(inputs):
    attention = layers.Dense(inputs.shape[-1], activation='tanh')(inputs)
    attention = layers.Dense(inputs.shape[-1], activation='softmax')(attention)
    return layers.Multiply()([inputs, attention])

input_layer = layers.Input(shape=(2,))
x = layers.Dense(256, activation='relu')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = attention_block(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
output_layer = layers.Dense(n_classes, activation='softmax')(x)
model_attention = Model(inputs=input_layer, outputs=output_layer)
model_attention.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_attention = model_attention.fit(X_train_s, y_train_cat, validation_data=(X_val_s, y_val_cat), 
                                        epochs=250, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=0)
val_acc_attention = max(history_attention.history['val_accuracy'])
nn_models_list.append(('Attention', model_attention, val_acc_attention, 'standard'))
print(f"Attention NN Validation Accuracy: {val_acc_attention:.4f}")

# 3.2 Multi-Scale Neural Network
print("\n[3.2] Multi-Scale Neural Network...")
input_layer = layers.Input(shape=(2,))

# Scale 1: Deep path
deep1 = layers.Dense(256, activation='relu')(input_layer)
deep1 = layers.BatchNormalization()(deep1)
deep1 = layers.Dropout(0.5)(deep1)
deep1 = layers.Dense(128, activation='relu')(deep1)
deep1 = layers.BatchNormalization()(deep1)
deep1 = layers.Dropout(0.4)(deep1)

# Scale 2: Medium path
deep2 = layers.Dense(128, activation='relu')(input_layer)
deep2 = layers.BatchNormalization()(deep2)
deep2 = layers.Dropout(0.4)(deep2)
deep2 = layers.Dense(64, activation='relu')(deep2)

# Scale 3: Shallow path
shallow = layers.Dense(64, activation='relu')(input_layer)

# Combine all scales
combined = layers.Concatenate()([deep1, deep2, shallow])
x = layers.Dense(128, activation='relu')(combined)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
output_layer = layers.Dense(n_classes, activation='softmax')(x)
model_multiscale = Model(inputs=input_layer, outputs=output_layer)
model_multiscale.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_multiscale = model_multiscale.fit(X_train_s, y_train_cat, validation_data=(X_val_s, y_val_cat),
                                          epochs=250, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=0)
val_acc_multiscale = max(history_multiscale.history['val_accuracy'])
nn_models_list.append(('MultiScale', model_multiscale, val_acc_multiscale, 'standard'))
print(f"Multi-Scale NN Validation Accuracy: {val_acc_multiscale:.4f}")

# 3.3 Dense Connection Network (DenseNet-style)
print("\n[3.3] Dense Connection Network...")
input_layer = layers.Input(shape=(2,))
x0 = layers.Dense(128, activation='relu')(input_layer)
x0 = layers.BatchNormalization()(x0)

x1 = layers.Dense(128, activation='relu')(x0)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Dropout(0.4)(x1)

concat1 = layers.Concatenate()([x0, x1])
x2 = layers.Dense(128, activation='relu')(concat1)
x2 = layers.BatchNormalization()(x2)
x2 = layers.Dropout(0.4)(x2)

concat2 = layers.Concatenate()([x0, x1, x2])
x3 = layers.Dense(64, activation='relu')(concat2)
x3 = layers.BatchNormalization()(x3)
x3 = layers.Dropout(0.3)(x3)

x = layers.Dense(64, activation='relu')(x3)
output_layer = layers.Dense(n_classes, activation='softmax')(x)
model_dense = Model(inputs=input_layer, outputs=output_layer)
model_dense.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_dense = model_dense.fit(X_train_s, y_train_cat, validation_data=(X_val_s, y_val_cat),
                                epochs=250, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=0)
val_acc_dense = max(history_dense.history['val_accuracy'])
nn_models_list.append(('DenseNet', model_dense, val_acc_dense, 'standard'))
print(f"Dense Connection NN Validation Accuracy: {val_acc_dense:.4f}")

# 3.4 Pyramidal Neural Network
print("\n[3.4] Pyramidal Neural Network...")
model_pyramid = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(2,)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])
model_pyramid.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_pyramid = model_pyramid.fit(X_train_s, y_train_cat, validation_data=(X_val_s, y_val_cat),
                                    epochs=250, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=0)
val_acc_pyramid = max(history_pyramid.history['val_accuracy'])
nn_models_list.append(('Pyramid', model_pyramid, val_acc_pyramid, 'standard'))
print(f"Pyramidal NN Validation Accuracy: {val_acc_pyramid:.4f}")

# 3.5 NN with Different Activations (ELU)
print("\n[3.5] Neural Network with ELU activation...")
model_elu = keras.Sequential([
    layers.Dense(256, activation='elu', input_shape=(2,)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='elu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='elu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='elu'),
    layers.Dropout(0.2),
    layers.Dense(n_classes, activation='softmax')
])
model_elu.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_elu = model_elu.fit(X_train_s, y_train_cat, validation_data=(X_val_s, y_val_cat),
                            epochs=250, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=0)
val_acc_elu = max(history_elu.history['val_accuracy'])
nn_models_list.append(('ELU', model_elu, val_acc_elu, 'standard'))
print(f"ELU NN Validation Accuracy: {val_acc_elu:.4f}")

# 3.6 NN with SELU activation (Self-Normalizing)
print("\n[3.6] Self-Normalizing Neural Network (SELU)...")
model_selu = keras.Sequential([
    layers.Dense(256, activation='selu', kernel_initializer='lecun_normal', input_shape=(2,)),
    layers.AlphaDropout(0.1),
    layers.Dense(128, activation='selu', kernel_initializer='lecun_normal'),
    layers.AlphaDropout(0.1),
    layers.Dense(64, activation='selu', kernel_initializer='lecun_normal'),
    layers.AlphaDropout(0.1),
    layers.Dense(32, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(n_classes, activation='softmax')
])
model_selu.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_selu = model_selu.fit(X_train_s, y_train_cat, validation_data=(X_val_s, y_val_cat),
                              epochs=250, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=0)
val_acc_selu = max(history_selu.history['val_accuracy'])
nn_models_list.append(('SELU', model_selu, val_acc_selu, 'standard'))
print(f"SELU NN Validation Accuracy: {val_acc_selu:.4f}")

# 3.7 Ensemble of 10 different seed models
print("\n[3.7] Training 10 NN models with different seeds...")
seed_models = []
seed_accs = []
for i in range(10):
    tf.random.set_seed(100 + i)
    np.random.seed(100 + i)
    
    model_seed = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(2,)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ])
    model_seed.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history_seed = model_seed.fit(X_train_s, y_train_cat, validation_data=(X_val_s, y_val_cat),
                                  epochs=200, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=0)
    val_acc_seed = max(history_seed.history['val_accuracy'])
    seed_models.append(model_seed)
    seed_accs.append(val_acc_seed)
    print(f"  Seed Model {i+1}: {val_acc_seed:.4f}")

avg_seed_acc = np.mean(seed_accs)
print(f"Average Seed Models Accuracy: {avg_seed_acc:.4f}")

# 3.8 NNs with different scalers
print("\n[3.8] Training NNs with different scalers...")

# NN with Robust Scaler
y_train_r_cat = to_categorical(y_train, num_classes=n_classes)
y_val_r_cat = to_categorical(y_val, num_classes=n_classes)

model_nn_robust = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(2,)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation='softmax')
])
model_nn_robust.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_nn_robust = model_nn_robust.fit(X_train_r, y_train_r_cat, validation_data=(X_val_r, y_val_r_cat),
                                        epochs=250, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=0)
val_acc_nn_robust = max(history_nn_robust.history['val_accuracy'])
nn_models_list.append(('NN_Robust', model_nn_robust, val_acc_nn_robust, 'robust'))
print(f"NN with Robust Scaler: {val_acc_nn_robust:.4f}")

# NN with MinMax Scaler
model_nn_minmax = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(2,)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation='softmax')
])
model_nn_minmax.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_nn_minmax = model_nn_minmax.fit(X_train_m, y_train_cat, validation_data=(X_val_m, y_val_cat),
                                        epochs=250, batch_size=16, callbacks=[early_stopping, reduce_lr], verbose=0)
val_acc_nn_minmax = max(history_nn_minmax.history['val_accuracy'])
nn_models_list.append(('NN_MinMax', model_nn_minmax, val_acc_nn_minmax, 'minmax'))
print(f"NN with MinMax Scaler: {val_acc_nn_minmax:.4f}")

print(f"\nTotal NN architectures trained: {len(nn_models_list)}")

# Select top 10 NN models
nn_models_list.sort(key=lambda x: x[2], reverse=True)
top_10_nns = nn_models_list[:10]
print(f"\nTop 10 NN Models:")
for name, model, acc, scaler_type in top_10_nns:
    print(f"  • {name:30s}: {acc:.4f} (scaler: {scaler_type})")

# ============================================================================
# 4. TRAIN ALL MODELS ON FULL DATA
# ============================================================================
print("\n" + "="*80)
print("[4] Training All Selected Models on Full Data")
print("="*80)

# Train top 5 SVMs on full data
print("\n[4.1] Training top 5 SVMs on full data...")
trained_svms = []
for name, model, acc, scaler_type in top_5_svms:
    if scaler_type == 'standard':
        X_full = X_train_standard
    elif scaler_type == 'robust':
        X_full = X_train_robust
    else:
        X_full = X_train_minmax
    
    model.fit(X_full, y_train_encoded)
    trained_svms.append((name, model, acc, scaler_type))
    print(f"  Trained: {name}")

# Train top 10 NNs on full data
print("\n[4.2] Training top 10 NNs on full data...")
y_full_cat = to_categorical(y_train_encoded, num_classes=n_classes)
trained_nns = []

for i, (name, model, acc, scaler_type) in enumerate(top_10_nns):
    # Recreate model architecture based on name
    if name == 'Attention':
        input_layer = layers.Input(shape=(2,))
        x = layers.Dense(256, activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = attention_block(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        output_layer = layers.Dense(n_classes, activation='softmax')(x)
        new_model = Model(inputs=input_layer, outputs=output_layer)
    elif name == 'MultiScale':
        input_layer = layers.Input(shape=(2,))
        deep1 = layers.Dense(256, activation='relu')(input_layer)
        deep1 = layers.BatchNormalization()(deep1)
        deep1 = layers.Dropout(0.5)(deep1)
        deep1 = layers.Dense(128, activation='relu')(deep1)
        deep1 = layers.BatchNormalization()(deep1)
        deep1 = layers.Dropout(0.4)(deep1)
        deep2 = layers.Dense(128, activation='relu')(input_layer)
        deep2 = layers.BatchNormalization()(deep2)
        deep2 = layers.Dropout(0.4)(deep2)
        deep2 = layers.Dense(64, activation='relu')(deep2)
        shallow = layers.Dense(64, activation='relu')(input_layer)
        combined = layers.Concatenate()([deep1, deep2, shallow])
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        output_layer = layers.Dense(n_classes, activation='softmax')(x)
        new_model = Model(inputs=input_layer, outputs=output_layer)
    elif name == 'DenseNet':
        input_layer = layers.Input(shape=(2,))
        x0 = layers.Dense(128, activation='relu')(input_layer)
        x0 = layers.BatchNormalization()(x0)
        x1 = layers.Dense(128, activation='relu')(x0)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.4)(x1)
        concat1 = layers.Concatenate()([x0, x1])
        x2 = layers.Dense(128, activation='relu')(concat1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.4)(x2)
        concat2 = layers.Concatenate()([x0, x1, x2])
        x3 = layers.Dense(64, activation='relu')(concat2)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Dropout(0.3)(x3)
        x = layers.Dense(64, activation='relu')(x3)
        output_layer = layers.Dense(n_classes, activation='softmax')(x)
        new_model = Model(inputs=input_layer, outputs=output_layer)
    elif name == 'Pyramid':
        new_model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(2,)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(n_classes, activation='softmax')
        ])
    elif name == 'ELU':
        new_model = keras.Sequential([
            layers.Dense(256, activation='elu', input_shape=(2,)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='elu'),
            layers.Dropout(0.2),
            layers.Dense(n_classes, activation='softmax')
        ])
    elif name == 'SELU':
        new_model = keras.Sequential([
            layers.Dense(256, activation='selu', kernel_initializer='lecun_normal', input_shape=(2,)),
            layers.AlphaDropout(0.1),
            layers.Dense(128, activation='selu', kernel_initializer='lecun_normal'),
            layers.AlphaDropout(0.1),
            layers.Dense(64, activation='selu', kernel_initializer='lecun_normal'),
            layers.AlphaDropout(0.1),
            layers.Dense(32, activation='selu', kernel_initializer='lecun_normal'),
            layers.Dense(n_classes, activation='softmax')
        ])
    else:  # NN_Robust, NN_MinMax or default
        new_model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(2,)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation='softmax')
        ])
    
    new_model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if scaler_type == 'standard':
        X_full = X_train_standard
    elif scaler_type == 'robust':
        X_full = X_train_robust
    else:
        X_full = X_train_minmax
    
    new_model.fit(X_full, y_full_cat, epochs=200, batch_size=16, verbose=0)
    trained_nns.append((name, new_model, acc, scaler_type))
    print(f"  Trained: {name}")

# Train 10 seed models on full data
print("\n[4.3] Training 10 seed models on full data...")
trained_seed_models = []
for i in range(10):
    tf.random.set_seed(100 + i)
    np.random.seed(100 + i)
    
    model_seed_final = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(2,)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ])
    model_seed_final.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model_seed_final.fit(X_train_standard, y_full_cat, epochs=150, batch_size=16, verbose=0)
    trained_seed_models.append(model_seed_final)
    print(f"  Trained Seed Model {i+1}")

# ============================================================================
# 5. ADVANCED ENSEMBLE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("[5] Generating Advanced Ensemble Predictions")
print("="*80)

# Helper function to get predictions based on scaler type
def get_test_data(scaler_type):
    if scaler_type == 'standard':
        return X_test_standard
    elif scaler_type == 'robust':
        return X_test_robust
    else:
        return X_test_minmax

# 5.1 Top 5 SVMs Weighted Ensemble
print("\n[5.1] Top 5 SVMs Weighted Soft Voting...")
svm_ensemble_proba = np.zeros((len(X_test), n_classes))
total_svm_weight = sum([acc for _, _, acc, _ in trained_svms])
for name, model, acc, scaler_type in trained_svms:
    X_test_scaled = get_test_data(scaler_type)
    proba = model.predict_proba(X_test_scaled)
    svm_ensemble_proba += proba * acc
svm_ensemble_proba /= total_svm_weight
svm_ensemble_pred = np.argmax(svm_ensemble_proba, axis=1)
svm_ensemble_labels = label_encoder.inverse_transform(svm_ensemble_pred)

# 5.2 Top 10 NNs Weighted Ensemble
print("[5.2] Top 10 NNs Weighted Soft Voting...")
nn_ensemble_proba = np.zeros((len(X_test), n_classes))
total_nn_weight = sum([acc for _, _, acc, _ in trained_nns])
for name, model, acc, scaler_type in trained_nns:
    X_test_scaled = get_test_data(scaler_type)
    proba = model.predict(X_test_scaled, verbose=0)
    nn_ensemble_proba += proba * acc
nn_ensemble_proba /= total_nn_weight
nn_ensemble_pred = np.argmax(nn_ensemble_proba, axis=1)
nn_ensemble_labels = label_encoder.inverse_transform(nn_ensemble_pred)

# 5.3 10 Seed Models Average Ensemble
print("[5.3] 10 Seed Models Average Ensemble...")
seed_ensemble_proba = np.zeros((len(X_test), n_classes))
for model in trained_seed_models:
    proba = model.predict(X_test_standard, verbose=0)
    seed_ensemble_proba += proba
seed_ensemble_proba /= len(trained_seed_models)
seed_ensemble_pred = np.argmax(seed_ensemble_proba, axis=1)
seed_ensemble_labels = label_encoder.inverse_transform(seed_ensemble_pred)

# 5.4 SVM-NN Combined Ensemble (Equal Weight)
print("[5.4] SVM-NN Combined Ensemble (Equal Weight)...")
svm_nn_equal_proba = (svm_ensemble_proba + nn_ensemble_proba) / 2
svm_nn_equal_pred = np.argmax(svm_nn_equal_proba, axis=1)
svm_nn_equal_labels = label_encoder.inverse_transform(svm_nn_equal_pred)

# 5.5 SVM-NN Combined Ensemble (Performance Weighted)
print("[5.5] SVM-NN Combined Ensemble (Performance Weighted)...")
avg_svm_acc = np.mean([acc for _, _, acc, _ in trained_svms])
avg_nn_acc = np.mean([acc for _, _, acc, _ in trained_nns])
total_weight = avg_svm_acc + avg_nn_acc
svm_nn_weighted_proba = (svm_ensemble_proba * avg_svm_acc + nn_ensemble_proba * avg_nn_acc) / total_weight
svm_nn_weighted_pred = np.argmax(svm_nn_weighted_proba, axis=1)
svm_nn_weighted_labels = label_encoder.inverse_transform(svm_nn_weighted_pred)

# 5.6 Ultimate Ensemble: All SVMs + All NNs + Seed Models
print("[5.6] Ultimate Ensemble: All Models Combined...")
ultimate_proba = np.zeros((len(X_test), n_classes))

# Add SVMs with weight
for name, model, acc, scaler_type in trained_svms:
    X_test_scaled = get_test_data(scaler_type)
    proba = model.predict_proba(X_test_scaled)
    ultimate_proba += proba * acc

# Add NNs with weight
for name, model, acc, scaler_type in trained_nns:
    X_test_scaled = get_test_data(scaler_type)
    proba = model.predict(X_test_scaled, verbose=0)
    ultimate_proba += proba * acc

# Add seed models with average weight
avg_seed_weight = np.mean(seed_accs)
for model in trained_seed_models:
    proba = model.predict(X_test_standard, verbose=0)
    ultimate_proba += proba * avg_seed_weight

total_ultimate_weight = sum([acc for _, _, acc, _ in trained_svms]) + \
                        sum([acc for _, _, acc, _ in trained_nns]) + \
                        (avg_seed_weight * len(trained_seed_models))
ultimate_proba /= total_ultimate_weight
ultimate_pred = np.argmax(ultimate_proba, axis=1)
ultimate_labels = label_encoder.inverse_transform(ultimate_pred)

# 5.7 Stacking Ensemble with Meta-Learner
print("[5.7] Stacking Ensemble with Meta-Learner...")

# Get base model predictions on validation set for meta-learner training
meta_train_features = []

# SVM predictions on validation
for name, model, acc, scaler_type in trained_svms[:3]:  # Use top 3
    if scaler_type == 'standard':
        X_val_scaled = X_val_s
    elif scaler_type == 'robust':
        X_val_scaled = X_val_r
    else:
        X_val_scaled = X_val_m
    proba = model.predict_proba(X_val_scaled)
    meta_train_features.append(proba)

# NN predictions on validation
for name, model, acc, scaler_type in trained_nns[:3]:  # Use top 3
    if scaler_type == 'standard':
        X_val_scaled = X_val_s
    elif scaler_type == 'robust':
        X_val_scaled = X_val_r
    else:
        X_val_scaled = X_val_m
    proba = model.predict(X_val_scaled, verbose=0)
    meta_train_features.append(proba)

meta_train_features = np.hstack(meta_train_features)

# Train meta-learner (Logistic Regression)
meta_learner = LogisticRegression(max_iter=1000, random_state=42)
meta_learner.fit(meta_train_features, y_val)

# Get base model predictions on test set
meta_test_features = []

# SVM predictions on test
for name, model, acc, scaler_type in trained_svms[:3]:
    X_test_scaled = get_test_data(scaler_type)
    proba = model.predict_proba(X_test_scaled)
    meta_test_features.append(proba)

# NN predictions on test
for name, model, acc, scaler_type in trained_nns[:3]:
    X_test_scaled = get_test_data(scaler_type)
    proba = model.predict(X_test_scaled, verbose=0)
    meta_test_features.append(proba)

meta_test_features = np.hstack(meta_test_features)

# Meta-learner predictions
stacking_pred = meta_learner.predict(meta_test_features)
stacking_labels = label_encoder.inverse_transform(stacking_pred)

# 5.8 Confidence-based Ensemble
print("[5.8] Confidence-based Ensemble...")
confidence_predictions = []
for i in range(len(X_test)):
    # Get predictions from all models
    all_probas = []
    
    # SVMs
    for name, model, acc, scaler_type in trained_svms[:5]:
        X_test_scaled = get_test_data(scaler_type)
        proba = model.predict_proba(X_test_scaled[i:i+1])[0]
        all_probas.append((proba, np.max(proba), acc))
    
    # NNs
    for name, model, acc, scaler_type in trained_nns[:5]:
        X_test_scaled = get_test_data(scaler_type)
        proba = model.predict(X_test_scaled[i:i+1], verbose=0)[0]
        all_probas.append((proba, np.max(proba), acc))
    
    # Select prediction from most confident model (weighted by accuracy)
    best_proba = max(all_probas, key=lambda x: x[1] * x[2])[0]
    confidence_predictions.append(np.argmax(best_proba))

confidence_labels = label_encoder.inverse_transform(confidence_predictions)

# 5.9 Voting Ensemble (Hard Voting)
print("[5.9] Hard Voting Ensemble...")
hard_voting_predictions = []
for i in range(len(X_test)):
    votes = []
    
    # SVM votes
    for name, model, acc, scaler_type in trained_svms:
        X_test_scaled = get_test_data(scaler_type)
        pred = model.predict(X_test_scaled[i:i+1])[0]
        votes.append(pred)
    
    # NN votes
    for name, model, acc, scaler_type in trained_nns[:5]:  # Use top 5 to balance
        X_test_scaled = get_test_data(scaler_type)
        proba = model.predict(X_test_scaled[i:i+1], verbose=0)[0]
        pred = np.argmax(proba)
        votes.append(pred)
    
    # Majority vote
    hard_voting_predictions.append(max(set(votes), key=votes.count))

hard_voting_labels = label_encoder.inverse_transform(hard_voting_predictions)

# ============================================================================
# 6. SAVE ALL PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("[6] Saving All Ensemble Predictions")
print("="*80)

predictions_dict = {
    'svm_ensemble_5': svm_ensemble_labels,
    'nn_ensemble_10': nn_ensemble_labels,
    'seed_ensemble_10': seed_ensemble_labels,
    'svm_nn_equal': svm_nn_equal_labels,
    'svm_nn_weighted': svm_nn_weighted_labels,
    'ultimate_ensemble': ultimate_labels,
    'stacking_ensemble': stacking_labels,
    'confidence_ensemble': confidence_labels,
    'hard_voting_ensemble': hard_voting_labels
}

for name, labels in predictions_dict.items():
    submission = pd.DataFrame({
        'sample_id': test_ids,
        'category': labels
    })
    submission.to_csv(f'{name}_predictions.csv', index=False)
    print(f"✓ {name} predictions saved")

# ============================================================================
# 7. COMPREHENSIVE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("[7] COMPREHENSIVE MODEL SUMMARY")
print("="*80)

print("\n📊 Top 5 SVM Models:")
for name, model, acc, scaler_type in trained_svms:
    print(f"  • {name:30s}: {acc:.4f} ({scaler_type})")

print(f"\n🧠 Top 10 NN Models:")
for name, model, acc, scaler_type in trained_nns:
    print(f"  • {name:30s}: {acc:.4f} ({scaler_type})")

print(f"\n🔄 Ensemble Models Generated:")
print(f"  • Top 5 SVMs Weighted Ensemble")
print(f"  • Top 10 NNs Weighted Ensemble")
print(f"  • 10 Seed Models Average Ensemble")
print(f"  • SVM-NN Equal Weight Ensemble")
print(f"  • SVM-NN Performance Weighted Ensemble")
print(f"  • Ultimate Ensemble (All Models)")
print(f"  • Stacking Ensemble with Meta-Learner")
print(f"  • Confidence-based Ensemble")
print(f"  • Hard Voting Ensemble")

print("\n🏆 Recommended Submissions:")
print("  1. ultimate_ensemble_predictions.csv (All models combined)")
print("  2. stacking_ensemble_predictions.csv (Meta-learner approach)")
print("  3. svm_nn_weighted_predictions.csv (Balanced SVM-NN)")

print("\n✅ All predictions generated successfully!")
print("="*80)

# ============================================================================
# 8. SAVE MODELS AND METADATA
# ============================================================================
print("\n[8] Saving Models and Metadata...")

import pickle

# Save scalers
with open('scalers.pkl', 'wb') as f:
    pickle.dump({
        'standard': scaler_standard,
        'robust': scaler_robust,
        'minmax': scaler_minmax,
        'label_encoder': label_encoder
    }, f)
print("✓ Scalers saved")

# Save SVM models
with open('trained_svms.pkl', 'wb') as f:
    pickle.dump(trained_svms, f)
print("✓ SVM models saved")

# Save NN models
for i, (name, model, acc, scaler_type) in enumerate(trained_nns):
    model.save(f'trained_nn_{i}_{name}.keras')
print("✓ NN models saved")

# Save seed models
for i, model in enumerate(trained_seed_models):
    model.save(f'seed_model_{i}.keras')
print("✓ Seed models saved")

# Save meta-learner
with open('meta_learner.pkl', 'wb') as f:
    pickle.dump(meta_learner, f)
print("✓ Meta-learner saved")

print("\n🎉 All tasks completed successfully!")
print("="*80)