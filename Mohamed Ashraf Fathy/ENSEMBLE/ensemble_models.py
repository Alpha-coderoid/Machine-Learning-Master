"""
Ensemble Models for Multi-Classifier Combination
=================================================
This script implements various ensemble techniques to combine predictions
from 10 different classifiers.

Ensemble Methods:
1. Hard Voting (Majority Vote)
2. Weighted Voting (based on individual model accuracy)
3. Soft Voting Approximation (using confidence from agreement)
4. Stacking with Meta-Learner (Logistic Regression, Random Forest)
5. Rank-Based Averaging
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# Load consolidated predictions
print("Loading consolidated predictions...")
df = pd.read_csv('consolidated_predictions.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Load label mapping for display
label_mapping = pd.read_csv('label_mapping.csv')
encoded_to_label = dict(zip(label_mapping['encoded'], label_mapping['label']))

# Separate true labels and predictions
y_true = df['true_label'].values
prediction_cols = ['lightgbm', 'catboost', 'logistic_regression', 'adaboost', 
                   'dcn', 'transformer', 'randomforest', 'knn', 'xgboost', 'mlp']
X_predictions = df[prediction_cols].values

print("="*70)
print("INDIVIDUAL MODEL PERFORMANCE")
print("="*70)

# Calculate individual model accuracies
model_accuracies = {}
model_f1_scores = {}
for i, col in enumerate(prediction_cols):
    acc = accuracy_score(y_true, X_predictions[:, i])
    f1 = f1_score(y_true, X_predictions[:, i], average='weighted')
    model_accuracies[col] = acc
    model_f1_scores[col] = f1
    print(f"{col:25s}: Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")

print("\n" + "="*70)
print("ENSEMBLE METHODS")
print("="*70)

results = {}

# ============================================================================
# 1. HARD VOTING (Majority Vote)
# ============================================================================
print("\n1. HARD VOTING (Majority Vote)")
print("-" * 50)

# Mode (most frequent) for each row
hard_vote_predictions = stats.mode(X_predictions, axis=1, keepdims=True)[0].flatten()
hard_vote_acc = accuracy_score(y_true, hard_vote_predictions)
hard_vote_f1 = f1_score(y_true, hard_vote_predictions, average='weighted')
results['Hard Voting'] = {'accuracy': hard_vote_acc, 'f1': hard_vote_f1}
print(f"Accuracy: {hard_vote_acc:.4f}")
print(f"F1-Score: {hard_vote_f1:.4f}")

# ============================================================================
# 2. WEIGHTED VOTING (based on accuracy)
# ============================================================================
print("\n2. WEIGHTED VOTING (Accuracy-Based)")
print("-" * 50)

weights = np.array([model_accuracies[col] for col in prediction_cols])
weights_normalized = weights / weights.sum()
print(f"Weights: {dict(zip(prediction_cols, np.round(weights_normalized, 4)))}")

# Weighted voting for each sample
n_classes = len(np.unique(y_true))
weighted_vote_predictions = []

for i in range(len(y_true)):
    class_votes = np.zeros(n_classes)
    for j, col in enumerate(prediction_cols):
        pred_class = int(X_predictions[i, j])
        class_votes[pred_class] += weights_normalized[j]
    weighted_vote_predictions.append(np.argmax(class_votes))

weighted_vote_predictions = np.array(weighted_vote_predictions)
weighted_vote_acc = accuracy_score(y_true, weighted_vote_predictions)
weighted_vote_f1 = f1_score(y_true, weighted_vote_predictions, average='weighted')
results['Weighted Voting (Acc)'] = {'accuracy': weighted_vote_acc, 'f1': weighted_vote_f1}
print(f"Accuracy: {weighted_vote_acc:.4f}")
print(f"F1-Score: {weighted_vote_f1:.4f}")

# ============================================================================
# 3. WEIGHTED VOTING (based on F1-Score)
# ============================================================================
print("\n3. WEIGHTED VOTING (F1-Score Based)")
print("-" * 50)

weights_f1 = np.array([model_f1_scores[col] for col in prediction_cols])
weights_f1_normalized = weights_f1 / weights_f1.sum()

weighted_vote_f1_predictions = []
for i in range(len(y_true)):
    class_votes = np.zeros(n_classes)
    for j, col in enumerate(prediction_cols):
        pred_class = int(X_predictions[i, j])
        class_votes[pred_class] += weights_f1_normalized[j]
    weighted_vote_f1_predictions.append(np.argmax(class_votes))

weighted_vote_f1_predictions = np.array(weighted_vote_f1_predictions)
weighted_vote_f1_acc = accuracy_score(y_true, weighted_vote_f1_predictions)
weighted_vote_f1_f1 = f1_score(y_true, weighted_vote_f1_predictions, average='weighted')
results['Weighted Voting (F1)'] = {'accuracy': weighted_vote_f1_acc, 'f1': weighted_vote_f1_f1}
print(f"Accuracy: {weighted_vote_f1_acc:.4f}")
print(f"F1-Score: {weighted_vote_f1_f1:.4f}")

# ============================================================================
# 4. TOP-K VOTING (Only best K models)
# ============================================================================
print("\n4. TOP-K VOTING (Best 5 Models)")
print("-" * 50)

# Select top 5 models by accuracy
sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
top_k = 5
top_k_models = [m[0] for m in sorted_models[:top_k]]
print(f"Top {top_k} models: {top_k_models}")

top_k_indices = [prediction_cols.index(m) for m in top_k_models]
top_k_predictions_arr = X_predictions[:, top_k_indices]
top_k_vote = stats.mode(top_k_predictions_arr, axis=1, keepdims=True)[0].flatten()
top_k_acc = accuracy_score(y_true, top_k_vote)
top_k_f1 = f1_score(y_true, top_k_vote, average='weighted')
results[f'Top-{top_k} Voting'] = {'accuracy': top_k_acc, 'f1': top_k_f1}
print(f"Accuracy: {top_k_acc:.4f}")
print(f"F1-Score: {top_k_f1:.4f}")

# ============================================================================
# 5. STACKING WITH LOGISTIC REGRESSION META-LEARNER
# ============================================================================
print("\n5. STACKING (Logistic Regression Meta-Learner)")
print("-" * 50)

# Use cross-validation for fair evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stacking_lr_predictions = np.zeros(len(y_true))

for train_idx, test_idx in cv.split(X_predictions, y_true):
    X_train, X_test = X_predictions[train_idx], X_predictions[test_idx]
    y_train, y_test = y_true[train_idx], y_true[test_idx]
    
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(X_train, y_train)
    stacking_lr_predictions[test_idx] = meta_model.predict(X_test)

stacking_lr_acc = accuracy_score(y_true, stacking_lr_predictions)
stacking_lr_f1 = f1_score(y_true, stacking_lr_predictions, average='weighted')
results['Stacking (LR)'] = {'accuracy': stacking_lr_acc, 'f1': stacking_lr_f1}
print(f"Accuracy: {stacking_lr_acc:.4f}")
print(f"F1-Score: {stacking_lr_f1:.4f}")

# ============================================================================
# 6. STACKING WITH RANDOM FOREST META-LEARNER
# ============================================================================
print("\n6. STACKING (Random Forest Meta-Learner)")
print("-" * 50)

stacking_rf_predictions = np.zeros(len(y_true))

for train_idx, test_idx in cv.split(X_predictions, y_true):
    X_train, X_test = X_predictions[train_idx], X_predictions[test_idx]
    y_train, y_test = y_true[train_idx], y_true[test_idx]
    
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    meta_model.fit(X_train, y_train)
    stacking_rf_predictions[test_idx] = meta_model.predict(X_test)

stacking_rf_acc = accuracy_score(y_true, stacking_rf_predictions)
stacking_rf_f1 = f1_score(y_true, stacking_rf_predictions, average='weighted')
results['Stacking (RF)'] = {'accuracy': stacking_rf_acc, 'f1': stacking_rf_f1}
print(f"Accuracy: {stacking_rf_acc:.4f}")
print(f"F1-Score: {stacking_rf_f1:.4f}")

# ============================================================================
# 7. STACKING WITH GRADIENT BOOSTING META-LEARNER
# ============================================================================
print("\n7. STACKING (Gradient Boosting Meta-Learner)")
print("-" * 50)

stacking_gb_predictions = np.zeros(len(y_true))

for train_idx, test_idx in cv.split(X_predictions, y_true):
    X_train, X_test = X_predictions[train_idx], X_predictions[test_idx]
    y_train, y_test = y_true[train_idx], y_true[test_idx]
    
    meta_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    meta_model.fit(X_train, y_train)
    stacking_gb_predictions[test_idx] = meta_model.predict(X_test)

stacking_gb_acc = accuracy_score(y_true, stacking_gb_predictions)
stacking_gb_f1 = f1_score(y_true, stacking_gb_predictions, average='weighted')
results['Stacking (GB)'] = {'accuracy': stacking_gb_acc, 'f1': stacking_gb_f1}
print(f"Accuracy: {stacking_gb_acc:.4f}")
print(f"F1-Score: {stacking_gb_f1:.4f}")

# ============================================================================
# 8. STACKING WITH MLP META-LEARNER
# ============================================================================
print("\n8. STACKING (MLP Meta-Learner)")
print("-" * 50)

stacking_mlp_predictions = np.zeros(len(y_true))

for train_idx, test_idx in cv.split(X_predictions, y_true):
    X_train, X_test = X_predictions[train_idx], X_predictions[test_idx]
    y_train, y_test = y_true[train_idx], y_true[test_idx]
    
    meta_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    meta_model.fit(X_train, y_train)
    stacking_mlp_predictions[test_idx] = meta_model.predict(X_test)

stacking_mlp_acc = accuracy_score(y_true, stacking_mlp_predictions)
stacking_mlp_f1 = f1_score(y_true, stacking_mlp_predictions, average='weighted')
results['Stacking (MLP)'] = {'accuracy': stacking_mlp_acc, 'f1': stacking_mlp_f1}
print(f"Accuracy: {stacking_mlp_acc:.4f}")
print(f"F1-Score: {stacking_mlp_f1:.4f}")

# ============================================================================
# 9. ORACLE (Upper Bound - at least one model is correct)
# ============================================================================
print("\n9. ORACLE (Theoretical Upper Bound)")
print("-" * 50)

oracle_correct = np.any(X_predictions == y_true.reshape(-1, 1), axis=1)
oracle_acc = oracle_correct.mean()
print(f"Oracle Accuracy (at least one model correct): {oracle_acc:.4f}")
results['Oracle (Upper Bound)'] = {'accuracy': oracle_acc, 'f1': None}

# ============================================================================
# SUMMARY RESULTS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY OF ALL RESULTS")
print("="*70)

# Create summary dataframe
summary_data = []

# Add individual models
for col in prediction_cols:
    summary_data.append({
        'Method': f"Individual: {col}",
        'Accuracy': model_accuracies[col],
        'F1-Score': model_f1_scores[col]
    })

# Add ensemble methods
for method, scores in results.items():
    summary_data.append({
        'Method': f"Ensemble: {method}",
        'Accuracy': scores['accuracy'],
        'F1-Score': scores['f1'] if scores['f1'] is not None else '-'
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Accuracy', ascending=False)
print("\n" + summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv('ensemble_results_summary.csv', index=False)
print("\nResults saved to: ensemble_results_summary.csv")

# ============================================================================
# BEST ENSEMBLE - Detailed Classification Report
# ============================================================================
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT - BEST ENSEMBLE")
print("="*70)

# Find best ensemble method
best_ensemble = max(
    [(k, v) for k, v in results.items() if v['f1'] is not None],
    key=lambda x: x[1]['accuracy']
)
print(f"\nBest Ensemble Method: {best_ensemble[0]}")
print(f"Accuracy: {best_ensemble[1]['accuracy']:.4f}")
print(f"F1-Score: {best_ensemble[1]['f1']:.4f}")

# Get predictions for best ensemble
if best_ensemble[0] == 'Hard Voting':
    best_predictions = hard_vote_predictions
elif best_ensemble[0] == 'Weighted Voting (Acc)':
    best_predictions = weighted_vote_predictions
elif best_ensemble[0] == 'Weighted Voting (F1)':
    best_predictions = weighted_vote_f1_predictions
elif best_ensemble[0] == f'Top-{top_k} Voting':
    best_predictions = top_k_vote
elif best_ensemble[0] == 'Stacking (LR)':
    best_predictions = stacking_lr_predictions
elif best_ensemble[0] == 'Stacking (RF)':
    best_predictions = stacking_rf_predictions
elif best_ensemble[0] == 'Stacking (GB)':
    best_predictions = stacking_gb_predictions
elif best_ensemble[0] == 'Stacking (MLP)':
    best_predictions = stacking_mlp_predictions

# Classification report
target_names = [encoded_to_label[i] for i in range(n_classes)]
print("\nClassification Report:")
print(classification_report(y_true, best_predictions, target_names=target_names))

# Save final predictions
final_df = pd.DataFrame({
    'true_label': y_true,
    'true_label_name': [encoded_to_label[l] for l in y_true],
    'ensemble_prediction': best_predictions.astype(int),
    'ensemble_prediction_name': [encoded_to_label[int(l)] for l in best_predictions]
})
final_df.to_csv('best_ensemble_predictions.csv', index=False)
print("Best ensemble predictions saved to: best_ensemble_predictions.csv")

print("\n" + "="*70)
print("ENSEMBLE MODELING COMPLETE!")
print("="*70)
