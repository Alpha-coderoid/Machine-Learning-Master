

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = "processed_data_v2"
OUTPUT_DIR = "models_v2"
RANDOM_STATE = 42
N_FOLDS = 5
SAMPLE_SIZE = 80000  # Sample for grid search (faster)

# ============================================================================
# Main
# ============================================================================
def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("KNN - K-Fold CV + Grid Search")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    train = pd.read_csv(f'{DATA_DIR}/train.csv')
    test = pd.read_csv(f'{DATA_DIR}/test.csv')
    
    X_train = train.drop(columns=['label', 'label_encoded'])
    y_train = train['label_encoded']
    X_test = test.drop(columns=['label', 'label_encoded'])
    y_test = test['label_encoded']
    
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Scale features
    print("\n[2/5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sample for grid search
    print(f"\n[3/5] Sampling {SAMPLE_SIZE:,} for Grid Search...")
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(len(X_train_scaled), min(SAMPLE_SIZE, len(X_train_scaled)), replace=False)
    X_sample = X_train_scaled[idx]
    y_sample = y_train.iloc[idx]
    
    # Grid Search
    print("\n[4/5] Grid Search with 5-Fold CV...")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier(n_jobs=-1)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    start = time.time()
    grid_search.fit(X_sample, y_sample)
    grid_time = time.time() - start
    
    print(f"\n  Grid Search completed in {grid_time:.1f}s")
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")
    
    # Cross-validation results
    print("\n  Cross-validation scores per fold:")
    cv_results = grid_search.cv_results_
    best_idx = grid_search.best_index_
    for i in range(N_FOLDS):
        score = cv_results[f'split{i}_test_score'][best_idx]
        print(f"    Fold {i+1}: {score:.4f}")
    print(f"    Mean: {grid_search.best_score_:.4f} (+/- {cv_results['std_test_score'][best_idx]*2:.4f})")
    
    # Train final model with best params on full training data
    print("\n[5/5] Training final model on full training data...")
    best_knn = KNeighborsClassifier(**grid_search.best_params_, n_jobs=-1)
    
    start = time.time()
    best_knn.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.1f}s")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    y_pred = best_knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(best_knn, f'{OUTPUT_DIR}/knn_model.joblib')
    joblib.dump(scaler, f'{OUTPUT_DIR}/knn_scaler.joblib')
    joblib.dump(grid_search.best_params_, f'{OUTPUT_DIR}/knn_best_params.joblib')
    
    # Save results
    results = {
        'model': 'KNN',
        'best_params': grid_search.best_params_,
        'cv_accuracy_mean': grid_search.best_score_,
        'cv_accuracy_std': cv_results['std_test_score'][best_idx],
        'test_accuracy': accuracy,
        'grid_search_time': grid_time,
        'train_time': train_time
    }
    
    print(f"\n  Saved: {OUTPUT_DIR}/knn_model.joblib")
    print(f"  Saved: {OUTPUT_DIR}/knn_scaler.joblib")
    print(f"  Saved: {OUTPUT_DIR}/knn_best_params.joblib")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    results = main()
