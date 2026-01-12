"""
DDoS Detection - XGBoost with K-Fold CV and Grid Search
========================================================
Trains XGBoost on the feature-selected dataset (v2) with:
- 5-Fold Stratified Cross-Validation
- Grid Search for hyperparameter tuning
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
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
SAMPLE_SIZE = 80000  # Sample for grid search

# ============================================================================
# Main
# ============================================================================
def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("XGBoost - K-Fold CV + Grid Search")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    train = pd.read_csv(f'{DATA_DIR}/train.csv')
    test = pd.read_csv(f'{DATA_DIR}/test.csv')
    
    X_train = train.drop(columns=['label', 'label_encoded'])
    y_train = train['label_encoded']
    X_test = test.drop(columns=['label', 'label_encoded'])
    y_test = test['label_encoded']
    
    num_classes = y_train.nunique()
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Classes: {num_classes}")
    
    # Sample for grid search
    print(f"\n[2/5] Sampling {SAMPLE_SIZE:,} for Grid Search...")
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(len(X_train), min(SAMPLE_SIZE, len(X_train)), replace=False)
    X_sample = X_train.iloc[idx]
    y_sample = y_train.iloc[idx]
    
    # Grid Search
    print("\n[3/5] Grid Search with 5-Fold CV...")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_weight': [1, 3]
    }
    
    xgb = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        xgb, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
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
    print("\n[4/5] Training final model on full training data...")
    best_xgb = XGBClassifier(
        **grid_search.best_params_,
        objective='multi:softmax',
        num_class=num_classes,
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    
    start = time.time()
    best_xgb.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.1f}s")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    y_pred = best_xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    print("\n[5/5] Top 10 Feature Importances:")
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature'][:35]:35} : {row['importance']:.4f}")
    
    # Save model
    joblib.dump(best_xgb, f'{OUTPUT_DIR}/xgboost_model.joblib')
    joblib.dump(grid_search.best_params_, f'{OUTPUT_DIR}/xgboost_best_params.joblib')
    importance.to_csv(f'{OUTPUT_DIR}/xgboost_feature_importance.csv', index=False)
    
    print(f"\n  Saved: {OUTPUT_DIR}/xgboost_model.joblib")
    print(f"  Saved: {OUTPUT_DIR}/xgboost_best_params.joblib")
    print(f"  Saved: {OUTPUT_DIR}/xgboost_feature_importance.csv")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return {
        'model': 'XGBoost',
        'best_params': grid_search.best_params_,
        'cv_accuracy_mean': grid_search.best_score_,
        'test_accuracy': accuracy
    }

if __name__ == "__main__":
    results = main()
