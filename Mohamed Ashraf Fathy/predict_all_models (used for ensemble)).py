"""
Generate predictions from all 3 models on test set
===================================================
Adds prediction columns for KNN, XGBoost, and MLP to test.csv
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ============================================================================
# MLP Model Definition (must match training)
# ============================================================================
class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[256, 128, 64], dropout=0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("GENERATING PREDICTIONS FROM ALL MODELS")
    print("=" * 60)
    
    # Load test data
    print("\n[1/5] Loading test data...")
    test = pd.read_csv('processed_data_v2/test.csv')
    X_test = test.drop(columns=['label', 'label_encoded'])
    
    print(f"  Test samples: {len(test):,}")
    print(f"  Features: {X_test.shape[1]}")
    
    # Load models
    print("\n[2/5] Loading models...")
    
    # KNN
    knn_model = joblib.load('models_v2/knn_model.joblib')
    knn_scaler = joblib.load('models_v2/knn_scaler.joblib')
    print("  Loaded: KNN model")
    
    # XGBoost
    xgb_model = joblib.load('models_v2/xgboost_model.joblib')
    print("  Loaded: XGBoost model")
    
    # MLP
    mlp_info = joblib.load('models_v2/mlp_model_info.joblib')
    mlp_scaler = joblib.load('models_v2/mlp_scaler.joblib')
    mlp_model = MLPClassifier(
        mlp_info['input_size'], 
        mlp_info['num_classes'],
        mlp_info['hidden_sizes'],
        mlp_info['dropout']
    )
    mlp_model.load_state_dict(torch.load('models_v2/mlp_weights.pth', weights_only=True))
    mlp_model.eval()
    print("  Loaded: MLP model")
    
    # Generate predictions
    print("\n[3/5] Generating KNN predictions...")
    X_knn = knn_scaler.transform(X_test)
    knn_preds = knn_model.predict(X_knn)
    print(f"  Done: {len(knn_preds):,} predictions")
    
    print("\n[4/5] Generating XGBoost predictions...")
    xgb_preds = xgb_model.predict(X_test)
    print(f"  Done: {len(xgb_preds):,} predictions")
    
    print("\n[5/5] Generating MLP predictions...")
    X_mlp = mlp_scaler.transform(X_test)
    X_mlp_tensor = torch.FloatTensor(X_mlp)
    
    with torch.no_grad():
        outputs = mlp_model(X_mlp_tensor)
        _, mlp_preds = outputs.max(1)
    mlp_preds = mlp_preds.numpy()
    print(f"  Done: {len(mlp_preds):,} predictions")
    
    # Add prediction columns to test data
    print("\n[6/6] Saving predictions...")
    test['pred_knn'] = knn_preds
    test['pred_xgboost'] = xgb_preds
    test['pred_mlp'] = mlp_preds
    
    # Save to new file
    output_file = 'processed_data_v2/test_with_predictions.csv'
    test.to_csv(output_file, index=False)
    
    print(f"\n  Saved: {output_file}")
    print(f"  Columns: {list(test.columns[-5:])}")
    
    # Show sample
    print("\n  Sample predictions (first 5 rows):")
    print(test[['label', 'label_encoded', 'pred_knn', 'pred_xgboost', 'pred_mlp']].head())
    
    # Accuracy summary
    from sklearn.metrics import accuracy_score
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    print(f"  KNN:     {accuracy_score(test['label_encoded'], test['pred_knn']):.4f}")
    print(f"  XGBoost: {accuracy_score(test['label_encoded'], test['pred_xgboost']):.4f}")
    print(f"  MLP:     {accuracy_score(test['label_encoded'], test['pred_mlp']):.4f}")
    
    print("\nDONE!")

if __name__ == "__main__":
    main()
