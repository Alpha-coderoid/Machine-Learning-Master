"""
XGBoost DDoS Detection - Prediction Module
===========================================
Simple prediction interface for GUI integration.

Usage:
    from xgboost_predict import predict_attack
    result = predict_attack(features_dict)
"""

import joblib
import numpy as np
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and label encoder at module load
_model = joblib.load(os.path.join(SCRIPT_DIR, 'xgboost_model.joblib'))
_label_encoder = joblib.load(os.path.join(SCRIPT_DIR, 'label_encoder.joblib'))
_feature_names = joblib.load(os.path.join(SCRIPT_DIR, 'feature_names.joblib'))

# Class labels
CLASS_NAMES = list(_label_encoder.classes_)


def predict_attack(features):
    """
    Predict DDoS attack type from network flow features.
    
    Args:
        features: Can be one of:
            - dict: {feature_name: value, ...}
            - list: [value1, value2, ...] in order of feature_names
            - numpy array: shape (n_features,) or (n_samples, n_features)
    
    Returns:
        dict: {
            'predicted_class': str,      # e.g., 'Benign', 'Syn', 'NTP'
            'predicted_label': int,      # e.g., 0, 8, 4
            'probabilities': dict        # {class_name: probability, ...}
        }
    """
    # Convert input to numpy array
    if isinstance(features, dict):
        # Ensure features are in correct order
        X = np.array([[features.get(f, 0) for f in _feature_names]])
    elif isinstance(features, list):
        X = np.array([features])
    else:
        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)
    
    # Get prediction
    pred_label = _model.predict(X)[0]
    pred_class = _label_encoder.inverse_transform([pred_label])[0]
    
    # Get probabilities if available
    try:
        probs = _model.predict_proba(X)[0]
        prob_dict = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
    except:
        prob_dict = {pred_class: 1.0}
    
    return {
        'predicted_class': pred_class,
        'predicted_label': int(pred_label),
        'probabilities': prob_dict
    }


def predict_batch(features_list):
    """
    Predict for multiple samples at once.
    
    Args:
        features_list: List of feature dicts or 2D numpy array
    
    Returns:
        list of prediction dicts
    """
    results = []
    for features in features_list:
        results.append(predict_attack(features))
    return results


def predict_from_csv(csv_path):
    """
    Predict from a CSV file containing one sample.
    
    Args:
        csv_path: Path to CSV file with feature columns
    
    Returns:
        dict: Prediction result
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Take first row only
    row = df.iloc[0]
    
    # Extract features (ignore label columns if present)
    features = {}
    for f in _feature_names:
        if f in row:
            features[f] = row[f]
        else:
            features[f] = 0  # Default if missing
    
    return predict_attack(features)


def get_feature_names():
    """Return list of required feature names in order."""
    return list(_feature_names)


def get_class_names():
    """Return list of possible class names."""
    return CLASS_NAMES


# Quick test when run directly
if __name__ == "__main__":
    print("XGBoost DDoS Prediction Module")
    print("=" * 40)
    print(f"Model loaded successfully")
    print(f"Features required: {len(_feature_names)}")
    print(f"Classes: {CLASS_NAMES}")
    
    # Test with dummy data
    dummy = {f: 0.0 for f in _feature_names}
    result = predict_attack(dummy)
    print(f"\nTest prediction: {result['predicted_class']}")
