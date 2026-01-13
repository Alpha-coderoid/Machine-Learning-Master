# XGBoost DDoS Detection - Deployment Package

## Files Included
- `xgboost_predict.py` - Prediction module with `predict_attack()` function
- `xgboost_model.joblib` - Trained XGBoost model
- `label_encoder.joblib` - Label encoder (converts predictions to class names)
- `feature_names.joblib` - List of 52 required feature names

## Requirements
```
pip install xgboost scikit-learn joblib numpy
```

## Usage

```python
from xgboost_predict import predict_attack, get_feature_names, get_class_names

# Get required feature names
features = get_feature_names()  # Returns list of 52 feature names

# Create feature dict (from network flow data)
flow_data = {
    'Fwd Packet Length Min': 40,
    'Packet Length Min': 40,
    'Flow Duration': 1000,
    # ... all 52 features
}

# Predict
result = predict_attack(flow_data)

print(result['predicted_class'])  # e.g., 'Benign', 'Syn', 'NTP'
print(result['predicted_label'])  # e.g., 0, 8, 4
print(result['probabilities'])    # Dict of class probabilities
```

## Classes (12 total)
Benign, DNS, LDAP, MSSQL, NTP, NetBIOS, Portmap, SNMP, Syn, TFTP, UDP, UDPLag

## Model Performance
- **Accuracy**: 97.89%
- **Best on all 12 classes**
