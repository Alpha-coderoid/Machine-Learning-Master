# Evaluation Metrics Report

## Overall Metrics

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|-------|----------|---------------------|-------------------|---------------|
| KNN | 0.9715 | 0.9708 | 0.9715 | 0.9708 |
| XGBoost | 0.9789 | 0.9787 | 0.9789 | 0.9783 |
| MLP | 0.9727 | 0.9731 | 0.9727 | 0.9713 |

## Per-Class F1-Score Comparison

| Class | Support | KNN | XGBoost | MLP | Best |
|-------|---------|-----|---------|-----|------|
| Benign | 19566 | 0.9979 | 0.9984 | 0.9956 | XGBoost |
| DNS | 734 | 0.4047 | 0.5526 | 0.3492 | XGBoost |
| LDAP | 669 | 0.5184 | 0.6474 | 0.4722 | XGBoost |
| MSSQL | 2947 | 0.9054 | 0.9213 | 0.8951 | XGBoost |
| NTP | 24274 | 0.9978 | 0.9981 | 0.9971 | XGBoost |
| NetBIOS | 248 | 0.6492 | 0.6819 | 0.6434 | XGBoost |
| Portmap | 137 | 0.3789 | 0.4775 | 0.0972 | XGBoost |
| SNMP | 543 | 0.6699 | 0.6788 | 0.5880 | XGBoost |
| Syn | 9875 | 0.9839 | 0.9946 | 0.9945 | XGBoost |
| TFTP | 19783 | 0.9971 | 0.9980 | 0.9970 | XGBoost |
| UDP | 5702 | 0.9376 | 0.9502 | 0.9424 | XGBoost |
| UDPLag | 1786 | 0.7396 | 0.8338 | 0.8286 | XGBoost |
