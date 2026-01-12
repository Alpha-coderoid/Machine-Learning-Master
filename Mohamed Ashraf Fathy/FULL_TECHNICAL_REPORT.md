# DDoS Attack Detection - Complete Technical Report

## 1. Introduction

This report documents the development of my machine learning-based DDoS (Distributed Denial of Service) attack detection system. I implemented and compared three classification models—K-Nearest Neighbors (KNN), XGBoost, and Multi-Layer Perceptron (MLP)—on the CICDDoS2019 benchmark dataset.

---

## 2. Dataset Overview

### 2.1 Source
**CICDDoS2019** - Published by the Canadian Institute for Cybersecurity (CIC), University of New Brunswick. Features extracted using CICFlowMeter tool.

### 2.2 Original Statistics
| Metric | Value |
|--------|-------|
| Total Samples | 431,371 |
| Original Features | 88 |
| Attack Types | 13 (12 DDoS + 1 Benign) |
| Format | Parquet files |

### 2.3 Attack Classes (Original)
| Class | Samples | Percentage |
|-------|---------|------------|
| NTP | 121,368 | 28.1% |
| TFTP | 98,917 | 22.9% |
| Benign | 97,831 | 22.7% |
| Syn | 49,373 | 11.4% |
| UDP | 28,510 | 6.6% |
| MSSQL | 14,735 | 3.4% |
| UDPLag | 8,927 | 2.1% |
| DNS | 3,669 | 0.9% |
| LDAP | 3,346 | 0.8% |
| SNMP | 2,717 | 0.6% |
| NetBIOS | 1,242 | 0.3% |
| Portmap | 685 | 0.2% |
| WebDDoS | 51 | 0.01% |

---

## 3. Data Preprocessing

### 3.1 Data Quality Issues Identified
1. **Infinity Values**: 977-43,345 inf values per file in `Flow Bytes/s` and `Flow Packets/s`
2. **Missing Values**: 4-29 NaN values per file
3. **Inconsistent Labels**: Training used "MSSQL", testing used "DrDoS_MSSQL"
4. **Severe Class Imbalance**: WebDDoS had only 51 samples (0.01%)

### 3.2 Preprocessing Steps Applied
| Step | Action | Reason |
|------|--------|--------|
| 1 | Label normalization | Unified "DrDoS_X" → "X" format |
| 2 | Drop non-predictive columns (Flow ID, IPs, Timestamps) | Prevent overfitting to specific hosts |
| 3 | Replace infinity with NaN | Prepare for imputation |
| 4 | Impute missing with median | Robust to outliers |
| 5 | Drop WebDDoS class | Too few samples (51) for learning |
| 6 | Feature selection | Remove useless/redundant features |
| 7 | StandardScaler | Required for KNN and MLP |

### 3.3 Final Dataset (v2)
| Metric | Value |
|--------|-------|
| Training Samples | 345,056 |
| Test Samples | 86,264 |
| Features | 52 (reduced from 77) |
| Classes | 12 (WebDDoS removed) |
| Train/Test Split | 80%/20% stratified |
| Random Seed | 42 |

---

## 4. Feature Selection

### 4.1 Analysis Methods Used
1. **Variance Analysis**: Identify near-zero variance features
2. **Correlation Analysis**: Find redundant features (r > 0.95)
3. **Random Forest Importance**: Tree-based feature ranking
4. **Mutual Information**: Statistical dependency with target

### 4.2 Features Removed (25 total)

**Zero Variance (15 features)**:
- URG Flags (Fwd/Bwd) - Always 0
- PSH Flags (Fwd/Bwd) - Near-constant
- FIN, PSH, ECE Flag Counts - Near-constant
- All Bulk Transfer features (6) - All zeros in DDoS traffic
- CWE Flag Count - Rarely used

**Redundant - Perfectly Correlated (10 features)**:
| Kept | Removed (r=1.00) |
|------|-----------------|
| Total Fwd Packets | Subflow Fwd Packets |
| Total Backward Packets | Subflow Bwd Packets |
| Fwd Packets Length Total | Subflow Fwd Bytes |
| Bwd Packets Length Total | Subflow Bwd Bytes |
| Fwd Packet Length Mean | Avg Fwd Segment Size |
| Bwd Packet Length Mean | Avg Bwd Segment Size |
| Flow Duration | Fwd IAT Total |
| Flow IAT Min | Fwd IAT Min |
| Flow IAT Max | Fwd IAT Max |
| Fwd PSH Flags | RST Flag Count |

### 4.3 Top 10 Most Important Features
| Rank | Feature | RF Importance | Mutual Info |
|------|---------|---------------|-------------|
| 1 | Fwd Packet Length Min | 0.093 | 1.57 |
| 2 | Packet Length Min | 0.086 | 1.58 |
| 3 | Avg Fwd Segment Size | 0.081 | 1.57 |
| 4 | Fwd Packet Length Mean | 0.056 | 1.57 |
| 5 | Fwd Packets Length Total | 0.055 | 1.54 |
| 6 | Fwd Packet Length Max | 0.042 | 1.56 |
| 7 | Packet Length Mean | 0.039 | 1.57 |
| 8 | Avg Packet Size | 0.038 | 1.60 |
| 9 | Flow Bytes/s | 0.036 | 1.44 |
| 10 | Subflow Fwd Bytes | 0.036 | 1.54 |

---

## 5. Model Selection Rationale

### Why These 3 Models?

| Model | Why Chosen |
|-------|------------|
| **KNN** | Non-parametric, no distribution assumptions, good baseline for comparison |
| **XGBoost** | State-of-the-art for tabular data, handles imbalance well, provides feature importance |
| **MLP** | Deep learning representative, can learn complex non-linear patterns, commonly used benchmark |

---

## 6. Model 1: K-Nearest Neighbors (KNN)

### 6.1 How KNN Works
KNN classifies a sample based on the majority vote of its k nearest neighbors in feature space. For a new sample:
1. Calculate distance to all training samples
2. Find the k closest neighbors
3. Assign the most common class among neighbors

### 6.2 Grid Search Configuration
| Parameter | Values Tested |
|-----------|--------------|
| `n_neighbors` | 3, 5, 7, 9, 11 |
| `weights` | uniform, distance |
| `metric` | euclidean, manhattan |
| **Total Combinations** | 5 × 2 × 2 = **20** |

### 6.3 Best Hyperparameters Found
| Parameter | Best Value | Explanation |
|-----------|------------|-------------|
| `n_neighbors` | **7** | Balances noise reduction vs decision boundary smoothness |
| `weights` | **distance** | Closer neighbors have more influence on classification |
| `metric` | **manhattan** | Works better with varying feature scales, less sensitive to outliers |

### 6.4 Cross-Validation Results (5-Fold)
| Fold | Accuracy |
|------|----------|
| 1 | 96.74% |
| 2 | 96.66% |
| 3 | 96.66% |
| 4 | 96.91% |
| 5 | 96.83% |
| **Mean ± Std** | **96.76% ± 0.19%** |

### 6.5 Final Test Results
| Metric | Value |
|--------|-------|
| Test Accuracy | **97.15%** |
| Grid Search Time | 224 seconds |

---

## 7. Model 2: XGBoost

### 7.1 How XGBoost Works
XGBoost (Extreme Gradient Boosting) is an ensemble method that:
1. Builds decision trees sequentially
2. Each new tree corrects errors of previous trees
3. Uses gradient descent to minimize loss function
4. Includes regularization (L1/L2) to prevent overfitting

Key advantages: handles missing values, built-in feature importance, parallelizable.

### 7.2 Grid Search Configuration
| Parameter | Values Tested | Description |
|-----------|--------------|-------------|
| `n_estimators` | 50, 100, 150 | Number of trees |
| `max_depth` | 6, 8, 10 | Maximum tree depth |
| `learning_rate` | 0.05, 0.1, 0.2 | Step size shrinkage |
| `min_child_weight` | 1, 3 | Minimum samples per leaf |
| **Total Combinations** | 3 × 3 × 3 × 2 = **54** |

### 7.3 Best Hyperparameters Found
| Parameter | Best Value | Explanation |
|-----------|------------|-------------|
| `n_estimators` | **100** | Sufficient trees for convergence without overfitting |
| `max_depth` | **6** | Moderate depth captures patterns without memorizing |
| `learning_rate` | **0.1** | Standard rate, good balance of speed and accuracy |
| `min_child_weight` | **3** | Regularization requiring 3+ samples per leaf |

### 7.4 Fixed Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `objective` | multi:softmax | Multi-class classification |
| `tree_method` | hist | Fast histogram-based algorithm |
| `n_jobs` | -1 | Use all CPU cores |

### 7.5 Cross-Validation Results (5-Fold)
| Fold | Accuracy |
|------|----------|
| 1 | 97.73% |
| 2 | 97.61% |
| 3 | 97.65% |
| 4 | 97.76% |
| 5 | 97.65% |
| **Mean ± Std** | **97.68% ± 0.12%** |

### 7.6 Final Test Results
| Metric | Value |
|--------|-------|
| Test Accuracy | **97.89%** |
| Grid Search Time | 454 seconds |

### 7.7 XGBoost Feature Importance (Top 10)
| Feature | Importance |
|---------|------------|
| ACK Flag Count | 0.272 |
| Packet Length Min | 0.161 |
| URG Flag Count | 0.091 |
| Bwd Packets/s | 0.065 |
| Fwd Act Data Packets | 0.059 |
| Fwd Packet Length Min | 0.055 |
| Fwd Packet Length Std | 0.054 |
| Idle Max | 0.048 |
| Idle Std | 0.036 |
| Fwd Packets Length Total | 0.031 |

---

## 8. Model 3: Multi-Layer Perceptron (MLP)

### 8.1 How MLP Works
An MLP is a feedforward neural network with:
1. **Input Layer**: Receives feature vector
2. **Hidden Layers**: Learn non-linear transformations
3. **Output Layer**: Produces class probabilities (softmax)

Each neuron computes: `output = activation(weights · input + bias)`

### 8.2 Architecture
```
Layer 1: Input (52 features)
    ↓
Layer 2: Dense(256) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Layer 3: Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Layer 4: Dense(64) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Layer 5: Dense(12) → Softmax (output)
```

### 8.3 Layer Explanations
| Component | Purpose |
|-----------|---------|
| **Dense (Linear)** | Learnable weight transformation |
| **BatchNorm** | Normalizes layer inputs, stabilizes training |
| **ReLU** | Non-linear activation: max(0, x) |
| **Dropout (0.3)** | Randomly zeros 30% of neurons to prevent overfitting |
| **Softmax** | Converts logits to class probabilities (sum to 1) |

### 8.4 Training Configuration
| Parameter | Value | Explanation |
|-----------|-------|-------------|
| Optimizer | Adam | Adaptive learning rate, momentum |
| Learning Rate | 0.001 | Standard starting point |
| Loss Function | CrossEntropyLoss | Standard for multi-class |
| Batch Size | 512 | Balance speed and gradient quality |
| Epochs | 50 (max) | Maximum training iterations |
| Early Stopping | Patience=5 | Stop if no improvement for 5 epochs |
| LR Scheduler | ReduceLROnPlateau | Halve LR when loss stalls |

### 8.5 Cross-Validation Results (5-Fold)
| Fold | Best Val Accuracy | Early Stop Epoch |
|------|-------------------|------------------|
| 1 | 97.23% | Epoch 22 |
| 2 | 97.25% | Epoch 39 |
| 3 | 97.08% | Epoch 21 |
| 4 | 96.99% | Epoch 17 |
| 5 | 97.04% | Epoch 21 |
| **Mean ± Std** | **97.12% ± 0.20%** | |

### 8.6 Final Test Results
| Metric | Value |
|--------|-------|
| Test Accuracy | **97.27%** |
| Training Device | Apple MPS GPU |

---

## 9. Model Comparison

### 9.1 Overall Results
| Model | CV Accuracy | Test Accuracy | Training Time |
|-------|-------------|---------------|---------------|
| **XGBoost** | 97.68% ± 0.12% | **97.89%** | ~10s |
| **MLP** | 97.12% ± 0.20% | 97.27% | ~5 min |
| **KNN** | 96.76% ± 0.19% | 97.15% | ~0.1s (inference slow) |

### 9.2 Per-Class F1 Scores
| Class | KNN | XGBoost | MLP |
|-------|-----|---------|-----|
| Benign | 1.00 | 1.00 | 1.00 |
| NTP | 1.00 | 1.00 | 1.00 |
| TFTP | 1.00 | 1.00 | 1.00 |
| Syn | 0.98 | 0.99 | 0.99 |
| UDP | 0.94 | 0.95 | 0.94 |
| MSSQL | 0.91 | 0.92 | 0.90 |
| UDPLag | 0.74 | 0.83 | 0.83 |
| SNMP | 0.67 | 0.68 | 0.59 |
| NetBIOS | 0.65 | 0.68 | 0.64 |
| LDAP | 0.52 | 0.65 | 0.47 |
| DNS | 0.40 | 0.55 | 0.35 |
| Portmap | 0.38 | 0.48 | 0.10 |

### 9.3 Key Findings
1. **XGBoost performs best** overall and most consistently across folds
2. **All models excel** on majority classes (Benign, NTP, TFTP, Syn)
3. **Minority classes struggle** (DNS, Portmap) - need more training data
4. **MLP and XGBoost** handle UDPLag better than KNN

---

## 10. Saved Artifacts

### 10.1 Models
| File | Description |
|------|-------------|
| `models_v2/knn_model.joblib` | Trained KNN classifier |
| `models_v2/xgboost_model.joblib` | Trained XGBoost classifier |
| `models_v2/mlp_weights.pth` | PyTorch MLP weights |

### 10.2 Preprocessors
| File | Description |
|------|-------------|
| `models_v2/knn_scaler.joblib` | StandardScaler for KNN |
| `models_v2/mlp_scaler.joblib` | StandardScaler for MLP |

### 10.3 Metadata
| File | Description |
|------|-------------|
| `models_v2/knn_best_params.joblib` | KNN optimal hyperparameters |
| `models_v2/xgboost_best_params.joblib` | XGBoost optimal hyperparameters |
| `models_v2/mlp_model_info.joblib` | MLP architecture details |
| `models_v2/xgboost_feature_importance.csv` | Feature rankings |

### 10.4 Data
| File | Description |
|------|-------------|
| `processed_data_v2/train.csv` | Preprocessed training data |
| `processed_data_v2/test.csv` | Preprocessed test data |
| `processed_data_v2/test_with_predictions.csv` | Test with all 3 model predictions |
| `processed_data_v2/features.csv` | List of 52 features used |
| `processed_data_v2/label_mapping.csv` | Label encoding reference |

---

## 11. Reproducibility

All experiments use:
- **Random Seed**: 42
- **Train/Test Split**: 80/20 stratified
- **Cross-Validation**: 5-Fold Stratified
- **Python Version**: 3.13
- **Key Libraries**: scikit-learn, xgboost, torch

---

## 12. Conclusion

We successfully developed a DDoS detection system achieving up to **97.89% accuracy** using XGBoost. The feature selection process reduced dimensionality by 32% (77→52 features) without sacrificing performance. All three models demonstrated strong generalization with consistent cross-validation results. XGBoost is recommended for deployment due to its superior accuracy, fast inference, and interpretable feature importance.
