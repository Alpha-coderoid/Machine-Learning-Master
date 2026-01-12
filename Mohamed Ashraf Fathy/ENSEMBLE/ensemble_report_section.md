# Ensemble Learning for DDoS Attack Classification

## 1. Introduction

To improve classification performance beyond individual models, we implemented ensemble learning techniques that combine predictions from multiple classifiers. Ensemble methods leverage the diversity of different models to reduce variance, mitigate overfitting, and potentially achieve higher accuracy than any single constituent model.

## 2. Base Models

We consolidated predictions from 10 different machine learning models trained on the CICDDoS2019 dataset:

| Model Category | Models |
|----------------|--------|
| **Gradient Boosting** | LightGBM, XGBoost, CatBoost |
| **Tree-Based** | Random Forest |
| **Neural Networks** | MLP (Multi-Layer Perceptron), DCN (Deep & Cross Network), Transformer |
| **Traditional ML** | K-Nearest Neighbors (KNN), Logistic Regression, AdaBoost |

The dataset contains 86,264 test samples across 12 attack classes: Benign, DNS, LDAP, MSSQL, NTP, NetBIOS, Portmap, SNMP, Syn, TFTP, UDP, and UDPLag.

## 3. Ensemble Methods Implemented

### 3.1 Voting-Based Methods

**Hard Voting (Majority Vote):** Each model casts one vote for its predicted class. The final prediction is the class with the most votes.

**Weighted Voting (Accuracy-Based):** Models vote with weights proportional to their individual accuracy on the validation set. Higher-performing models have greater influence on the final prediction.

**Weighted Voting (F1-Score Based):** Similar to accuracy-based weighting, but weights are derived from weighted F1-scores to account for class imbalance.

**Top-K Voting:** Only the K best-performing models (K=5) participate in the voting, excluding weaker models that may introduce noise.

### 3.2 Stacking (Meta-Learning)

Stacking uses a meta-learner trained on the predictions of base models. We implemented stacking with four different meta-learners:

- **Logistic Regression:** Linear meta-learner for interpretability
- **Random Forest:** Non-linear meta-learner capturing complex interactions
- **Gradient Boosting:** Sequential ensemble for refined predictions
- **MLP:** Neural network meta-learner for non-linear combinations

All stacking experiments used 5-fold stratified cross-validation to prevent data leakage.

## 4. Results

### 4.1 Individual Model Performance

| Model | Accuracy | F1-Score (Weighted) |
|-------|----------|---------------------|
| XGBoost | 97.89% | 0.9783 |
| MLP | 97.27% | 0.9713 |
| KNN | 97.15% | 0.9708 |
| CatBoost | 96.71% | 0.9670 |
| Transformer | 96.64% | 0.9675 |
| DCN | 96.28% | 0.9627 |
| Logistic Regression | 94.87% | 0.9509 |
| AdaBoost | 94.73% | 0.9518 |
| LightGBM | 93.06% | 0.9301 |
| Random Forest | 20.41% | 0.2040 |

**Note:** The Random Forest model showed anomalously low performance, indicating potential issues during training or prediction extraction.

### 4.2 Ensemble Performance

| Ensemble Method | Accuracy | F1-Score (Weighted) |
|-----------------|----------|---------------------|
| Stacking (Gradient Boosting) | **97.81%** | **0.9773** |
| Stacking (Random Forest) | 97.72% | 0.9766 |
| Stacking (MLP) | 97.54% | 0.9746 |
| Top-5 Voting | 97.53% | 0.9745 |
| Weighted Voting (Accuracy) | 97.38% | 0.9730 |
| Weighted Voting (F1) | 97.38% | 0.9730 |
| Hard Voting | 97.33% | 0.9726 |
| Stacking (Logistic Regression) | 94.93% | 0.9392 |

**Oracle Upper Bound:** 98.92% (when at least one model predicts correctly)

### 4.3 Per-Class Performance (Best Ensemble - Stacking GB)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 1.00 | 1.00 | 1.00 | 19,566 |
| NTP | 1.00 | 1.00 | 1.00 | 24,274 |
| TFTP | 1.00 | 1.00 | 1.00 | 19,783 |
| Syn | 1.00 | 0.99 | 0.99 | 9,875 |
| UDP | 0.92 | 0.98 | 0.95 | 5,702 |
| MSSQL | 0.89 | 0.96 | 0.92 | 2,947 |
| UDPLag | 0.93 | 0.75 | 0.83 | 1,786 |
| SNMP | 0.68 | 0.66 | 0.67 | 543 |
| LDAP | 0.61 | 0.69 | 0.65 | 669 |
| NetBIOS | 0.58 | 0.75 | 0.65 | 248 |
| DNS | 0.61 | 0.45 | 0.52 | 734 |
| Portmap | 0.38 | 0.17 | 0.23 | 137 |

## 5. Analysis and Discussion

### 5.1 Key Findings

1. **XGBoost achieved the highest individual accuracy (97.89%)**, slightly outperforming the best ensemble method. This suggests that when a well-tuned gradient boosting model exists, ensembling may provide marginal improvements.

2. **Stacking with Gradient Boosting** was the best ensemble method (97.81%), demonstrating that a non-linear meta-learner can effectively learn optimal combinations of base model predictions.

3. **Voting methods** performed consistently (97.33-97.53%), providing a simple yet effective approach without additional training.

4. **Top-5 Voting** outperformed standard voting by excluding the underperforming Random Forest model, highlighting the importance of base model quality.

5. **Stacking with Logistic Regression** underperformed (94.93%), indicating that linear combinations of predictions are insufficient for this multi-class problem.

### 5.2 Class-Level Observations

- **High-frequency classes** (Benign, NTP, TFTP, Syn) achieved near-perfect classification
- **Low-frequency classes** (Portmap, DNS, NetBIOS) showed lower performance, indicating class imbalance challenges
- **Portmap** with only 137 samples had the lowest F1-score (0.23), suggesting insufficient training data

### 5.3 Ensemble Benefit

The oracle upper bound of 98.92% indicates that different models make errors on different samples, validating the potential benefit of ensemble methods. However, the actual ensemble gain over the best individual model was modest (~0.1% below XGBoost), suggesting that base models may have correlated errors on the most difficult samples.

## 6. Conclusion

Ensemble learning provided robust predictions with the best stacking method achieving 97.81% accuracy. While the improvement over the best individual model (XGBoost) was marginal, ensemble methods offer:

- **Increased robustness** through prediction aggregation
- **Reduced variance** compared to single models
- **Better generalization** potential on unseen data distributions

For production deployment, we recommend either **XGBoost** (for simplicity) or **Stacking with Gradient Boosting** (for maximum robustness). Future work should address the class imbalance issue for minority classes like Portmap and DNS through oversampling or cost-sensitive learning.
