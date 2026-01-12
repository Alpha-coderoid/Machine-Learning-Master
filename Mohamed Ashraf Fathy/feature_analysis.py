"""
Feature Analysis for DDoS Detection
====================================
Scientific analysis to determine which features to keep/remove.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('processed_data/train_unscaled.csv')

X = train.drop(columns=['label', 'label_encoded'])
y = train['label_encoded']

print(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"Classes: {train['label'].nunique()}")

# ============================================================================
# 1. VARIANCE ANALYSIS - Remove near-zero variance features
# ============================================================================
print("\n" + "="*60)
print("1. VARIANCE ANALYSIS")
print("="*60)
print("Features with near-zero variance provide no discriminative power.")

variances = X.var().sort_values()
low_var_threshold = 0.01
low_var_features = variances[variances < low_var_threshold]

print(f"\nFeatures with variance < {low_var_threshold}:")
for feat, var in low_var_features.items():
    print(f"  {feat}: {var:.6f}")

# ============================================================================
# 2. CORRELATION ANALYSIS - Remove highly correlated (redundant) features
# ============================================================================
print("\n" + "="*60)
print("2. CORRELATION ANALYSIS")
print("="*60)
print("Highly correlated features are redundant - keeping one is sufficient.")

corr_matrix = X.corr().abs()

# Find pairs with correlation > 0.95
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

print(f"\nHighly correlated pairs (r > 0.95): {len(high_corr_pairs)}")
for pair in sorted(high_corr_pairs, key=lambda x: -x['correlation'])[:15]:
    print(f"  {pair['feature1'][:30]:30} <-> {pair['feature2'][:30]:30} : {pair['correlation']:.3f}")

# ============================================================================
# 3. FEATURE IMPORTANCE (Random Forest)
# ============================================================================
print("\n" + "="*60)
print("3. FEATURE IMPORTANCE (Random Forest)")
print("="*60)
print("Tree-based importance shows which features best split the data.")

# Sample for speed
sample_size = min(50000, len(X))
np.random.seed(42)
idx = np.random.choice(len(X), sample_size, replace=False)
X_sample = X.iloc[idx]
y_sample = y.iloc[idx]

rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_sample, y_sample)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
for i, row in importance_df.head(20).iterrows():
    print(f"  {row['feature'][:40]:40} : {row['importance']:.4f}")

print("\nBottom 15 Least Important Features:")
for i, row in importance_df.tail(15).iterrows():
    print(f"  {row['feature'][:40]:40} : {row['importance']:.6f}")

# ============================================================================
# 4. MUTUAL INFORMATION - Statistical dependency with target
# ============================================================================
print("\n" + "="*60)
print("4. MUTUAL INFORMATION ANALYSIS")
print("="*60)
print("Measures statistical dependency between feature and target class.")

mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mutual_info': mi_scores
}).sort_values('mutual_info', ascending=False)

print("\nTop 15 by Mutual Information:")
for i, row in mi_df.head(15).iterrows():
    print(f"  {row['feature'][:40]:40} : {row['mutual_info']:.4f}")

print("\nBottom 10 by Mutual Information (least informative):")
for i, row in mi_df.tail(10).iterrows():
    print(f"  {row['feature'][:40]:40} : {row['mutual_info']:.4f}")

# ============================================================================
# 5. FEATURE CATEGORIES & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*60)
print("5. RECOMMENDATIONS")
print("="*60)

# Combine metrics
combined = importance_df.merge(mi_df, on='feature')
combined['rank_importance'] = combined['importance'].rank(ascending=False)
combined['rank_mi'] = combined['mutual_info'].rank(ascending=False)
combined['avg_rank'] = (combined['rank_importance'] + combined['rank_mi']) / 2
combined = combined.sort_values('avg_rank')

# Features to definitely keep (top performers)
keep_features = combined.head(30)['feature'].tolist()

# Features to consider removing (low importance + low MI)
remove_candidates = combined[
    (combined['importance'] < 0.005) & (combined['mutual_info'] < 0.1)
]['feature'].tolist()

print("\n✅ FEATURES TO KEEP (High Importance + High MI):")
for f in keep_features[:20]:
    row = combined[combined['feature'] == f].iloc[0]
    print(f"  {f[:45]:45} | Imp: {row['importance']:.4f} | MI: {row['mutual_info']:.4f}")

print(f"\n❌ FEATURES TO CONSIDER REMOVING ({len(remove_candidates)} features):")
for f in remove_candidates:
    row = combined[combined['feature'] == f].iloc[0]
    print(f"  {f[:45]:45} | Imp: {row['importance']:.6f} | MI: {row['mutual_info']:.4f}")

# Save results
combined.to_csv('feature_analysis.csv', index=False)
print(f"\nFull analysis saved to: feature_analysis.csv")
