"""
Consolidate all model predictions from 5 CSV files into a single file.
Maps all text labels to encoded values using label_mapping.csv.
"""

import pandas as pd

# Load the label mapping
label_mapping_df = pd.read_csv('label_mapping.csv')
label_to_encoded = dict(zip(label_mapping_df['label'], label_mapping_df['encoded']))
print("Label mapping loaded:")
print(label_to_encoded)
print()

# Function to convert text labels to encoded values
def encode_label(value):
    """Convert text label to encoded value if it's a string."""
    if isinstance(value, str):
        return label_to_encoded.get(value, value)
    return value

# Load all CSV files
print("Loading CSV files...")

# 1. lightgbm_test_predictions.csv - has 'true_label' column (text format)
lightgbm_df = pd.read_csv('lightgbm_test_predictions.csv')
print(f"1. lightgbm_test_predictions.csv: {len(lightgbm_df)} rows")
print(f"   Columns at end: {lightgbm_df.columns[-5:].tolist()}")
# The 'prediction' column has text labels like "UDP", "Benign" - convert to encoded
lightgbm_df['lightgbm'] = lightgbm_df['prediction'].apply(encode_label)

# 2. prediction.csv - has 'catboost' column (already encoded)
prediction_df = pd.read_csv('prediction.csv')
print(f"2. prediction.csv: {len(prediction_df)} rows")
print(f"   Columns at end: {prediction_df.columns[-5:].tolist()}")

# 3. test_predictions.csv - has logistic_regression, adaboost, dcn, transformer (all encoded)
test_pred_df = pd.read_csv('test_predictions.csv')
print(f"3. test_predictions.csv: {len(test_pred_df)} rows")
print(f"   Columns at end: {test_pred_df.columns[-6:].tolist()}")

# 4. test_randomforest.csv - has 'randomforest' column (already encoded)
rf_df = pd.read_csv('test_randomforest.csv')
print(f"4. test_randomforest.csv: {len(rf_df)} rows")
print(f"   Columns at end: {rf_df.columns[-5:].tolist()}")

# 5. test_with_predictions.csv - has pred_knn, pred_xgboost, pred_mlp (all encoded)
knn_xgb_mlp_df = pd.read_csv('test_with_predictions.csv')
print(f"5. test_with_predictions.csv: {len(knn_xgb_mlp_df)} rows")
print(f"   Columns at end: {knn_xgb_mlp_df.columns[-5:].tolist()}")

print()

# Verify all files have the same number of rows
row_counts = {
    'lightgbm': len(lightgbm_df),
    'prediction': len(prediction_df),
    'test_predictions': len(test_pred_df),
    'test_randomforest': len(rf_df),
    'test_with_predictions': len(knn_xgb_mlp_df)
}
print(f"Row counts: {row_counts}")

# Create the consolidated dataframe
# Use the true label (label_encoded) from one of the files as the ground truth
consolidated = pd.DataFrame()

# Add the true label (encoded) - using from lightgbm_df
consolidated['true_label'] = lightgbm_df['label_encoded']

# Add predictions from each file
# 1. LightGBM (converted from text to encoded)
consolidated['lightgbm'] = lightgbm_df['lightgbm']

# 2. CatBoost
consolidated['catboost'] = prediction_df['catboost']

# 3. Logistic Regression, AdaBoost, DCN, Transformer
consolidated['logistic_regression'] = test_pred_df['label_encoded_logistic_regression']
consolidated['adaboost'] = test_pred_df['label_encoded_adaboost']
consolidated['dcn'] = test_pred_df['label_encoded_dcn']
consolidated['transformer'] = test_pred_df['label_encoded_transformer']

# 4. Random Forest
consolidated['randomforest'] = rf_df['randomforest']

# 5. KNN, XGBoost, MLP
consolidated['knn'] = knn_xgb_mlp_df['pred_knn']
consolidated['xgboost'] = knn_xgb_mlp_df['pred_xgboost']
consolidated['mlp'] = knn_xgb_mlp_df['pred_mlp']

# Display info about the consolidated dataframe
print("\n" + "="*50)
print("CONSOLIDATED PREDICTIONS DATAFRAME")
print("="*50)
print(f"\nShape: {consolidated.shape}")
print(f"\nColumns: {consolidated.columns.tolist()}")
print(f"\nFirst 10 rows:")
print(consolidated.head(10))

print(f"\nData types:")
print(consolidated.dtypes)

print(f"\nUnique values per column:")
for col in consolidated.columns:
    print(f"  {col}: {consolidated[col].nunique()} unique values")

# Save to CSV
output_file = 'consolidated_predictions.csv'
consolidated.to_csv(output_file, index=False)
print(f"\nSaved consolidated predictions to: {output_file}")

# Also create a version with label names for easier reading
consolidated_with_names = consolidated.copy()
encoded_to_label = dict(zip(label_mapping_df['encoded'], label_mapping_df['label']))
for col in consolidated_with_names.columns:
    consolidated_with_names[col + '_name'] = consolidated_with_names[col].map(encoded_to_label)

# Reorder columns to have encoded and name side by side
cols_ordered = []
for col in consolidated.columns:
    cols_ordered.append(col)
    cols_ordered.append(col + '_name')

consolidated_with_names = consolidated_with_names[cols_ordered]
consolidated_with_names.to_csv('consolidated_predictions_with_names.csv', index=False)
print("Saved consolidated predictions with label names to: consolidated_predictions_with_names.csv")

print("\nDone!")
