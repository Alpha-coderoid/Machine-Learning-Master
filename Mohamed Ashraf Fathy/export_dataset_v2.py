"""
Export Preprocessed DDoS Dataset (v2 - Feature Selected)
=========================================================
- Removes 17 useless features (zero variance, bulk features)
- Removes redundant features (correlation > 0.95)
- Drops WebDDoS class (only 51 samples)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = "archive"
OUTPUT_DIR = "processed_data_v2"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features to REMOVE (zero variance / useless)
FEATURES_TO_REMOVE = [
    # Zero variance / constant
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'FIN Flag Count', 'PSH Flag Count', 'ECE Flag Count',
    # Bulk transfer features (all zeros in DDoS traffic)
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    # Low value
    'CWE Flag Count', 'Bwd Packet Length Std',
]

# Redundant features to remove (keep first, remove second)
REDUNDANT_TO_REMOVE = [
    'Subflow Fwd Packets',   # Identical to Total Fwd Packets
    'Subflow Bwd Packets',   # Identical to Total Backward Packets  
    'Subflow Fwd Bytes',     # Identical to Fwd Packets Length Total
    'Subflow Bwd Bytes',     # Identical to Bwd Packets Length Total
    'Avg Fwd Segment Size',  # Identical to Fwd Packet Length Mean
    'Avg Bwd Segment Size',  # Identical to Bwd Packet Length Mean
    'Fwd IAT Total',         # ~ Flow Duration (r=0.999)
    'Fwd IAT Min',           # ~ Flow IAT Min (r=0.998)
    'Fwd IAT Max',           # ~ Flow IAT Max (r=0.995)
    'RST Flag Count',        # Identical to Fwd PSH Flags
]

COLUMNS_TO_DROP = [
    'Unnamed: 0', 'Flow ID', ' Source IP', ' Destination IP',
    ' Timestamp', 'SimillarHTTP', ' Inbound'
]

# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("EXPORTING DATASET v2 (Feature Selected)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load all parquet files
    print("\n[1/7] Loading data...")
    all_dfs = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(DATA_DIR, filename))
            all_dfs.append(df)
            print(f"  Loaded: {filename} ({len(df):,} rows)")
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total samples: {len(df):,}")
    
    # 2. Normalize labels
    print("\n[2/7] Normalizing labels...")
    label_col = ' Label' if ' Label' in df.columns else 'Label'
    
    def normalize_label(label):
        label = str(label).strip()
        if label.startswith('DrDoS_'):
            label = label.replace('DrDoS_', '')
        return {'UDP-lag': 'UDPLag', 'BENIGN': 'Benign'}.get(label, label)
    
    df[label_col] = df[label_col].apply(normalize_label)
    
    # 3. Drop WebDDoS class
    print("\n[3/7] Dropping WebDDoS class...")
    before = len(df)
    df = df[df[label_col] != 'WebDDoS']
    print(f"  Removed {before - len(df)} WebDDoS samples")
    print(f"  Remaining classes: {sorted(df[label_col].unique())}")
    
    # 4. Drop non-predictive columns
    print("\n[4/7] Dropping non-predictive columns...")
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 5. Separate features and labels, process features
    print("\n[5/7] Feature selection...")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    print(f"  Starting with {len(numeric_cols)} numeric features")
    
    # Remove useless features
    useless_in_data = [f for f in FEATURES_TO_REMOVE if f in X.columns]
    X = X.drop(columns=useless_in_data, errors='ignore')
    print(f"  Removed {len(useless_in_data)} useless features")
    
    # Remove redundant features
    redundant_in_data = [f for f in REDUNDANT_TO_REMOVE if f in X.columns]
    X = X.drop(columns=redundant_in_data, errors='ignore')
    print(f"  Removed {len(redundant_in_data)} redundant features")
    
    print(f"  Final feature count: {len(X.columns)}")
    
    # Handle infinity and missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median()
    X = X.fillna(medians)
    
    # 6. Create train/test split
    print("\n[6/7] Creating train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    # 7. Save files
    print("\n[7/7] Saving files...")
    
    # Training set
    train_df = X_train.copy()
    train_df['label'] = y_train.values
    train_df['label_encoded'] = label_encoder.transform(y_train)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/train.csv")
    
    # Test set  
    test_df = X_test.copy()
    test_df['label'] = y_test.values
    test_df['label_encoded'] = label_encoder.transform(y_test)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/test.csv")
    
    # Label mapping
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    pd.DataFrame(list(label_mapping.items()), columns=['encoded', 'label']).to_csv(
        os.path.join(OUTPUT_DIR, 'label_mapping.csv'), index=False
    )
    print(f"  Saved: {OUTPUT_DIR}/label_mapping.csv")
    
    # Feature list
    pd.DataFrame({'feature': X.columns.tolist()}).to_csv(
        os.path.join(OUTPUT_DIR, 'features.csv'), index=False
    )
    print(f"  Saved: {OUTPUT_DIR}/features.csv")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Random State: {RANDOM_STATE}")
    print(f"  Test Size: {TEST_SIZE}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {len(X.columns)} (reduced from 77)")
    print(f"  Classes: {len(label_encoder.classes_)} (WebDDoS removed)")
    print(f"\n  Classes: {list(label_encoder.classes_)}")
    
    print("\n" + "=" * 60)
    print("FILES CREATED:")
    print("=" * 60)
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── train.csv         ({len(train_df):,} rows × {len(X.columns)+2} cols)")
    print(f"    ├── test.csv          ({len(test_df):,} rows × {len(X.columns)+2} cols)")
    print(f"    ├── label_mapping.csv")
    print(f"    └── features.csv      ({len(X.columns)} features)")
    print("\nDONE!")

if __name__ == "__main__":
    main()
