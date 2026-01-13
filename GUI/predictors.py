import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import os
from catboost import CatBoostClassifier
import torch
import torch.nn.functional as F

def load_label_mapping(path="ML_Project/label_mapping.csv"):
    df = pd.read_csv(path)

    # Detect mapping direction automatically
    if "encoded" in df.columns and "label" in df.columns:
        mapping = dict(zip(df["encoded"], df["label"]))
    else:
        raise ValueError("label_mapping.csv must contain 'label' and 'label_encoded' columns")

    return mapping

LABEL_MAPPING = load_label_mapping()

# ======================
# LightGBM
# ======================
def predict_lightgbm(df):
    model = joblib.load("ML_Project/Models/best_lightgbm_model.pkl")

    X = df.drop(columns=["label", "label_encoded"], errors="ignore")
    X = X.values.astype(np.float32)

    preds_encoded = model.predict(X)
    preds = [LABEL_MAPPING[int(p)] for p in preds_encoded]

    df_out = df.copy()
    df_out["prediction"] = preds
    return df_out

# ======================
# CNN-LSTM
# ======================
def predict_cnn_lstm(df):
    model = tf.keras.models.load_model(
        "ML_Project/Models/best_cnn_lstm_fold_model.h5"
    )

    X = df.drop(columns=["label", "label_encoded"], errors="ignore")
    X = X.values.astype(np.float32)

    # reshape for (samples, timesteps=1, features)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    probs = model.predict(X, verbose=0)
    preds_encoded = probs.argmax(axis=1)
    preds = [LABEL_MAPPING[int(p)] for p in preds_encoded]

    df_out = df.copy()
    df_out["prediction"] = preds
    return df_out


# ======================
# Random Forest
# ======================

def predict_random_forest(df):
    # Load model (+ scaler if used)
    model = joblib.load("ML_Project/Models/random_forest.pkl")

    # OPTIONAL: load scaler if RF was trained with one
    scaler_path = "ML_Project/Models/scaler.joblib"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # Prepare features
    X = df.drop(columns=["label", "label_encoded"], errors="ignore")
    X = X.values.astype(np.float32)

    # Apply scaler if available
    if scaler:
        X = scaler.transform(X)

    # Predict (encoded)
    y_pred_encoded = model.predict(X)

    # Decode labels
    y_pred_labels = [LABEL_MAPPING[int(i)] for i in y_pred_encoded]

    # Output
    df_out = df.copy()
    df_out["prediction"] = y_pred_labels
    return df_out

# ======================
# LogisticRegression
# ======================
def predict_logistic_regression(df):
    model = joblib.load("ML_Project/Models/logistic_regression.joblib")
    scaler = joblib.load("ML_Project/Models/scaler.joblib")

    # Prepare features
    X = df.drop(columns=["label", "label_encoded"], errors="ignore")
    X = scaler.transform(X.values.astype(np.float32))

    # Predict
    y_pred_encoded = model.predict(X)
    y_proba = model.predict_proba(X)

    # Decode labels
    y_pred_labels = [LABEL_MAPPING[int(i)] for i in y_pred_encoded]

    df_out = df.copy()
    df_out["prediction"] = y_pred_labels
    df_out["confidence"] = np.max(y_proba, axis=1)

    return df_out

# ======================
# AdaBoost
# ======================
def predict_adaboost(df):
    # Load model
    model = joblib.load("ML_Project/Models/adaboost.joblib")

    # Prepare features (drop labels if present)
    X = df.drop(columns=["label", "label_encoded"], errors="ignore")
    X = X.values.astype(np.float32)

    # Predict
    y_pred_encoded = model.predict(X)

    # Probabilities (AdaBoost supports this)
    y_proba = model.predict_proba(X)

    # Decode labels
    y_pred_labels = [LABEL_MAPPING[int(i)] for i in y_pred_encoded]

    # Output DataFrame
    df_out = df.copy()
    df_out["prediction"] = y_pred_labels
    df_out["confidence"] = np.max(y_proba, axis=1)

    return df_out

# ======================
# CVAE
# ======================

def predict_cvae(df):
    # Load trained CVAE model (IMPORTANT: compile=False)
    model = tf.keras.models.load_model(
        "ML_Project/Models/best_cvae_model.h5",
        compile=False
    )

    # Prepare features
    X = df.drop(columns=["label", "label_encoded"], errors="ignore")
    X = X.values.astype(np.float32)

    # Predict probabilities
    probs = model.predict(X, verbose=0)

    # Encoded predictions
    y_pred_encoded = np.argmax(probs, axis=1)

    # Decode labels
    y_pred_labels = [LABEL_MAPPING[int(i)] for i in y_pred_encoded]

    # Output DataFrame
    df_out = df.copy()
    df_out["prediction"] = y_pred_labels
    df_out["confidence"] = np.max(probs, axis=1)

    return df_out

# ======================
# CatBoost
# ======================

def predict_catboost(df):

    # Load model
    cat_model = CatBoostClassifier()
    cat_model.load_model("ML_Project/Models/catboost_model.cbm")

    # Prepare features
    X = df.drop(columns=["label", "label_encoded"], errors="ignore")
    X = X.values.astype(np.float32)

    # Predict (encoded labels)
    y_pred_encoded = cat_model.predict(X).astype(int).flatten()

    # Predict probabilities (optional but recommended)
    y_proba = cat_model.predict_proba(X)

    # Decode labels
    y_pred_labels = [LABEL_MAPPING[int(i)] for i in y_pred_encoded]

    # Output DataFrame
    df_out = df.copy()
    df_out["prediction"] = y_pred_labels
    df_out["confidence"] = np.max(y_proba, axis=1)

    return df_out

# ======================
# XGBoost
# ======================

def predict_xgboost(df):
    """
    XGBoost prediction using DataFrame input.
    Consistent with all other models.
    """

    # Load model ONCE (or move outside if you want caching)
    model = joblib.load("ML_Project/xgboost_deploy/xgboost_model.joblib")

    # OPTIONAL: load scaler if used
    scaler_path = "ML_Project/xgboost_deploy/scaler.joblib"
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # Prepare features
    X = df.drop(columns=["label", "label_encoded"], errors="ignore")
    X = X.values.astype(np.float32)

    if scaler:
        X = scaler.transform(X)

    # Predict encoded labels
    y_pred_encoded = model.predict(X)

    # Predict probabilities (if supported)
    try:
        y_proba = model.predict_proba(X)
        confidence = np.max(y_proba, axis=1)
    except:
        confidence = np.ones(len(y_pred_encoded))

    # Decode labels
    y_pred_labels = [LABEL_MAPPING[int(i)] for i in y_pred_encoded]

    # Output DataFrame
    df_out = df.copy()
    df_out["prediction"] = y_pred_labels
    df_out["confidence"] = confidence

    return df_out

# ======================
# DCN (Deep & Cross Network)
# ======================

def predict_dcn(df, device=None):
    """
    DCN prediction using DataFrame input.
    Consistent with all other models.
    """

    # Select device automatically
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths (relative & safe)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(SCRIPT_DIR, "Models")

    model_path = os.path.join(MODELS_DIR, "dcn_model.pt")
    scaler_path = os.path.join(MODELS_DIR, "dcn_scaler.joblib")

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    n_features = checkpoint["n_features"]
    n_classes = checkpoint["n_classes"]

    # Rebuild DCN architecture (must match training)
    model = DCN(
        n_features=n_features,
        n_classes=n_classes,
        n_cross_layers=3,
        deep_layers=[128, 64, 32],
        dropout=0.1
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ======================
    # Prepare features
    # ======================
    X = df.drop(columns=["label", "label_encoded"], errors="ignore")
    X = X.values.astype(np.float32)

    # Scale
    X_scaled = scaler.transform(X)

    # To tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # ======================
    # Predict
    # ======================
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()

    # Encoded predictions
    y_pred_encoded = np.argmax(probs, axis=1)

    # Decode labels
    y_pred_labels = [LABEL_MAPPING[int(i)] for i in y_pred_encoded]

    # Confidence
    confidence = np.max(probs, axis=1)

    # ======================
    # Output DataFrame
    # ======================
    df_out = df.copy()
    df_out["prediction"] = y_pred_labels
    df_out["confidence"] = confidence

    return df_out

# ======================
# Tabular Transformer
# ======================

def predict_tabular_transformer(df, device=None):
    """
    Tabular Transformer prediction using DataFrame input.
    Consistent with all other models.
    """

    # Select device automatically
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths (relative & safe)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(SCRIPT_DIR, "Models")

    model_path = "ML_Project/Models/scaler.joblib"
    scaler_path = "ML_Project/Models/scaler.joblib"

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    n_features = checkpoint["n_features"]
    n_classes = checkpoint["n_classes"]

    # Rebuild Transformer architecture (MUST match training)
    model = TabularTransformer(
        n_features=n_features,
        n_classes=n_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ======================
    # Prepare features
    # ======================
    X = df.drop(columns=["label", "label_encoded"], errors="ignore")
    X = X.values.astype(np.float32)

    # Scale
    X_scaled = scaler.transform(X)

    # To tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # ======================
    # Predict
    # ======================
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()

    # Encoded predictions
    y_pred_encoded = np.argmax(probs, axis=1)

    # Decode labels
    y_pred_labels = [LABEL_MAPPING[int(i)] for i in y_pred_encoded]

    # Confidence
    confidence = np.max(probs, axis=1)

    # ======================
    # Output DataFrame
    # ======================
    df_out = df.copy()
    df_out["prediction"] = y_pred_labels
    df_out["confidence"] = confidence

    return df_out

# ======================
# MODEL REGISTRY
# ======================
MODELS = {
    "LightGBM": predict_lightgbm,
    "CVAE": predict_cvae,    
    "CNN-LSTM": predict_cnn_lstm,
    "Random Forest": predict_random_forest,
    "AdaBoost": predict_adaboost,
    "Logistic Regression": predict_logistic_regression,
    "CatBoost": predict_catboost,
    "XGBoost":predict_xgboost,
    "DCN": predict_dcn,
    "Tabular Transformer": predict_tabular_transformer
}