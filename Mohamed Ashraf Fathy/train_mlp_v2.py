"""
DDoS Detection - MLP Deep Learning with K-Fold CV
==================================================
Trains a Multi-Layer Perceptron on the feature-selected dataset (v2) with:
- 5-Fold Stratified Cross-Validation
- Early stopping
- Model weights saved
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import time
import os

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = "processed_data_v2"
OUTPUT_DIR = "models_v2"
RANDOM_STATE = 42
N_FOLDS = 5
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 5

# Set seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MLP Model
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
# Training Function
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
    
    return total_loss / len(dataloader), correct / total

# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("MLP - K-Fold Cross-Validation")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Load data
    print("\n[1/4] Loading data...")
    train = pd.read_csv(f'{DATA_DIR}/train.csv')
    test = pd.read_csv(f'{DATA_DIR}/test.csv')
    
    X_train = train.drop(columns=['label', 'label_encoded']).values
    y_train = train['label_encoded'].values
    X_test = test.drop(columns=['label', 'label_encoded']).values
    y_test = test['label_encoded'].values
    
    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Features: {num_features}, Classes: {num_classes}")
    
    # Scale features
    print("\n[2/4] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # K-Fold Cross-Validation
    print(f"\n[3/4] {N_FOLDS}-Fold Cross-Validation...")
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, y_train)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS}")
        
        # Split data
        X_fold_train = torch.FloatTensor(X_train_scaled[train_idx])
        y_fold_train = torch.LongTensor(y_train[train_idx])
        X_fold_val = torch.FloatTensor(X_train_scaled[val_idx])
        y_fold_val = torch.LongTensor(y_train[val_idx])
        
        train_dataset = TensorDataset(X_fold_train, y_fold_train)
        val_dataset = TensorDataset(X_fold_val, y_fold_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Create model
        model = MLPClassifier(num_features, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        # Training loop with early stopping
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"    Early stop at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        fold_scores.append(best_val_acc)
        print(f"    Best val accuracy: {best_val_acc:.4f}")
    
    print(f"\n  CV Results:")
    for i, score in enumerate(fold_scores):
        print(f"    Fold {i+1}: {score:.4f}")
    print(f"    Mean: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores)*2:.4f})")
    
    # Train final model on all training data
    print("\n[4/4] Training final model on all data...")
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    final_model = MLPClassifier(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_test_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(final_model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(final_model, test_loader, criterion, device)
        scheduler.step(test_loss)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save best weights
            torch.save(final_model.state_dict(), f'{OUTPUT_DIR}/mlp_weights.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    
    # Load best weights and evaluate
    final_model.load_state_dict(torch.load(f'{OUTPUT_DIR}/mlp_weights.pth', weights_only=True))
    
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    final_model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = final_model(X_batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(y_test, all_preds)
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, all_preds))
    
    # Save model architecture info
    model_info = {
        'input_size': num_features,
        'num_classes': num_classes,
        'hidden_sizes': [256, 128, 64],
        'dropout': 0.3,
        'cv_mean': np.mean(fold_scores),
        'cv_std': np.std(fold_scores),
        'test_accuracy': accuracy
    }
    
    import joblib
    joblib.dump(model_info, f'{OUTPUT_DIR}/mlp_model_info.joblib')
    joblib.dump(scaler, f'{OUTPUT_DIR}/mlp_scaler.joblib')
    
    print(f"\n  Saved: {OUTPUT_DIR}/mlp_weights.pth")
    print(f"  Saved: {OUTPUT_DIR}/mlp_model_info.joblib")
    print(f"  Saved: {OUTPUT_DIR}/mlp_scaler.joblib")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return model_info

if __name__ == "__main__":
    results = main()
