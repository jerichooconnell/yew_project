#!/usr/bin/env python3
"""Train MLP on expanded dataset (raw embeddings, no scaling)."""
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import pickle
import json
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load data
X = np.load('data/processed/expanded_X_all.npy')
y = np.load('data/processed/expanded_y_all.npy')
print(f'Data: {X.shape}, pos={y.sum():.0f}, neg={(1-y).sum():.0f}')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

# Weighted sampler for class imbalance
pos_weight = (1 - y_train).sum() / y_train.sum()
print(f'pos_weight: {pos_weight:.2f}')

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_ds = TensorDataset(X_val_t, y_val_t)
val_loader = DataLoader(val_ds, batch_size=2048)

# MLP model (same architecture as existing)
class YewMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = YewMLP().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print('Training MLP...')
t0 = time.time()
best_auc = 0
patience = 10
no_improve = 0

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    # Validate
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(yb.numpy())
    auc = roc_auc_score(all_y, all_probs)
    
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), 'results/predictions/south_vi_large/mlp_raw_model_expanded.pth')
        no_improve = 0
    else:
        no_improve += 1
    
    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1}: AUC={auc:.4f} (best={best_auc:.4f})')
    
    if no_improve >= patience:
        print(f'  Early stopping at epoch {epoch+1}')
        break

train_time = time.time() - t0
print(f'Training time: {train_time:.1f}s')

# Load best model and evaluate
model.load_state_dict(torch.load('results/predictions/south_vi_large/mlp_raw_model_expanded.pth'))
model.eval()
with torch.no_grad():
    y_proba = torch.sigmoid(model(X_val_t.to(device))).cpu().numpy()
y_pred = (y_proba >= 0.5).astype(int)
auc = roc_auc_score(y_val, y_proba)
f1 = f1_score(y_val, y_pred)
acc = accuracy_score(y_val, y_pred)
print(f'AUC={auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}')

# Speed test: 1M pixels
print('\nInference speed test (1M pixels)...')
test_X = torch.randn(1_000_000, 64, dtype=torch.float32)

# GPU MLP
model.eval()
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(test_X), 500_000):
        batch = test_X[i:i+500_000].to(device)
        _ = torch.sigmoid(model(batch)).cpu()
mlp_gpu_time = time.time() - t0
print(f'MLP (GPU):    {mlp_gpu_time:.2f}s')

# XGBoost
import xgboost as xgb
bst = xgb.Booster()
bst.load_model('results/predictions/south_vi_large/xgb_raw_model_expanded.json')
test_np = test_X.numpy()
dtest = xgb.DMatrix(test_np)
t0 = time.time()
_ = bst.predict(dtest)
xgb_time = time.time() - t0
print(f'XGBoost (CPU): {xgb_time:.2f}s')

# RF
with open('results/predictions/south_vi_large/rf_raw_model_expanded.pkl', 'rb') as f:
    rf = pickle.load(f)
t0 = time.time()
_ = rf.predict_proba(test_np)[:, 1]
rf_time = time.time() - t0
print(f'RF (CPU):     {rf_time:.2f}s')

print(f'\nSpeedups vs RF: MLP={rf_time/mlp_gpu_time:.1f}x, XGB={rf_time/xgb_time:.1f}x')

# Save metrics
metrics = {'auc_roc': auc, 'f1': f1, 'accuracy': acc, 'train_time_s': train_time,
           'inference_1M_s': mlp_gpu_time, 'device': str(device)}
with open('results/predictions/south_vi_large/mlp_raw_expanded_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print('\nSaved mlp_raw_model_expanded.pth')
