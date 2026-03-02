#!/usr/bin/env python3
"""Train MLP on land-only filtered data."""
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load FILTERED data (land only)
X = np.load('data/processed/expanded_X_land_only.npy')
y = np.load('data/processed/expanded_y_land_only.npy')

# Replace any remaining NaN/inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print(f'Data: {X.shape}, pos={y.sum():.0f}, neg={(1-y).sum():.0f}')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

pos_weight = (1 - y_train).sum() / y_train.sum()
print(f'pos_weight: {pos_weight:.2f}')

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_ds = TensorDataset(X_val_t, y_val_t)
val_loader = DataLoader(val_ds, batch_size=2048)

class YewMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = YewMLP().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print('Training MLP (land-only data)...')
t0 = time.time()
best_auc = 0
patience, no_improve = 10, 0

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            all_probs.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy())
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

model.load_state_dict(torch.load('results/predictions/south_vi_large/mlp_raw_model_expanded.pth'))
model.eval()
with torch.no_grad():
    y_proba = torch.sigmoid(model(X_val_t.to(device))).cpu().numpy()
y_pred = (y_proba >= 0.5).astype(int)
auc = roc_auc_score(y_val, y_proba)
f1 = f1_score(y_val, y_pred)
acc = accuracy_score(y_val, y_pred)
print(f'Final: AUC={auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}')

metrics = {'auc_roc': auc, 'f1': f1, 'accuracy': acc, 'train_time_s': train_time,
           'n_samples': len(X), 'n_positives': int(y.sum()), 'n_negatives': int((1-y).sum()),
           'filtered': 'land_only'}
with open('results/predictions/south_vi_large/mlp_raw_expanded_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('Saved mlp_raw_model_expanded.pth (land-only)')
