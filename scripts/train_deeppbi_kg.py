#!/usr/bin/env python3
"""
DeepPBI-KG Experimental Protocol (C1, C2, C3)
PyTorch Neural Network for Phage-Host Interaction Prediction.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, f1_score, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "features" / "deeppbi_kg"
MODELS_DIR = BASE_DIR / "models" / "deeppbi_kg"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "C1_train_B").mkdir(exist_ok=True)
(MODELS_DIR / "C2_train_A").mkdir(exist_ok=True)
(MODELS_DIR / "C3_finetune").mkdir(exist_ok=True)

# DeepPBI-KG PyTorch DNN Architecture
class DeepPBIKG_Model(nn.Module):
    def __init__(self, input_dim=2560):
        super(DeepPBIKG_Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.sigmoid(self.fc3(x))
        return x

def calculate_metrics(y_true, y_pred_prob):
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    mcc = matthews_corrcoef(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    acc = accuracy_score(y_true, y_pred_bin)
    return {
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "MCC": float(mcc),
        "F1": float(f1),
        "Accuracy": float(acc)
    }

def run_deeppbi_kg_experiments():
    print("=== Running DeepPBI-KG Experiments (C1, C2, C3) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    # Load Dataset A & B features
    data_a = np.load(FEATURES_DIR / "dataset_A_features.npz")
    data_b = np.load(FEATURES_DIR / "dataset_B_features.npz")
    
    X_a, y_a = torch.tensor(data_a["X"], dtype=torch.float32), torch.tensor(data_a["y"], dtype=torch.float32).unsqueeze(1)
    X_b, y_b = torch.tensor(data_b["X"], dtype=torch.float32), torch.tensor(data_b["y"], dtype=torch.float32).unsqueeze(1)
    
    # Load Dataset B split indices
    with open(BASE_DIR / "data" / "klebphacol_train_indices.json", "r") as f:
        train_b_idx = json.load(f)
    with open(BASE_DIR / "data" / "klebphacol_test_indices.json", "r") as f:
        test_b_idx = json.load(f)
        
    X_b_train, y_b_train = X_b[train_b_idx], y_b[train_b_idx]
    X_b_test, y_b_test = X_b[test_b_idx], y_b[test_b_idx]
    
    results = {}
    
    # -------------------------------------------------------------
    # C1: Train on B, Test on B
    # -------------------------------------------------------------
    print("\n--- Condition C1: Train on B, Test on B ---")
    model_c1 = DeepPBIKG_Model().to(device)
    optimizer_c1 = optim.Adam(model_c1.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    loader_b_train = DataLoader(TensorDataset(X_b_train, y_b_train), batch_size=64, shuffle=True)
    
    model_c1.train()
    for epoch in range(25):
        for bx, by in loader_b_train:
            bx, by = bx.to(device), by.to(device)
            optimizer_c1.zero_grad()
            out = model_c1(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer_c1.step()
            
    model_c1.eval()
    with torch.no_grad():
        preds_c1 = model_c1(X_b_test.to(device)).cpu().numpy().flatten()
    metrics_c1 = calculate_metrics(y_b_test.numpy().flatten(), preds_c1)
    results["C1_Train_B"] = metrics_c1
    torch.save(model_c1.state_dict(), MODELS_DIR / "C1_train_B" / "model.pt")
    print(f"  C1 Test Results: ROC_AUC={metrics_c1['ROC_AUC']:.4f}, PR_AUC={metrics_c1['PR_AUC']:.4f}, MCC={metrics_c1['MCC']:.4f}")
    
    # -------------------------------------------------------------
    # C2: Train on A, Test on B
    # -------------------------------------------------------------
    print("\n--- Condition C2: Train on A, Test on B ---")
    model_c2 = DeepPBIKG_Model().to(device)
    optimizer_c2 = optim.Adam(model_c2.parameters(), lr=1e-3, weight_decay=1e-4)
    
    loader_a_train = DataLoader(TensorDataset(X_a, y_a), batch_size=128, shuffle=True)
    
    model_c2.train()
    for epoch in range(15):
        for bx, by in loader_a_train:
            bx, by = bx.to(device), by.to(device)
            optimizer_c2.zero_grad()
            out = model_c2(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer_c2.step()
            
    model_c2.eval()
    with torch.no_grad():
        preds_c2 = model_c2(X_b_test.to(device)).cpu().numpy().flatten()
    metrics_c2 = calculate_metrics(y_b_test.numpy().flatten(), preds_c2)
    results["C2_Train_A"] = metrics_c2
    torch.save(model_c2.state_dict(), MODELS_DIR / "C2_train_A" / "model.pt")
    print(f"  C2 Test Results: ROC_AUC={metrics_c2['ROC_AUC']:.4f}, PR_AUC={metrics_c2['PR_AUC']:.4f}, MCC={metrics_c2['MCC']:.4f}")

    # -------------------------------------------------------------
    # C3: Train on A, Fine-tune on B, Test on B
    # -------------------------------------------------------------
    print("\n--- Condition C3: Train on A, Fine-tune on B, Test on B ---")
    model_c3 = DeepPBIKG_Model().to(device)
    model_c3.load_state_dict(torch.load(MODELS_DIR / "C2_train_A" / "model.pt"))
    
    # Fine-tuning with lower learning rate
    optimizer_c3 = optim.Adam(model_c3.parameters(), lr=1e-4, weight_decay=1e-4)
    
    model_c3.train()
    for epoch in range(15):
        for bx, by in loader_b_train:
            bx, by = bx.to(device), by.to(device)
            optimizer_c3.zero_grad()
            out = model_c3(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer_c3.step()
            
    model_c3.eval()
    with torch.no_grad():
        preds_c3 = model_c3(X_b_test.to(device)).cpu().numpy().flatten()
    metrics_c3 = calculate_metrics(y_b_test.numpy().flatten(), preds_c3)
    results["C3_Finetune"] = metrics_c3
    torch.save(model_c3.state_dict(), MODELS_DIR / "C3_finetune" / "model.pt")
    print(f"  C3 Test Results: ROC_AUC={metrics_c3['ROC_AUC']:.4f}, PR_AUC={metrics_c3['PR_AUC']:.4f}, MCC={metrics_c3['MCC']:.4f}")
    
    # Save overall DeepPBI-KG results summary
    df_res = pd.DataFrame(results).T
    df_res.to_csv(RESULTS_DIR / "deeppbi_kg_results.csv")
    print("\nDeepPBI-KG Experiments Summary:")
    print(df_res)

if __name__ == "__main__":
    run_deeppbi_kg_experiments()
