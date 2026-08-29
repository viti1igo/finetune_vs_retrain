#!/usr/bin/env python3
"""
Generate and Export Confusion Matrices & Classification Metrics for Phase 7 Extensions:
1. Leave-One-Subject-Out Cross-Validation (LOSO CV) for DeepPBI-KG
2. Leave-One-Subject-Out Cross-Validation (LOSO CV) for PhageHostLearn
3. Reverse Transfer (Pretrain on Dataset B, Fine-tune on Dataset A)

Saves confusion matrix PNG plots and CSV metrics into results/confusion_matrices/
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, auc

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
CM_DIR = RESULTS_DIR / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

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

def compute_all_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_bin)
    
    roc_auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall_vals, precision_vals)
    
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1_Score": float(f1),
        "MCC": float(mcc),
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc)
    }, cm

def plot_and_save_cm(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-Infection (0)", "Infection (1)"],
                yticklabels=["Non-Infection (0)", "Infection (1)"])
    plt.title(title, fontweight="bold", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Ground Truth Label")
    plt.tight_layout()
    plt.savefig(CM_DIR / filename, dpi=300)
    plt.close()

def main():
    print("=== Generating & Exporting Phase 7 Extensions Confusion Matrices ===", flush=True)
    
    data_a = np.load(FEATURES_DIR / "phagehostlearn" / "dataset_A_features.npz")
    data_b_kg = np.load(FEATURES_DIR / "deeppbi_kg" / "dataset_B_features.npz")
    data_b_phl = np.load(FEATURES_DIR / "phagehostlearn" / "dataset_B_features.npz")
    
    pairs_b = data_b_phl["pairs"]
    hosts_b = np.array([p.split("::")[1] for p in pairs_b])
    unique_hosts = sorted(list(set(hosts_b)))
    
    metrics_summary = []
    
    # -----------------------------------------------------------------
    # 1. Leave-One-Subject-Out CV (LOSO CV) for DeepPBI-KG
    # -----------------------------------------------------------------
    print("Evaluating DeepPBI-KG LOSO CV across 30 host strain subjects...", flush=True)
    loso_y_true_kg = []
    loso_y_pred_kg = []
    
    for subject_host in unique_hosts:
        test_mask = (hosts_b == subject_host)
        train_mask = ~test_mask
        
        X_tr = torch.tensor(data_b_kg["X"][train_mask], dtype=torch.float32)
        y_tr = torch.tensor(data_b_kg["y"][train_mask], dtype=torch.float32).unsqueeze(1)
        X_te = torch.tensor(data_b_kg["X"][test_mask], dtype=torch.float32)
        y_te = data_b_kg["y"][test_mask]
        
        model = DeepPBIKG_Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        model.train()
        for epoch in range(15):
            optimizer.zero_grad()
            out = model(X_tr)
            loss = criterion(out, y_tr)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            preds = model(X_te).numpy().flatten()
            
        loso_y_true_kg.extend(y_te)
        loso_y_pred_kg.extend(preds)
        
    m_kg_loso, cm_kg_loso = compute_all_metrics(np.array(loso_y_true_kg), np.array(loso_y_pred_kg))
    m_kg_loso["Model"] = "DeepPBI-KG"
    m_kg_loso["Condition"] = "Phase7_LOSO_CV"
    metrics_summary.append(m_kg_loso)
    plot_and_save_cm(cm_kg_loso, "DeepPBI-KG Confusion Matrix (Phase 7 LOSO CV)", "cm_phase7_deeppbi_kg_loso_cv.png")
    print(f"  DeepPBI-KG LOSO CV CM: TP={cm_kg_loso[1,1]}, TN={cm_kg_loso[0,0]}, FP={cm_kg_loso[0,1]}, FN={cm_kg_loso[1,0]}", flush=True)

    # -----------------------------------------------------------------
    # 2. Leave-One-Subject-Out CV (LOSO CV) for PhageHostLearn
    # -----------------------------------------------------------------
    print("Evaluating PhageHostLearn LOSO CV across 30 host strain subjects...", flush=True)
    loso_y_true_phl = []
    loso_y_pred_phl = []
    
    for subject_host in unique_hosts:
        test_mask = (hosts_b == subject_host)
        train_mask = ~test_mask
        
        X_tr, y_tr = data_b_phl["X"][train_mask], data_b_phl["y"][train_mask]
        X_te, y_te = data_b_phl["X"][test_mask], data_b_phl["y"][test_mask]
        
        clf = xgb.XGBClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=1)
        clf.fit(X_tr, y_tr)
        preds = clf.predict_proba(X_te)[:, 1]
        
        loso_y_true_phl.extend(y_te)
        loso_y_pred_phl.extend(preds)
        
    m_phl_loso, cm_phl_loso = compute_all_metrics(np.array(loso_y_true_phl), np.array(loso_y_pred_phl))
    m_phl_loso["Model"] = "PhageHostLearn"
    m_phl_loso["Condition"] = "Phase7_LOSO_CV"
    metrics_summary.append(m_phl_loso)
    plot_and_save_cm(cm_phl_loso, "PhageHostLearn Confusion Matrix (Phase 7 LOSO CV)", "cm_phase7_phagehostlearn_loso_cv.png")
    print(f"  PhageHostLearn LOSO CV CM: TP={cm_phl_loso[1,1]}, TN={cm_phl_loso[0,0]}, FP={cm_phl_loso[0,1]}, FN={cm_phl_loso[1,0]}", flush=True)

    # -----------------------------------------------------------------
    # 3. Reverse Domain Transfer (Pre-train on B -> Fine-tune on A)
    # -----------------------------------------------------------------
    print("Evaluating Reverse Domain Transfer (Pre-train B -> Fine-tune A)...", flush=True)
    # Train pre-trained model on Dataset B
    clf_b = xgb.XGBClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=1)
    clf_b.fit(data_b_phl["X"], data_b_phl["y"])
    
    # Split Dataset A into 80% Train, 20% Test for fine-tuning evaluation
    n_a = len(data_a["y"])
    np.random.seed(42)
    shuffled_idx = np.random.permutation(n_a)
    split_a = int(0.8 * n_a)
    tr_a_idx, te_a_idx = shuffled_idx[:split_a], shuffled_idx[split_a:]
    
    dtrain_a = xgb.DMatrix(data_a["X"][tr_a_idx], label=data_a["y"][tr_a_idx])
    dtest_a = xgb.DMatrix(data_a["X"][te_a_idx])
    
    # Fine-tune with incremental booster
    booster_b = clf_b.get_booster()
    params = {"max_depth": 6, "eta": 0.1, "objective": "binary:logistic", "eval_metric": "logloss", "nthread": 1}
    booster_reverse = xgb.train(params, dtrain_a, num_boost_round=30, xgb_model=booster_b)
    preds_reverse = booster_reverse.predict(dtest_a)
    
    m_reverse, cm_reverse = compute_all_metrics(data_a["y"][te_a_idx], preds_reverse)
    m_reverse["Model"] = "PhageHostLearn"
    m_reverse["Condition"] = "Phase7_ReverseTransfer_B_to_A"
    metrics_summary.append(m_reverse)
    plot_and_save_cm(cm_reverse, "PhageHostLearn Confusion Matrix (Reverse Transfer B -> A)", "cm_phase7_reverse_transfer_b_to_a.png")
    print(f"  Reverse Transfer (B->A) CM: TP={cm_reverse[1,1]}, TN={cm_reverse[0,0]}, FP={cm_reverse[0,1]}, FN={cm_reverse[1,0]}", flush=True)

    # Export metrics dataframe
    df_metrics = pd.DataFrame(metrics_summary)
    cols = ["Model", "Condition", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1_Score", "MCC", "ROC_AUC", "PR_AUC"]
    df_metrics = df_metrics[cols]
    
    csv_out = CM_DIR / "phase7_confusion_matrix_metrics_summary.csv"
    df_metrics.to_csv(csv_out, index=False)
    print(f"\nSaved Phase 7 metrics summary to {csv_out}", flush=True)
    print(df_metrics.to_string(), flush=True)

if __name__ == "__main__":
    main()
