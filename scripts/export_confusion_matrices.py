#!/usr/bin/env python3
"""
Generate and Export Confusion Matrices & Classification Metrics.
Evaluates all conditions (C1, C2, C3 Option A, C3 Option C) for DeepPBI-KG and PhageHostLearn.
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
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    print("=== Generating & Exporting Confusion Matrices and Evaluation Metrics ===", flush=True)
    
    data_a_phl = np.load(FEATURES_DIR / "phagehostlearn" / "dataset_A_features.npz")
    data_b_kg = np.load(FEATURES_DIR / "deeppbi_kg" / "dataset_B_features.npz")
    data_b_phl = np.load(FEATURES_DIR / "phagehostlearn" / "dataset_B_features.npz")
    
    with open(BASE_DIR / "data" / "klebphacol_train_indices.json", "r") as f:
        train_idx = json.load(f)
    with open(BASE_DIR / "data" / "klebphacol_test_indices.json", "r") as f:
        test_idx = json.load(f)
        
    X_test_kg = torch.tensor(data_b_kg["X"][test_idx], dtype=torch.float32)
    y_test_kg = data_b_kg["y"][test_idx]
    
    X_train_phl = data_b_phl["X"][train_idx]
    y_train_phl = data_b_phl["y"][train_idx]
    X_test_phl = data_b_phl["X"][test_idx]
    y_test_phl = data_b_phl["y"][test_idx]
    
    metrics_summary = []
    
    # 1. DeepPBI-KG Models
    kg_conditions = [
        ("C1_Train_B", MODELS_DIR / "deeppbi_kg" / "C1_train_B" / "model.pt"),
        ("C2_Train_A", MODELS_DIR / "deeppbi_kg" / "C2_train_A" / "model.pt"),
        ("C3_Finetune", MODELS_DIR / "deeppbi_kg" / "C3_finetune" / "model.pt")
    ]
    
    for cond_name, model_path in kg_conditions:
        if model_path.exists():
            model = DeepPBIKG_Model()
            model.load_state_dict(torch.load(str(model_path)))
            model.eval()
            with torch.no_grad():
                preds_prob = model(X_test_kg).numpy().flatten()
            
            metrics, cm = compute_all_metrics(y_test_kg, preds_prob)
            metrics["Model"] = "DeepPBI-KG"
            metrics["Condition"] = cond_name
            metrics_summary.append(metrics)
            
            title = f"DeepPBI-KG Confusion Matrix ({cond_name})"
            filename = f"cm_deeppbi_kg_{cond_name.lower()}.png"
            plot_and_save_cm(cm, title, filename)
            print(f"  Exported DeepPBI-KG {cond_name} Confusion Matrix: TP={cm[1,1]}, TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}", flush=True)

    # 2. PhageHostLearn Models
    # C1: Train from scratch on B
    model_c1 = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
    model_c1.fit(X_train_phl, y_train_phl)
    preds_c1 = model_c1.predict_proba(X_test_phl)[:, 1]
    m_c1, cm_c1 = compute_all_metrics(y_test_phl, preds_c1)
    m_c1["Model"] = "PhageHostLearn"
    m_c1["Condition"] = "C1_Train_B"
    metrics_summary.append(m_c1)
    plot_and_save_cm(cm_c1, "PhageHostLearn Confusion Matrix (C1_Train_B)", "cm_phagehostlearn_c1_train_b.png")
    print(f"  Exported PhageHostLearn C1_Train_B Confusion Matrix: TP={cm_c1[1,1]}, TN={cm_c1[0,0]}, FP={cm_c1[0,1]}, FN={cm_c1[1,0]}", flush=True)

    # C2: Train on A, Test on B
    model_c2 = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
    model_c2.fit(data_a_phl["X"], data_a_phl["y"])
    preds_c2 = model_c2.predict_proba(X_test_phl)[:, 1]
    m_c2, cm_c2 = compute_all_metrics(y_test_phl, preds_c2)
    m_c2["Model"] = "PhageHostLearn"
    m_c2["Condition"] = "C2_Train_A"
    metrics_summary.append(m_c2)
    plot_and_save_cm(cm_c2, "PhageHostLearn Confusion Matrix (C2_Train_A)", "cm_phagehostlearn_c2_train_a.png")
    print(f"  Exported PhageHostLearn C2_Train_A Confusion Matrix: TP={cm_c2[1,1]}, TN={cm_c2[0,0]}, FP={cm_c2[0,1]}, FN={cm_c2[1,0]}", flush=True)

    # C3 Option A: Incremental Boosting
    dtrain_b = xgb.DMatrix(X_train_phl, label=y_train_phl)
    dtest_b = xgb.DMatrix(X_test_phl)
    dtrain_a = xgb.DMatrix(data_a_phl["X"], label=data_a_phl["y"])
    
    params = {"max_depth": 6, "eta": 0.1, "objective": "binary:logistic", "eval_metric": "logloss", "nthread": 1}
    booster_a = xgb.train(params, dtrain_a, num_boost_round=50)
    booster_c3a = xgb.train(params, dtrain_b, num_boost_round=50, xgb_model=booster_a)
    preds_c3a = booster_c3a.predict(dtest_b)
    m_c3a, cm_c3a = compute_all_metrics(y_test_phl, preds_c3a)
    m_c3a["Model"] = "PhageHostLearn"
    m_c3a["Condition"] = "C3_OptionA_IncrementalBoosting"
    metrics_summary.append(m_c3a)
    plot_and_save_cm(cm_c3a, "PhageHostLearn Confusion Matrix (C3_OptionA_IncrementalBoosting)", "cm_phagehostlearn_c3_optiona_incrementalboosting.png")
    print(f"  Exported PhageHostLearn C3_OptionA Confusion Matrix: TP={cm_c3a[1,1]}, TN={cm_c3a[0,0]}, FP={cm_c3a[0,1]}, FN={cm_c3a[1,0]}", flush=True)

    # C3 Option C: Feature Transfer
    feat_encoder = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    feat_encoder.fit(data_a_phl["X"], data_a_phl["y"])
    
    train_transfer_feats = np.column_stack([X_train_phl, feat_encoder.decision_function(X_train_phl)])
    test_transfer_feats = np.column_stack([X_test_phl, feat_encoder.decision_function(X_test_phl)])
    
    model_c3c = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
    model_c3c.fit(train_transfer_feats, y_train_phl)
    preds_c3c = model_c3c.predict_proba(test_transfer_feats)[:, 1]
    m_c3c, cm_c3c = compute_all_metrics(y_test_phl, preds_c3c)
    m_c3c["Model"] = "PhageHostLearn"
    m_c3c["Condition"] = "C3_OptionC_FeatureTransfer"
    metrics_summary.append(m_c3c)
    plot_and_save_cm(cm_c3c, "PhageHostLearn Confusion Matrix (C3_OptionC_FeatureTransfer)", "cm_phagehostlearn_c3_optionc_featuretransfer.png")
    print(f"  Exported PhageHostLearn C3_OptionC Confusion Matrix: TP={cm_c3c[1,1]}, TN={cm_c3c[0,0]}, FP={cm_c3c[0,1]}, FN={cm_c3c[1,0]}", flush=True)

    df_metrics = pd.DataFrame(metrics_summary)
    cols = ["Model", "Condition", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1_Score", "MCC", "ROC_AUC", "PR_AUC"]
    df_metrics = df_metrics[cols]
    
    csv_out = CM_DIR / "confusion_matrix_metrics_summary.csv"
    df_metrics.to_csv(csv_out, index=False)
    print(f"\nSaved metrics summary to {csv_out}", flush=True)
    print(df_metrics.to_string(), flush=True)

if __name__ == "__main__":
    main()
