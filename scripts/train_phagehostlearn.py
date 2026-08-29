#!/usr/bin/env python3
"""
PhageHostLearn Experimental Protocol (C1, C2, C3)
XGBoost Classifier with Incremental Fine-Tuning for Phage-Host Interaction Prediction.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, f1_score, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "features" / "phagehostlearn"
MODELS_DIR = BASE_DIR / "models" / "phagehostlearn"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "C1_train_B").mkdir(exist_ok=True)
(MODELS_DIR / "C2_train_A").mkdir(exist_ok=True)
(MODELS_DIR / "C3_finetune").mkdir(exist_ok=True)

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

def run_phagehostlearn_experiments():
    print("=== Running PhageHostLearn (XGBoost) Experiments (C1, C2, C3) ===")
    
    # Load Dataset A & B features
    data_a = np.load(FEATURES_DIR / "dataset_A_features.npz")
    data_b = np.load(FEATURES_DIR / "dataset_B_features.npz")
    
    X_a, y_a = data_a["X"], data_a["y"]
    X_b, y_b = data_b["X"], data_b["y"]
    
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
    model_c1 = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
    )
    model_c1.fit(X_b_train, y_b_train)
    preds_c1 = model_c1.predict_proba(X_b_test)[:, 1]
    metrics_c1 = calculate_metrics(y_b_test, preds_c1)
    results["C1_Train_B"] = metrics_c1
    model_c1.save_model(MODELS_DIR / "C1_train_B" / "model.json")
    print(f"  C1 Test Results: ROC_AUC={metrics_c1['ROC_AUC']:.4f}, PR_AUC={metrics_c1['PR_AUC']:.4f}, MCC={metrics_c1['MCC']:.4f}")
    
    # -------------------------------------------------------------
    # C2: Train on A, Test on B
    # -------------------------------------------------------------
    print("\n--- Condition C2: Train on A, Test on B ---")
    model_c2 = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
    )
    model_c2.fit(X_a, y_a)
    preds_c2 = model_c2.predict_proba(X_b_test)[:, 1]
    metrics_c2 = calculate_metrics(y_b_test, preds_c2)
    results["C2_Train_A"] = metrics_c2
    model_c2.save_model(MODELS_DIR / "C2_train_A" / "model.json")
    print(f"  C2 Test Results: ROC_AUC={metrics_c2['ROC_AUC']:.4f}, PR_AUC={metrics_c2['PR_AUC']:.4f}, MCC={metrics_c2['MCC']:.4f}")

    # -------------------------------------------------------------
    # C3 Option A: Train on A, Fine-tune (Incremental Boosting) on B, Test on B
    # -------------------------------------------------------------
    print("\n--- Condition C3 Option A: Incremental Boosting Fine-Tune ---")
    dtrain_b = xgb.DMatrix(X_b_train, label=y_b_train)
    dtest_b = xgb.DMatrix(X_b_test, label=y_b_test)
    
    dtrain_a = xgb.DMatrix(X_a, label=y_a)
    params = {'max_depth': 6, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    booster_a = xgb.train(params, dtrain_a, num_boost_round=100)
    
    booster_c3a = xgb.train(params, dtrain_b, num_boost_round=50, xgb_model=booster_a)
    preds_c3a = booster_c3a.predict(dtest_b)
    metrics_c3a = calculate_metrics(y_b_test, preds_c3a)
    results["C3_OptionA_IncrementalBoosting"] = metrics_c3a
    booster_c3a.save_model(MODELS_DIR / "C3_finetune" / "model.json")
    print(f"  C3 Option A Results: ROC_AUC={metrics_c3a['ROC_AUC']:.4f}, PR_AUC={metrics_c3a['PR_AUC']:.4f}, Accuracy={metrics_c3a['Accuracy']:.4f}")

    # -------------------------------------------------------------
    # C3 Option C: Feature Transfer (Domain-Adapted ESM-2 Representation)
    # -------------------------------------------------------------
    print("\n--- Condition C3 Option C: Feature Transfer (Domain Adaptation) ---")
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    # 1. Feature transfer encoder pre-trained on A
    feat_encoder = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    feat_encoder.fit(X_a, y_a)
    
    # 2. Extract domain-aligned transfer features for B
    train_transfer_feats = np.column_stack([X_b_train, feat_encoder.decision_function(X_b_train)])
    test_transfer_feats = np.column_stack([X_b_test, feat_encoder.decision_function(X_b_test)])
    
    # 3. XGBoost head trained on transferred features
    model_c3c = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model_c3c.fit(train_transfer_feats, y_b_train)
    
    preds_c3c = model_c3c.predict_proba(test_transfer_feats)[:, 1]
    metrics_c3c = calculate_metrics(y_b_test, preds_c3c)
    results["C3_OptionC_FeatureTransfer"] = metrics_c3c
    model_c3c.save_model(MODELS_DIR / "C3_finetune" / "model_option_c.json")
    print(f"  C3 Option C Results: ROC_AUC={metrics_c3c['ROC_AUC']:.4f}, PR_AUC={metrics_c3c['PR_AUC']:.4f}, Accuracy={metrics_c3c['Accuracy']:.4f}")
    
    # Save overall PhageHostLearn results summary
    df_res = pd.DataFrame(results).T
    df_res.to_csv(RESULTS_DIR / "phagehostlearn_results.csv")
    print("\nPhageHostLearn Experiments Summary:")
    print(df_res)

if __name__ == "__main__":
    run_phagehostlearn_experiments()
