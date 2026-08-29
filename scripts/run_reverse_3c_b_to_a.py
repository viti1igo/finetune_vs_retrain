#!/usr/bin/env python3
"""Reproducible reverse Method 3C experiment: KlebPhaCol (B) -> PhageHostLearn (A).

ESM-2 embeddings are inputs only.  The source-domain Ridge compatibility model is
fit on B, then its decision score is appended to A features for a newly trained
target-domain XGBoost head.  Dataset A evaluation is host-disjoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_DIR = BASE_DIR / "features" / "phagehostlearn"
DATA_DIR = BASE_DIR / "data" / "reverse_3c_b_to_a"
RESULT_DIR = BASE_DIR / "results" / "reverse_3c_b_to_a"
MODEL_DIR = BASE_DIR / "models" / "phagehostlearn" / "reverse_3c_b_to_a"
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "reverse_3c_b_to_a_results.ipynb"
DOC_PATH = BASE_DIR / "doc" / "reverse_3c_b_to_a.md"


def host_from_pair(pair: str) -> str:
    return pair.split("::", 1)[1]


def make_host_split(pairs: np.ndarray, y: np.ndarray, seed: int, test_fraction: float = 0.2):
    """Choose a deterministic host-disjoint split with prevalence close to full A."""
    hosts = np.array(sorted({host_from_pair(p) for p in pairs}))
    n_test = round(len(hosts) * test_fraction)
    all_prevalence = float(y.mean())
    rng = np.random.default_rng(seed)
    best = None
    # Candidate search maintains host grouping while avoiding an accidentally
    # all-negative test cohort in this extremely imbalanced data set.
    for _ in range(2000):
        test_hosts = np.sort(rng.choice(hosts, size=n_test, replace=False))
        test_mask = np.isin([host_from_pair(p) for p in pairs], test_hosts)
        if y[test_mask].min() == y[test_mask].max() or y[~test_mask].min() == y[~test_mask].max():
            continue
        score = abs(float(y[test_mask].mean()) - all_prevalence)
        candidate = (score, test_hosts, test_mask)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not make a two-class host-disjoint Dataset A split")
    _, test_hosts, test_mask = best
    train_hosts = np.array(sorted(set(hosts) - set(test_hosts)))
    train_indices = np.flatnonzero(~test_mask)
    test_indices = np.flatnonzero(test_mask)
    assert not set(train_hosts).intersection(test_hosts)
    assert len(train_indices) + len(test_indices) == len(pairs)
    assert set(train_indices).isdisjoint(test_indices)
    return train_hosts, test_hosts, train_indices, test_indices


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, probability)
    return {
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
        "PR_AUC": float(auc(recall, precision)),
        "MCC": float(matthews_corrcoef(y_true, prediction)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Positive_predictions": int(prediction.sum()),
    }


def classifier(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
        random_state=seed, n_jobs=1,
    )


def save_figures(y: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    names = {"A_from_scratch": "A from scratch", "B_zero_shot": "B source-only zero shot", "B_to_A_reverse_3C": "B to A reverse 3C"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, p in probabilities.items():
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{names[key]} (AUC={roc_auc_score(y, p):.3f})")
        axes[1].plot(recall, precision, label=f"{names[key]} (AP={auc(recall, precision):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Dataset A held-out hosts: ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Dataset A held-out hosts: precision-recall")
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULT_DIR / "roc_pr_curves.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, p) in zip(axes, probabilities.items()):
        cm = confusion_matrix(y, p >= 0.5)
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm): ax.text(j, i, str(value), ha="center", va="center")
        ax.set(title=names[key], xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="True")
    fig.tight_layout(); fig.savefig(RESULT_DIR / "confusion_matrices.png", dpi=180); plt.close(fig)


def write_notebook() -> None:
    """Create a dependency-light executable notebook that reads saved outputs."""
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Reverse Method 3C results: KlebPhaCol → PhageHostLearn\n", "This notebook reads the persisted artifacts from `results/reverse_3c_b_to_a`; it contains no hard-coded metrics."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Image\n\nROOT = Path.cwd().resolve()\nif not (ROOT / 'results').exists():\n    ROOT = ROOT.parent\nRESULTS = ROOT / 'results' / 'reverse_3c_b_to_a'\nDATA = ROOT / 'data' / 'reverse_3c_b_to_a'\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["manifest = json.loads((DATA / 'dataset_A_host_split_manifest.json').read_text())\nmetrics = pd.read_csv(RESULTS / 'metrics.csv', index_col='Model')\ndisplay(metrics)\nprint(json.dumps(manifest['leakage_checks'], indent=2))\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["predictions = pd.read_csv(RESULTS / 'held_out_predictions.csv')\nprint(predictions.groupby('host')['label'].agg(['count', 'sum']).head())\ndisplay(Image(filename=str(RESULTS / 'roc_pr_curves.png')))\ndisplay(Image(filename=str(RESULTS / 'confusion_matrices.png')))\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## Interpretation\nCompare reverse 3C against both same-target training and source-only zero shot using ROC-AUC, PR-AUC, MCC, F1, and accuracy. Consult the accompanying report for data-quality limitations."]},
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "phage_finetune", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n")


def write_report(manifest: dict, result: pd.DataFrame) -> None:
    headers = ["Model", *result.columns.tolist()]
    markdown_rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for model, values in result.iterrows():
        formatted = [model, *[f"{value:.6f}" if isinstance(value, float) else str(value) for value in values]]
        markdown_rows.append("| " + " | ".join(formatted) + " |")
    rows = "\n".join(markdown_rows)
    DOC_PATH.write_text(f"""# Reverse Method 3C: KlebPhaCol → PhageHostLearn

## Purpose

This reproducible Stage 7 experiment reverses Method 3C. Frozen ESM-2 pair embeddings are used as fixed inputs; **ESM-2 is not trained or fine-tuned**. A `StandardScaler + RidgeClassifier` compatibility encoder is fit on the full KlebPhaCol source (Dataset B), its continuous decision score is appended to Dataset A features, and a new XGBoost head is trained only on Dataset A training hosts.

## Dataset A host-disjoint evaluation

- Seed: `{manifest['seed']}`
- Dataset A pairs: {manifest['pair_counts']['total']}; hosts: {manifest['host_counts']['total']}
- Train: {manifest['pair_counts']['train']} pairs across {manifest['host_counts']['train']} hosts
- Held out: {manifest['pair_counts']['test']} pairs across {manifest['host_counts']['test']} unseen hosts
- Leakage checks: {json.dumps(manifest['leakage_checks'])}
- Input dimension: 2,560; reverse-3C augmented dimension: 2,561.

## Held-out Dataset A results

{rows}

The notebook at `notebooks/reverse_3c_b_to_a_results.ipynb` loads these saved results and figures. Prediction-level values are in `results/reverse_3c_b_to_a/held_out_predictions.csv`.

## Comparisons

1. **A from scratch** trains XGBoost on Dataset A training hosts.
2. **B source-only zero shot** trains XGBoost on Dataset B and scores held-out Dataset A hosts without adaptation.
3. **B → A reverse 3C** uses the Dataset B Ridge compatibility score plus a target-specific XGBoost head trained on Dataset A training hosts.

## Limitation requiring resolution

The current checked-in feature generator creates Dataset B labels with `np.random.choice(...)` rather than loading observed KlebPhaCol interactions. Consequently, this run verifies pipeline reproducibility and leakage isolation, but cannot support a biological claim about reverse transfer until Dataset B is regenerated from curated experimental labels. This report supersedes the inconsistent legacy Stage 7 reverse-transfer summaries; only artifacts produced by this runner are reported here.
""")


def main(seed: int) -> None:
    for path in (DATA_DIR, RESULT_DIR, MODEL_DIR): path.mkdir(parents=True, exist_ok=True)
    data_a = np.load(FEATURE_DIR / "dataset_A_features.npz")
    data_b = np.load(FEATURE_DIR / "dataset_B_features.npz")
    X_a, y_a, pairs_a = data_a["X"], data_a["y"], data_a["pairs"]
    X_b, y_b = data_b["X"], data_b["y"]
    if X_a.shape[1] != 2560 or X_b.shape[1] != 2560: raise ValueError("Expected 2,560-dimensional ESM-2 pair features")
    train_hosts, test_hosts, train_idx, test_idx = make_host_split(pairs_a, y_a, seed)
    X_train, X_test, y_train, y_test = X_a[train_idx], X_a[test_idx], y_a[train_idx], y_a[test_idx]
    manifest = {"seed": seed, "method": "prevalence-matched host-disjoint 80/20 candidate search (2,000 candidates)", "source_dataset": "KlebPhaCol (B; full feature set)", "target_dataset": "PhageHostLearn (A)", "host_counts": {"total": int(len(train_hosts) + len(test_hosts)), "train": int(len(train_hosts)), "test": int(len(test_hosts))}, "pair_counts": {"total": int(len(y_a)), "train": int(len(train_idx)), "test": int(len(test_idx))}, "class_counts": {"train_positive": int(y_train.sum()), "test_positive": int(y_test.sum())}, "train_hosts": train_hosts.tolist(), "test_hosts": test_hosts.tolist(), "train_indices": train_idx.tolist(), "test_indices": test_idx.tolist(), "leakage_checks": {"host_sets_disjoint": bool(not set(train_hosts).intersection(test_hosts)), "all_pairs_assigned_once": bool(len(train_idx) + len(test_idx) == len(y_a)), "both_target_partitions_have_two_classes": bool(y_train.min() != y_train.max() and y_test.min() != y_test.max())}}
    (DATA_DIR / "dataset_A_host_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    source_only = classifier(seed).fit(X_b, y_b)
    scratch = classifier(seed).fit(X_train, y_train)
    encoder = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0)).fit(X_b, y_b)
    transfer_train = np.column_stack([X_train, encoder.decision_function(X_train)])
    transfer_test = np.column_stack([X_test, encoder.decision_function(X_test)])
    assert transfer_train.shape[1] == 2561 and transfer_test.shape[1] == 2561
    reverse_3c = classifier(seed).fit(transfer_train, y_train)
    probabilities = {"A_from_scratch": scratch.predict_proba(X_test)[:, 1], "B_zero_shot": source_only.predict_proba(X_test)[:, 1], "B_to_A_reverse_3C": reverse_3c.predict_proba(transfer_test)[:, 1]}
    result = pd.DataFrame({key: metrics(y_test, value) for key, value in probabilities.items()}).T
    result.index.name = "Model"; result.to_csv(RESULT_DIR / "metrics.csv")
    predictions = pd.DataFrame({"pair_index": test_idx, "pair": pairs_a[test_idx], "host": [host_from_pair(p) for p in pairs_a[test_idx]], "label": y_test})
    for key, value in probabilities.items(): predictions[f"probability_{key}"] = value
    predictions.to_csv(RESULT_DIR / "held_out_predictions.csv", index=False)
    joblib.dump(encoder, MODEL_DIR / "source_b_compatibility_encoder.joblib")
    source_only.save_model(MODEL_DIR / "source_b_zero_shot_xgboost.json")
    scratch.save_model(MODEL_DIR / "target_a_from_scratch_xgboost.json")
    reverse_3c.save_model(MODEL_DIR / "target_a_reverse_3c_xgboost.json")
    save_figures(y_test, probabilities); write_notebook(); write_report(manifest, result)
    print(result.to_string()); print(f"Artifacts written to {RESULT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
