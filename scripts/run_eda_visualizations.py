#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) & Visualization Script for Datasets A & B.
Generates comprehensive charts analyzing schema, class imbalance, host range,
K-locus diversity, ESM-2 embedding projections, and train/test split distributions.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIR_A = DATA_DIR / "dataset_A_phagehostlearn"
DIR_B = DATA_DIR / "dataset_B_klebphacol"

# Styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def run_eda():
    print("=== Generating EDA Visualizations for Dataset A & Dataset B ===")
    
    # -------------------------------------------------------------
    # 1. Class Imbalance Comparison
    # -------------------------------------------------------------
    df_inter_a = pd.read_csv(DIR_A / "standardized_interactions.csv", index_col=0)
    vals_a = df_inter_a.values.flatten()
    vals_a = vals_a[~np.isnan(vals_a)]
    
    data_b_feat = np.load(BASE_DIR / "features" / "phagehostlearn" / "dataset_B_features.npz")
    y_b = data_b_feat["y"]
    
    pos_a, neg_a = int((vals_a > 0).sum()), int((vals_a == 0).sum())
    pos_b, neg_b = int((y_b > 0).sum()), int((y_b == 0).sum())
    
    df_imbalance = pd.DataFrame([
        {"Dataset": "Dataset A (PhageHostLearn)", "Class": "Positive (Infection)", "Count": pos_a, "Percentage": pos_a/len(vals_a)*100},
        {"Dataset": "Dataset A (PhageHostLearn)", "Class": "Negative (No Infection)", "Count": neg_a, "Percentage": neg_a/len(vals_a)*100},
        {"Dataset": "Dataset B (KlebPhaCol)", "Class": "Positive (Infection)", "Count": pos_b, "Percentage": pos_b/len(y_b)*100},
        {"Dataset": "Dataset B (KlebPhaCol)", "Class": "Negative (No Infection)", "Count": neg_b, "Percentage": neg_b/len(y_b)*100},
    ])
    
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_imbalance, x="Dataset", y="Percentage", hue="Class", palette="Set2")
    plt.title("Class Imbalance Comparison: Dataset A vs Dataset B", fontsize=13, fontweight="bold")
    plt.ylabel("Percentage of Total Pairs (%)")
    plt.ylim(0, 110)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 2),
                        ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_class_imbalance.png", dpi=300)
    plt.close()
    print("  Saved eda_class_imbalance.png")
    
    # -------------------------------------------------------------
    # 2. Phage Host Range & Host Susceptibility Distributions
    # -------------------------------------------------------------
    phage_host_range = (df_inter_a > 0).sum(axis=0) # sum per phage
    host_susceptibility = (df_inter_a > 0).sum(axis=1) # sum per host
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(phage_host_range, kde=True, ax=axes[0], color="teal", bins=15)
    axes[0].set_title("Phage Host Range Distribution (Infected Hosts / Phage)", fontweight="bold")
    axes[0].set_xlabel("Number of Susceptible Host Strains")
    axes[0].set_ylabel("Phage Count")
    
    sns.histplot(host_susceptibility, kde=True, ax=axes[1], color="coral", bins=15)
    axes[1].set_title("Host Susceptibility Distribution (Infecting Phages / Host)", fontweight="bold")
    axes[1].set_xlabel("Number of Infecting Phages")
    axes[1].set_ylabel("Host Strain Count")
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_phage_host_distributions.png", dpi=300)
    plt.close()
    print("  Saved eda_phage_host_distributions.png")
    
    # -------------------------------------------------------------
    # 3. K-Locus Diversity Distribution
    # -------------------------------------------------------------
    host_meta_a = pd.read_csv(DIR_A / "host_metadata.csv")
    plt.figure(figsize=(10, 5))
    top_kloci = host_meta_a["K_locus"].value_counts().head(10).reset_index()
    top_kloci.columns = ["K_locus", "Count"]
    
    sns.barplot(data=top_kloci, x="K_locus", y="Count", palette="viridis")
    plt.title("Top K-Locus Types Distribution in Dataset A", fontsize=13, fontweight="bold")
    plt.xlabel("K-Locus Type")
    plt.ylabel("Number of Strains")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_klocus_diversity.png", dpi=300)
    plt.close()
    print("  Saved eda_klocus_diversity.png")
    
    # -------------------------------------------------------------
    # 4. ESM-2 Protein Embedding PCA Space Projection
    # -------------------------------------------------------------
    df_loci_a = pd.read_csv(DIR_A / "esm2_embeddings_loci.csv")
    feat_cols = [c for c in df_loci_a.columns if c != "accession"]
    X_loci = df_loci_a[feat_cols].values
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_loci)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c="dodgerblue", alpha=0.7, edgecolors="k", s=50)
    plt.title(f"ESM-2 K-Locus Embedding Space (PCA Projection)\nExplained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%", fontweight="bold")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_esm2_embedding_pca.png", dpi=300)
    plt.close()
    print("  Saved eda_esm2_embedding_pca.png")
    
    # -------------------------------------------------------------
    # 5. Train vs Test Split Class Distribution (Dataset B)
    # -------------------------------------------------------------
    with open(BASE_DIR / "data" / "klebphacol_train_indices.json") as f:
        tr_idx = json.load(f)
    with open(BASE_DIR / "data" / "klebphacol_test_indices.json") as f:
        te_idx = json.load(f)
        
    y_tr = y_b[tr_idx]
    y_te = y_b[te_idx]
    
    df_split = pd.DataFrame([
        {"Split": "Train (80%)", "Interaction": "Positive", "Count": int((y_tr > 0).sum())},
        {"Split": "Train (80%)", "Interaction": "Negative", "Count": int((y_tr == 0).sum())},
        {"Split": "Test (20%)", "Interaction": "Positive", "Count": int((y_te > 0).sum())},
        {"Split": "Test (20%)", "Interaction": "Negative", "Count": int((y_te == 0).sum())},
    ])
    
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=df_split, x="Split", y="Count", hue="Interaction", palette="Accent")
    plt.title("Dataset B Train vs Test Split Interaction Distribution", fontweight="bold")
    plt.ylabel("Number of Pairwise Interactions")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2., h + 15),
                        ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_train_test_split.png", dpi=300)
    plt.close()
    print("  Saved eda_train_test_split.png")

if __name__ == "__main__":
    run_eda()
    print("EDA Visualizations generation complete.")
