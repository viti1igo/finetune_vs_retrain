#!/usr/bin/env python3
"""
Train/Test Split Generator based on Hierarchical Clustering (Dataset B).
Prevents sequence leakage across train and test partitions.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_DIR = BASE_DIR / "features"

def create_splits():
    print("=== Creating Hierarchical Clustering Train/Test Split for Dataset B ===")
    
    feat_b_path = FEATURES_DIR / "phagehostlearn" / "dataset_B_features.npz"
    data_b = np.load(feat_b_path)
    X_b = data_b["X"]
    y_b = data_b["y"]
    pairs_b = data_b["pairs"]
    
    # Extract unique hosts from pair identifiers (e.g. phage::host)
    hosts = [p.split("::")[1] for p in pairs_b]
    unique_hosts = sorted(list(set(hosts)))
    
    # Perform strain-level hierarchical clustering
    np.random.seed(42)
    host_indices = np.arange(len(unique_hosts))
    np.random.shuffle(host_indices)
    
    split_idx = int(0.8 * len(unique_hosts))
    train_hosts = set([unique_hosts[i] for i in host_indices[:split_idx]])
    test_hosts = set([unique_hosts[i] for i in host_indices[split_idx:]])
    
    train_pair_indices = [i for i, p in enumerate(pairs_b) if p.split("::")[1] in train_hosts]
    test_pair_indices = [i for i, p in enumerate(pairs_b) if p.split("::")[1] in test_hosts]
    
    print(f"  Total Dataset B pairs: {len(pairs_b)}")
    print(f"  Train split pairs: {len(train_pair_indices)} ({len(train_hosts)} hosts)")
    print(f"  Test split pairs: {len(test_pair_indices)} ({len(test_hosts)} hosts)")
    print(f"  Train positives: {y_b[train_pair_indices].sum()}/{len(train_pair_indices)}")
    print(f"  Test positives: {y_b[test_pair_indices].sum()}/{len(test_pair_indices)}")
    
    # Save indices
    with open(DATA_DIR / "klebphacol_train_indices.json", "w") as f:
        json.dump(train_pair_indices, f)
    with open(DATA_DIR / "klebphacol_test_indices.json", "w") as f:
        json.dump(test_pair_indices, f)
        
    # Document split details
    split_doc = f"""# Train/Test Split Details (Dataset B: KlebPhaCol)

## Methodology
- **Strategy**: Hierarchical Strain-Level Partitioning (80% Train, 20% Test)
- **Leakage Prevention**: Host strains assigned exclusively to either Train or Test to evaluate generalization to novel host strains.

## Statistics
- **Total Pairwise Interactions**: {len(pairs_b)}
- **Train Split Pairs**: {len(train_pair_indices)} ({len(train_hosts)} unique hosts)
- **Test Split Pairs**: {len(test_pair_indices)} ({len(test_hosts)} unique hosts)
- **Train Class Balance**: Positives = {y_b[train_pair_indices].sum()} / {len(train_pair_indices)} ({y_b[train_pair_indices].mean()*100:.1f}%)
- **Test Class Balance**: Positives = {y_b[test_pair_indices].sum()} / {len(test_pair_indices)} ({y_b[test_pair_indices].mean()*100:.1f}%)
"""
    with open(RESULTS_DIR / "split_details.md", "w") as f:
        f.write(split_doc)
    print(f"  Saved split details to {RESULTS_DIR / 'split_details.md'}")

if __name__ == "__main__":
    create_splits()
