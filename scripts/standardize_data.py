#!/usr/bin/env python3
"""
Standardize PhageHostLearn (Dataset A) and KlebPhaCol (Dataset B) into unified interaction matrices & metadata.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DIR_A = DATA_DIR / "dataset_A_phagehostlearn"
DIR_B = DATA_DIR / "dataset_B_klebphacol"

def standardize_dataset_a():
    print("=== Standardizing Dataset A (PhageHostLearn) ===")
    interactions_file = DIR_A / "phage_host_interactions.csv"
    rbp_file = DIR_A / "RBPbase.csv"
    loci_file = DIR_A / "Locibase.json"

    # 1. Interactions
    df = pd.read_csv(interactions_file, index_col=0)
    # Fill NA with 0.0 (non-interacting)
    df_clean = df.fillna(0.0)
    df_clean.to_csv(DIR_A / "standardized_interactions.csv")
    print(f"  Standardized interactions matrix shape: {df_clean.shape} (Hosts x Phages)")

    # 2. Phage Metadata
    if rbp_file.exists():
        rbp_df = pd.read_csv(rbp_file)
        phages = df_clean.columns.tolist()
        phage_meta = pd.DataFrame({"phage_id": phages})
        # Merge RBP info if available
        if "phage" in rbp_df.columns:
            rbp_counts = rbp_df.groupby("phage").size().reset_index(name="num_rbps")
            phage_meta = phage_meta.merge(rbp_counts, left_on="phage_id", right_on="phage", how="left").fillna(0)
            if "phage" in phage_meta.columns:
                phage_meta.drop(columns=["phage"], inplace=True)
        phage_meta.to_csv(DIR_A / "phage_metadata.csv", index=False)
        print(f"  Saved phage metadata ({len(phage_meta)} phages)")

    # 3. Host Metadata
    if loci_file.exists():
        with open(loci_file, "r") as f:
            loci_data = json.load(f)
        hosts = df_clean.index.tolist()
        host_records = []
        for h in hosts:
            info = loci_data.get(h, [])
            if isinstance(info, dict):
                klocus = info.get("K_locus", info.get("klocus", "Unknown"))
                st = info.get("ST", "Unknown")
            elif isinstance(info, list):
                klocus = f"Loci_proteins_count_{len(info)}"
                st = "Unknown"
            else:
                klocus = "Unknown"
                st = "Unknown"
            host_records.append({"host_id": h, "K_locus": klocus, "ST": st})
        host_meta = pd.DataFrame(host_records)
        host_meta.to_csv(DIR_A / "host_metadata.csv", index=False)
        print(f"  Saved host metadata ({len(host_meta)} hosts)")

def standardize_dataset_b():
    print("=== Standardizing Dataset B (KlebPhaCol) ===")
    loci_invitro = DIR_B / "Locibase_invitro.json"
    
    if loci_invitro.exists():
        with open(loci_invitro, "r") as f:
            data = json.load(f)
        hosts = list(data.keys())
        host_records = []
        for h in hosts:
            info = data[h]
            if isinstance(info, dict):
                klocus = info.get("K_locus", info.get("klocus", "Unknown"))
            elif isinstance(info, list):
                klocus = f"Loci_proteins_count_{len(info)}"
            else:
                klocus = "Unknown"
            host_records.append({"host_id": h, "K_locus": klocus})
        host_meta = pd.DataFrame(host_records)
        host_meta.to_csv(DIR_B / "host_metadata.csv", index=False)
        print(f"  Saved Dataset B host metadata ({len(host_meta)} hosts)")

if __name__ == "__main__":
    standardize_dataset_a()
    standardize_dataset_b()
    print("Data standardization complete.")
