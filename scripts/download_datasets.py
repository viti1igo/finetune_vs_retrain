#!/usr/bin/env python3
"""
Dataset Downloader for PhageHostLearn (Dataset A) and KlebPhaCol (Dataset B).
"""

import os
import sys
import zipfile
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DIR_A = DATA_DIR / "dataset_A_phagehostlearn"
DIR_B = DATA_DIR / "dataset_B_klebphacol"

DIR_A.mkdir(parents=True, exist_ok=True)
DIR_B.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest_path.name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
    print(f"  Saved {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)")

def download_phagehostlearn_dataset():
    print("=== Downloading PhageHostLearn dataset (Dataset A) from Zenodo (11061100) ===")
    files = [
        "phage_host_interactions.csv",
        "RBPbase.csv",
        "Locibase.json",
        "esm2_embeddings_rbp.csv",
        "esm2_embeddings_loci.csv",
        "phages_genomes.zip",
        "klebsiella_genomes.zip"
    ]
    for filename in files:
        url = f"https://zenodo.org/api/records/11061100/files/{filename}/content"
        dest = DIR_A / filename
        if not dest.exists() or dest.stat().st_size == 0:
            try:
                download_file(url, dest)
                if filename.endswith(".zip"):
                    print(f"  Extracting {filename}...")
                    with zipfile.ZipFile(dest, "r") as zip_ref:
                        extract_dir = DIR_A / filename.replace(".zip", "")
                        zip_ref.extractall(extract_dir)
                    print(f"  Extracted to {extract_dir.name}")
            except Exception as e:
                print(f"  Failed downloading {filename}: {e}")
        else:
            print(f"  {filename} already exists.")

def setup_klebphacol_dataset():
    print("=== Setting up KlebPhaCol dataset (Dataset B) ===")
    # Using existing PhageHostLearn in vitro / KlebPhaCol validation subset
    files = [
        ("Locibase_invitro.json", "https://zenodo.org/api/records/11061100/files/Locibase_invitro.json/content"),
        ("esm2_embeddings_loci_invitro.csv", "https://zenodo.org/api/records/11061100/files/esm2_embeddings_loci_invitro.csv/content")
    ]
    for filename, url in files:
        dest = DIR_B / filename
        if not dest.exists() or dest.stat().st_size == 0:
            try:
                download_file(url, dest)
            except Exception as e:
                print(f"  Failed downloading {filename}: {e}")
        else:
            print(f"  {filename} already exists.")

if __name__ == "__main__":
    download_phagehostlearn_dataset()
    setup_klebphacol_dataset()
    print("\nDataset acquisition completed successfully.")
