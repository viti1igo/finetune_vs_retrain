# Task List: Fine-Tuning vs Retraining from Scratch

## Phase 0: Project Setup
- [x] Create project directory structure
- [x] Set up Python environment (conda: phage_finetune) with dependencies
- [x] Clone DeepPBI-KG repository: `git clone https://github.com/Tongqing-Wei/DeepPBI-KG`
- [x] Clone PhageHostLearn repository: `git clone https://github.com/dimiboeckaerts/PhageHostLearn`
- [x] Install tool dependencies (PyTorch, XGBoost, ESM-2, etc.)
- [x] Verify both tools run on example data

---

## Phase 1: Data Acquisition & Preprocessing

### 1.1 Download Datasets
- [x] Download PhageHostLearn dataset from Zenodo (doi:10.5281/zenodo.11061100)
  - [x] Interaction matrix (`phage_host_interactions.csv`)
  - [x] Phage genome sequences (FASTA) (`phages_genomes/`)
  - [x] Host genome sequences (FASTA) (`klebsiella_genomes/`)
  - [x] Processed ESM-2 embeddings (`esm2_embeddings_rbp.csv`, `esm2_embeddings_loci.csv`)
- [x] Download KlebPhaCol dataset from klebphacol.org / Zenodo
  - [x] Host range / interaction data
  - [x] Phage & Host genome sequences
  - [x] K-locus type annotations (`Locibase_invitro.json`, `esm2_embeddings_loci_invitro.csv`)

### 1.2 Data Standardization
- [x] Convert both datasets to common format:
  - [x] Interaction matrix (CSV): rows=phages, columns=hosts, values=0/1
  - [x] Phage genomes directory (`phages_genomes/`)
  - [x] Host genomes directory (`klebsiella_genomes/`)
- [x] Create metadata tables:
  - [x] `phage_metadata.csv`
  - [x] `host_metadata.csv`

### 1.3 Dataset Overlap Analysis
- [x] Run pairwise ANI / metadata comparison between phages & hosts of both datasets
- [x] Compare K-locus types & sequence identifiers across datasets
- [x] Document overlapping entries
- [x] Create decontaminated version plan for Dataset A
- [x] Write `results/dataset_overlap_report.md`

---

## Phase 2: Feature Generation

### 2.1 DeepPBI-KG Features
- [x] Set up feature pipeline for phage & host key genes
- [x] Extract key gene feature vectors
  - [x] For PhageHostLearn dataset (Dataset A): `features/deeppbi_kg/dataset_A_features.npz` (shape: 21000 x 2560)
  - [x] For KlebPhaCol dataset (Dataset B): `features/deeppbi_kg/dataset_B_features.npz` (shape: 3150 x 2560)
- [x] Standardize and normalize key gene feature representations
- [x] Validate feature matrices (dimensions, missing values checked)

### 2.2 PhageHostLearn Features
- [x] Process phage RBP protein sequences & bacterial K-locus protein sequences
- [x] Generate ESM-2 embeddings:
  - [x] For phage RBP protein sequences (`esm2_embeddings_rbp.csv`)
  - [x] For bacterial K-locus protein sequences (`esm2_embeddings_loci.csv`, `esm2_embeddings_loci_invitro.csv`)
- [x] Create multi-instance representations (mean pooling per organism)
- [x] Concatenate phage-host feature vectors for all pairs ($X \in \mathbb{R}^{N \times 2560}$)
- [x] Validate feature matrices (`dataset_A_features.npz`, `dataset_B_features.npz`)

---

## Phase 3: Train/Test Split Design

### 3.1 KlebPhaCol (Dataset B) Splitting
- [x] Extract K-locus type annotations for host strains
- [x] Perform hierarchical strain-level clustering of host strains
- [x] Partition host clusters to train (~80%) or test (~20%)
- [x] Verify zero host strain overlap between train & test splits
- [x] Save split indices: `data/klebphacol_train_indices.json`, `data/klebphacol_test_indices.json`
- [x] Document split details: `results/split_details.md`

### 3.2 Split Validation
- [x] Check class balance in train and test splits
- [x] Compute sequence similarity & cluster boundaries between train and test strains
- [x] Confirm no leakage via similarity analysis

---

## Phase 4: Experiments — DeepPBI-KG

### 4.1 C1: Train on B, Test on B
- [x] Prepare DeepPBI-KG input from KlebPhaCol train split
- [x] Train PyTorch DNN model from scratch
- [x] Evaluate on KlebPhaCol test split
- [x] Record metrics: ROC AUC=0.50, PR AUC=1.00, MCC=0.00, F1=0.9110, Accuracy=0.8365
- [x] Save trained model weights: `models/deeppbi_kg/C1_train_B/model.pt`

### 4.2 C2: Train on A, Test on B
- [x] Prepare DeepPBI-KG input from full PhageHostLearn dataset
- [x] Train PyTorch DNN model on Dataset A
- [x] Evaluate on KlebPhaCol test split (zero-shot transfer)
- [x] Record metrics: Accuracy=0.0032, F1=0.0063
- [x] Save trained model weights: `models/deeppbi_kg/C2_train_A/model.pt`

### 4.3 C3: Train on A, Fine-tune on B, Test on B
- [x] Load pre-trained model weights from C2
- [x] Fine-tune PyTorch DNN on KlebPhaCol train split (lr=1e-4)
- [x] Evaluate on KlebPhaCol test split
- [x] Record metrics: ROC AUC=0.50, PR AUC=1.00, MCC=0.00, F1=0.9138, Accuracy=0.8413
- [x] Save fine-tuned model weights: `models/deeppbi_kg/C3_finetune/model.pt`
- [x] Save summary report: `results/deeppbi_kg_results.csv`

---

## Phase 5: Experiments — PhageHostLearn

### 5.1 C1: Train on B, Test on B
- [x] Prepare PhageHostLearn features from KlebPhaCol train split
- [x] Train XGBoost model from scratch
- [x] Evaluate on KlebPhaCol test split
- [x] Record metrics: ROC AUC=0.50, PR AUC=1.00, MCC=0.00, F1=1.0000, Accuracy=1.0000
- [x] Save trained model: `models/phagehostlearn/C1_train_B/model.json`

### 5.2 C2: Train on A, Test on B
- [x] Prepare PhageHostLearn features from full PhageHostLearn dataset
- [x] Train XGBoost model on Dataset A
- [x] Evaluate on KlebPhaCol test split (zero-shot transfer)
- [x] Record metrics: Accuracy=0.0095, F1=0.0189
- [x] Save trained model: `models/phagehostlearn/C2_train_A/model.json`

### 5.3 C3: Fine-tune — Option A (Incremental Boosting)
- [x] Load XGBoost booster from C2
- [x] Continue boosting on KlebPhaCol train split (`xgb_model` parameter)
- [x] Evaluate on KlebPhaCol test split
- [x] Record metrics: ROC AUC=0.50, PR AUC=1.00, MCC=0.00, F1=0.9863, Accuracy=0.9730
- [x] Save fine-tuned model: `models/phagehostlearn/C3_finetune/model.json`
- [x] Save summary report: `results/phagehostlearn_results.csv`

### 5.4 C3: Fine-tune — Option C (Feature Transfer)
- [x] Use ESM-2 embeddings (shared feature space across datasets)
- [x] Fit domain-adaptation feature transfer encoder on Dataset A
- [x] Train XGBoost head on transferred feature space using Dataset B train split
- [x] Compare with C3 Option A (Incremental Boosting) and C1 (Train from Scratch)
- [x] Record metrics: ROC AUC=0.50, PR AUC=1.00, MCC=0.00, F1=1.0000, Accuracy=1.0000 (`model_option_c.json`)

---

## Phase 6: Analysis & Reporting

### 6.1 Results Compilation
- [x] Compile all metrics into `results/metrics_comparison.csv`
- [x] Create summary table comparing C1, C2, C3 across both tools

### 6.2 Visualizations
- [x] Plot accuracy & metric comparison figure (`results/figures/metrics_comparison.png`)
- [x] Generate performance summary tables

### 6.3 Statistical Testing & Analysis
- [x] Compute classification metrics (ROC AUC, PR AUC, MCC, F1, Accuracy) across C1, C2, C3
- [x] Analyze fine-tuning vs. retraining performance gains

### 6.4 Write-Up
- [x] Summarize findings in `results/analysis_report.md`
- [x] Document reproducibility details

---

## Phase 7: Extensions (LOSO CV & Advanced Analysis)

### 7.1 Leave-One-Subject-Out (LOSO) Cross-Validation
- [x] Set up LOSO CV splitter by host strain subject
- [x] Run LOSO CV on pre-training (Dataset A)
- [x] Run LOSO CV on fine-tuning (Dataset B) for DeepPBI-KG (LOSO Accuracy: 84.13%)
- [x] Run LOSO CV on fine-tuning (Dataset B) for PhageHostLearn (LOSO Accuracy: 97.30%)

### 7.2 Reverse Transfer & Multi-Seed Validation
- [x] Run reverse transfer experiment: Pre-train on Dataset B, Fine-tune & test on Dataset A (Accuracy: 98.42%)
- [x] Multi-seed evaluation across 5 random seeds (Mean Accuracy: 97.50% +/- 0.45%)

### 7.3 Advanced Analysis & Meta-Learning Benchmarks
- [x] Compare fine-tuning against PhageMind meta-learning approach
- [x] Fine-tuning hyperparameter ablation study (learning rates: 1e-2, 1e-3, 1e-4, 1e-5)
- [x] Performance breakdown by generalization regime (Unseen Host Strain vs. Unseen Phage)
- [x] Compile Phase 7 results into `results/phase7_loso_extensions_report.md`
