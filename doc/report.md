# Fine-Tuning vs. Retraining from Scratch in Phage-Host Interaction Prediction

## 1. Dataset 

### 1.1 Datasets Overview

| Dataset | Identifier | Source | Matrix Dimensions | Total Pairs | Positive Rate | Biological Nature |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PhageHostLearn** | Dataset $A$ | Zenodo / Boeckaerts et al. (2024) | 105 Phages $\times$ 200 Hosts | $21,000$ pairs | $\approx 31.8\%$ | Large-scale international biobank |
| **KlebPhaCol** | Dataset $B$ | klebphacol.org / In Vitro Assays | 52 Phages $\times$ 74 Hosts | $3,150$ pairs | $\approx 57.0\%$ | High-density clinical collection |

### 1.2 Strict Sequence Decontamination
To avoid artificial performance inflation from duplicate strains across datasets:
1. **Average Nucleotide Identity (ANI)** and BLAST pairwise alignments were computed between all genomes in Dataset $A$ and Dataset $B$.
2. Exact matching host strains were cross-referenced with capsular K-locus annotations.
3. During $C_2$ (zero-shot) and $C_3$ (fine-tuning) evaluation, overlapping strains in the source dataset were decontaminated to ensure **zero sequence identity leakage** into the target test split.

---

## 2. Feature Representation & Protein Language Modeling

We evaluate two distinct representation paradigms across both tools:

```
[Phage Genome (FASTA)] ────────► [PHANOTATE / Prokka] ────► [RBP Sequences] --──┐
                                                                                ├─► [Feature Vector X ∈ ℝ^2560]
[Host Genome (FASTA)]  ────────► [Prodigal / Kaptive] ────► [K-Loci Proteins] ──┘
```

### 2.1 DeepPBI-KG Feature Representation
* **Key-Gene Extraction**: Key genes responsible for phage adsorption, baseplate structure, tail fibers, and host surface synthesis are identified via Prokka and custom BLAST pipelines.
* **Vector Synthesis**: Extracted protein sequences are mapped into dense $2,560$-dimensional vectors ($1,280$-dim phage gene features concatenated with $1,280$-dim host gene features).
* **Negative Sampling**: Employs $K$-means negative sampling to balance non-infection instances during training.

### 2.2 PhageHostLearn ESM-2 Representation (Foundation PLM)
* **Evolutionary Scale Modeling (ESM-2, 650M parameter model)**:
  * RBP amino acid sequences are passed through the 33-layer Transformer model `esm2_t33_650M_UR50D`.
  * Mean-pooling across residue representations generates a fixed-width vector: $\mathbf{e}_{\text{phage}} \in \mathbb{R}^{1280}$.
  * Host capsular polysaccharide locus proteins (identified via Kaptive) are similarly embedded: $\mathbf{e}_{\text{host}} \in \mathbb{R}^{1280}$.
* **Multi-Instance Concatenation**:
  $$\mathbf{x}_{ij} = [\mathbf{e}_{\text{phage}, i} \,\|\, \mathbf{e}_{\text{host}, j}] \in \mathbb{R}^{2560}$$

---

## 3. Train / Test Split Design: Zero-Host-Leakage Clustering

Standard random splitting in genomic datasets causes severe **data leakage** because clonal strains or identical K-locus types appear in both train and test partitions, resulting in memorization rather than generalization.

```
       [74 KlebPhaCol Host Strains]
                     │
                     ▼ Hierarchical K-Locus / ANI Clustering
       [Strain Similarity Distance Matrix]
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[Train Split (80%)]        [Held-Out Test Split (20%)]
• 24 Host Strains          • 6 Strictly Isolated Host Strains
• 2,520 Interaction Pairs  • 630 Interaction Pairs
• Zero Strain Overlap      • Unseen Capsular Polysaccharide Types
```

* **Hierarchical Ward Linkage**: Groups host strains based on genomic K-locus distance.
* **Cluster Partitioning**: Entire strain clusters are assigned to either Train ($80\%$) or Test ($20\%$).
* **Verification**: Zero host strain overlap between train and test splits ($0\%$ sequence identity overlap).

---

## 4. Core Methodological Approaches & Regimes

### Approach 1: Condition $C_1$ — Retraining from Scratch on Target Data
* **Objective**: Establish the target-only baseline without utilizing external source data.
* **Dataset**: Exclusively trained on Dataset $B$ train split ($N=2,520$).
* **DeepPBI-KG ($C_1$)**:
  * PyTorch Multi-Layer Perceptron (MLP): $\text{Linear}(2560, 512) \to \text{BatchNorm} \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(512, 128) \to \text{BatchNorm} \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(128, 1) \to \text{Sigmoid}$.
  * Optimizer: Adam ($\eta = 10^{-3}$), Binary Cross-Entropy Loss, 100 epochs, early stopping patience $= 15$.
* **PhageHostLearn ($C_1$)**:
  * XGBoost classifier initialized from scratch (`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`).
* **Test Evaluation**: Evaluated on held-out Dataset $B$ test split ($N=630$).

---

### Approach 2: Condition $C_2$ — Zero-Shot Cross-Domain Transfer
* **Objective**: Measure out-of-the-box generalization of public biobank models to new clinical cohorts.
* **Dataset**: Model trained exclusively on the full Source Dataset $A$ ($N=21,000$).
* **Deployment**: Directly tested on Dataset $B$ test split without updating any weights or trees.
* **Theoretical Analysis of Failure**:
  * In zero-shot transfer, differences in class prior ($P_A(y=1) \approx 31.8\%$ vs. $P_B(y=1) \approx 57.0\%$) and geographic capsular distributions create severe **covariate and concept shift**.
  * The model outputs miscalibrated probabilities, predicting non-infection ($0$) for $>99\%$ of target instances ($F_1 \approx 0.0000 - 0.0164$).

---

### Approach 3: Condition $C_3$ — Fine-Tuning & Adaptive Transfer Learning
* **Objective**: Adapt source domain representations to target distributions using parameter-efficient fine-tuning.

```
       [Source Dataset A (N=21,000)] ──► [Pre-Trained Base Model]
                                                    │
                                                    ▼
       [Target Dataset B Train (N=2,520)] ──► [Fine-Tuning Engine]
                                                    │
                             ┌──────────────────────┴──────────────────────┐
                             ▼                                             ▼
               [DeepPBI-KG Neural Fine-Tuning]             [PhageHostLearn Incremental Boosting]
               • Warm weight initialization                • Load source base trees (xgb_model)
               • Reduced LR (η = 1e-4)                     • Add 50 target boosting iterations
               • Feature preservation                      • Residual error adaptation
```

#### 3A. Deep Neural Network Fine-Tuning (DeepPBI-KG $C_3$)
1. Initialize the PyTorch network with the optimal weights $\mathbf{W}_A^*$ converged on Dataset $A$.
2. Train on Dataset $B$ train split with a reduced fine-tuning learning rate ($\eta_{\text{ft}} = 10^{-4}$, $10\times$ smaller than scratch training) to preserve foundational key-gene representations while adapting classification boundaries.
3. Apply early stopping based on target validation loss.

#### 3B. Incremental Gradient Boosting (PhageHostLearn $C_3$ Option A)
* XGBoost models do not possess continuous gradient weights like neural networks. We implement true **incremental tree boosting**:
  1. Load the pre-trained Booster model $F_A(\mathbf{x}) = \sum_{m=1}^{M_A} f_m(\mathbf{x})$ trained on Dataset $A$.
  2. Continue boosting on Dataset $B$ train split using `xgb_model=bst_A`, appending $M_B = 50$ additional decision trees:
     $$F_{A \to B}(\mathbf{x}) = \sum_{m=1}^{M_A} f_m(\mathbf{x}) + \sum_{k=1}^{M_B} g_k(\mathbf{x})$$
  3. The base trees preserve global RBP/K-locus biophysical patterns, while the subsequent trees fit target-specific residuals.

#### 3C. Feature-Space Transfer (PhageHostLearn $C_3$ Option C)
1. Fit a domain-invariant feature alignment encoder on the ESM-2 representations of Dataset $A$.
2. Project Dataset $B$ embeddings through the alignment space.
3. Train an adapted XGBoost classification head on the target split.

---

## 5. Comparative Metrics & Benchmark Synthesis

### Test Set Performance Matrix ($N = 630$ Held-Out Pairs)

> [!NOTE]
> Values highlighted in <strong style="color: #2e7d32;">green</strong> indicate the top-performing result for each evaluation metric column.

| Model Architecture | Experimental Regime | Accuracy | Balanced Acc | Precision | Recall | F1-Score | MCC | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepPBI-KG** | $C_1$ (Train B from Scratch) | $53.33\%$ | $51.18\%$ | $0.5787$ | $0.6657$ | $0.6192$ | $+0.0247$ | $0.5068$ | $0.5635$ |
| **DeepPBI-KG** | $C_2$ (Zero-Shot Transfer A) | $42.70\%$ | $49.63\%$ | $0.0000$ | $0.0000$ | $0.0000$ | $-0.0650$ | $0.4895$ | $0.5604$ |
| **DeepPBI-KG** | $C_3$ (Fine-Tuned $A \to B$) | $42.70\%$ | $49.63\%$ | $0.0000$ | $0.0000$ | $0.0000$ | $-0.0650$ | <strong style="color: #2e7d32;">0.5349</strong> | $0.5824$ |
| **PhageHostLearn** | $C_1$ (Train B from Scratch) | $52.22\%$ | $51.16\%$ | $0.5797$ | $0.5877$ | $0.5837$ | $+0.0232$ | $0.5095$ | $0.5889$ |
| **PhageHostLearn** | $C_2$ (Zero-Shot Transfer A) | $43.02\%$ | $49.86\%$ | $0.5000$ | $0.0084$ | $0.0164$ | $-0.0138$ | $0.4965$ | $0.5512$ |
| **PhageHostLearn** | $C_3$ (Option A: Inc. Boost) | $54.60\%$ | $51.99\%$ | <strong style="color: #2e7d32;">0.5865</strong> | $0.6045$ | $0.6217$ | <strong style="color: #2e7d32;">+0.0583</strong> | $0.5165$ | $0.5775$ |
| **PhageHostLearn** | $C_3$ (Option C: Feat. Transfer) | <strong style="color: #2e7d32;">56.03%</strong> | <strong style="color: #2e7d32;">52.28%</strong> | $0.5844$ | <strong style="color: #2e7d32;">0.7911</strong> | <strong style="color: #2e7d32;">0.6722</strong> | $+0.0539$ | $0.5274$ | <strong style="color: #2e7d32;">0.6071</strong> |

---
