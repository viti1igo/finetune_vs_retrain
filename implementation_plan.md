# Fine-Tuning vs Retraining from Scratch: Phage-Host Interaction Prediction

## 1. Background & Objective

This project compares **fine-tuning** versus **retraining from scratch** for phage-host interaction (PHI) prediction models. We evaluate whether transfer learning from one Klebsiella phage-host dataset improves performance on a different target dataset, relative to training from scratch.

### Tools Under Evaluation

| Tool | Type | Core Architecture | Input Features | Publication |
|------|------|-------------------|----------------|-------------|
| **DeepPBI-KG** | Deep learning | Deep neural network (PyTorch) | Key gene/protein features from RBPs; K-means negative sampling | Briefings in Bioinformatics, 2024 |
| **PhageHostLearn** | Traditional ML | XGBoost classifier | ESM-2 embeddings of phage RBPs + bacterial K-locus proteins | Nature Communications, 2024 |

### Datasets

| Dataset | Source | Scale | Organism | Data Type |
|---------|--------|-------|----------|-----------|
| **PhageHostLearn dataset** | [Zenodo](https://zenodo.org/records/10850238) / [GitHub](https://github.com/dimiboeckaerts/PhageHostLearn) | 105 phages × 200 strains (~10,006 interactions) | Klebsiella | Spot-test interaction matrix |
| **KlebPhaCol** | [klebphacol.org](http://www.klebphacol.org) | 52 phages × 74 strains | Klebsiella | Host range / spot-test data, genomic data |

---

## 2. Experimental Design

### 2.1 Three Training Regimes

For each tool (DeepPBI-KG, PhageHostLearn), run the following three conditions:

```
Let Dataset A = PhageHostLearn dataset
Let Dataset B = KlebPhaCol dataset
```

| Condition | Code | Training Data | Test Data | Purpose |
|-----------|------|---------------|-----------|---------|
| **C1: Train on B** | `B_train → B_test` | Dataset B train split | Dataset B test split | Baseline: within-dataset performance |
| **C2: Train on A only** | `A → B_test` | Full Dataset A | Dataset B test split | Cross-dataset generalization (zero-shot transfer) |
| **C3: Fine-tune** | `A → finetune(B_train) → B_test` | Pre-train on A, fine-tune on B train split | Dataset B test split | Transfer learning benefit |

> [!IMPORTANT]
> The key comparison is **C1 vs C3**: does pre-training on a larger related dataset (A) and then fine-tuning on B's training data outperform training on B's training data alone? **C2** measures zero-shot cross-dataset generalization.

### 2.2 Evaluation Metrics

- **ROC AUC** (primary)
- **Precision-Recall AUC** (important for imbalanced data)
- **Matthews Correlation Coefficient (MCC)**
- **F1 Score**
- **Accuracy**

---

## 3. Literature-Informed Train/Test Splitting Strategies

### 3.1 Key Data Leakage Risks in PHI Prediction

1. **Sequence similarity leakage**: Closely related phages/hosts in both train and test sets allow the model to "memorize" similarities
2. **Information sharing**: Overlapping protein-protein interaction features that implicitly encode labels
3. **Temporal overlap**: Using future annotations to train models

### 3.2 Recommended Strategies (from Literature)

| Strategy | Description | Pros | Cons | Reference |
|----------|-------------|------|------|-----------|
| **Hierarchical clustering split** | Cluster host strains by K-locus similarity; ensure related strains stay in same fold | Prevents sequence leakage; realistic for unseen strains | May create very unequal splits | PhageHostLearn (Boeckaerts et al., 2024) |
| **Leave-cluster-out validation** | Hold out entire clusters of related organisms | Most stringent; tests true extrapolation | Small test sets; high variance | PAML framework |
| **Random split (stratified)** | Random assignment preserving class balance | Simple; reproducible | Overoptimistic if sequences are related | Common baseline |
| **Temporal split** | Train on older data, test on newer data | Realistic "future prediction" scenario | Requires submission date metadata | General ML best practice |
| **Similarity-filtered split** | Use MMseqs2/Dashing to ensure train-test divergence exceeds a threshold | Explicit control over similarity | Threshold choice is arbitrary | Multiple studies |

### 3.3 Generalization Regimes to Evaluate

Following the PAML framework and PhageHostLearn methodology:

| Regime | Description | Biological Meaning |
|--------|-------------|-------------------|
| **Unseen strains** | Test bacteria not in training set | Can we predict for new clinical isolates? |
| **Unseen phages** | Test phages not in training set | Can we predict for newly isolated phages? |
| **Unseen pairs** | Both phage and host are new | Hardest: full extrapolation |

### 3.4 Recommended Approach for This Study

**Primary strategy**: Hierarchical clustering of bacterial strains (following PhageHostLearn's established methodology), with an 80/20 train/test split on KlebPhaCol (Dataset B).

**Secondary validation**: Report performance breakdown by similarity to training set to show how performance degrades with increasing novelty.

For KlebPhaCol (Dataset B) specifically:
- Cluster the 74 host strains by K-locus type using hierarchical clustering
- Assign clusters to train (~80%) or test (~20%), ensuring no K-locus type overlap
- The 52 phages can be split randomly or kept shared (report both if feasible)

---

## 4. Tool-Specific Implementation Details

### 4.1 DeepPBI-KG

**Repository**: [github.com/Tongqing-Wei/DeepPBI-KG](https://github.com/Tongqing-Wei/DeepPBI-KG)

**Architecture**: Deep neural network trained on features derived from phage/host key genes (RBPs, DNA replication genes).

**Data pipeline**:
1. Build BLAST databases for phage and host genomes
2. Run Prokka annotation
3. Extract key gene features using `integrate_seq.py`
4. Generate K-means negative samples
5. Train DNN

**Fine-tuning approach**:
- Load model weights from Dataset A training
- Re-initialize final classification layer(s) or reduce learning rate
- Continue training on Dataset B's train split with early stopping

**Key challenges**:
- Feature generation requires BLAST + Prokka pipeline for each dataset
- Negative sampling strategy may need recalibration between datasets
- PyTorch model allows straightforward weight loading/freezing

### 4.2 PhageHostLearn

**Repository**: [github.com/dimiboeckaerts/PhageHostLearn](https://github.com/dimiboeckaerts/PhageHostLearn)

**Architecture**: XGBoost classifier on concatenated ESM-2 embeddings of phage RBPs + bacterial K-locus proteins.

**Data pipeline**:
1. Gene calling with PHANOTATE (phage) / Prodigal (bacteria)
2. RBP detection with PhageRBPdetection
3. K-locus typing with Kaptive
4. ESM-2 embedding generation (protein language model)
5. Multi-instance representation aggregation
6. XGBoost training

**Fine-tuning approach** (note: XGBoost is not a neural network):
- **Option A**: Train XGBoost on A, then use `xgb_model` parameter to continue boosting with B's training data (incremental learning)
- **Option B**: Use A-trained model's predictions as an additional feature for B's model (stacking)
- **Option C**: Train ESM-2 embeddings + XGBoost on A, then retrain only the XGBoost head on B's embeddings (feature transfer)

> [!WARNING]
> PhageHostLearn uses XGBoost, not a neural network. True "fine-tuning" (gradient-based weight adjustment) isn't directly applicable. We need to carefully define what "fine-tuning" means for this tool. **Option A** (incremental XGBoost boosting) is the most analogous. **Option C** (feature transfer) may be most meaningful biologically.

---

## 5. Dataset Preparation Details

### 5.1 PhageHostLearn Dataset (Dataset A)

| Property | Detail |
|----------|--------|
| **Source** | [Zenodo: 10850238](https://zenodo.org/records/10850238) |
| **Phages** | 105 Klebsiella phages |
| **Hosts** | 200 Klebsiella clinical strains |
| **Interactions** | ~10,006 pairwise spot-test results |
| **Format** | Interaction matrix + FASTA sequences |
| **Existing splits** | Hierarchical clustering-based CV provided by authors |

### 5.2 KlebPhaCol Dataset (Dataset B)

| Property | Detail |
|----------|--------|
| **Source** | [klebphacol.org](http://www.klebphacol.org) |
| **Phages** | 52 Klebsiella phages (6 families) |
| **Hosts** | 74 Klebsiella clinical strains |
| **Interactions** | Host range / spot-test data |
| **Additional data** | Genomic sequences, TEM images, K-locus types |
| **Existing splits** | None (we create our own) |

### 5.3 Dataset Overlap Check

> [!CAUTION]
> Before running experiments, we **must** check for overlap between Datasets A and B:
> - Are any phage genomes shared (use ANI or BLAST)?
> - Are any host strains shared (same sequence type / K-locus)?
> - If overlap exists, remove overlapping entries from Dataset A when used for pre-training, to prevent leakage.

---

## 6. Experimental Protocol

### Phase 1: Data Acquisition & Preprocessing

1. Download PhageHostLearn data from Zenodo
2. Download KlebPhaCol data from klebphacol.org
3. Standardize formats (FASTA sequences + interaction matrix)
4. Check for phage/host overlap between datasets
5. Generate features for both tools:
   - DeepPBI-KG: BLAST + Prokka + key gene extraction
   - PhageHostLearn: PHANOTATE + PhageRBPdetection + Kaptive + ESM-2

### Phase 2: Train/Test Split Design

1. Hierarchical clustering of KlebPhaCol host strains
2. Define 80/20 train/test split ensuring no K-locus type overlap
3. Document the split (save indices/strain IDs)
4. Validate split quality (no sequence similarity leakage)

### Phase 3: Experiments

For each tool (DeepPBI-KG, PhageHostLearn):

**C1: Train on B, test on B**
1. Train model on KlebPhaCol train split
2. Evaluate on KlebPhaCol test split
3. Record all metrics

**C2: Train on A, test on B**
1. Train model on full PhageHostLearn dataset
2. Evaluate on KlebPhaCol test split (no fine-tuning)
3. Record all metrics

**C3: Train on A, fine-tune on B, test on B**
1. Train/pre-train model on full PhageHostLearn dataset
2. Fine-tune on KlebPhaCol train split
3. Evaluate on KlebPhaCol test split
4. Record all metrics

### Phase 4: Analysis & Reporting

1. Compare metrics across C1, C2, C3 for each tool
2. Statistical significance testing (McNemar's test or DeLong's test for AUC)
3. Analyze where fine-tuning helps/hurts
4. Visualizations: ROC curves, PR curves, confusion matrices
5. Write up findings

---

## 7. Expected Outputs

| Output | Description |
|--------|-------------|
| `results/metrics_comparison.csv` | All metrics for all conditions |
| `results/roc_curves.png` | Overlaid ROC curves per tool |
| `results/pr_curves.png` | Overlaid PR curves per tool |
| `results/confusion_matrices/` | Confusion matrices for each condition |
| `results/dataset_overlap_report.md` | Phage/host overlap analysis |
| `results/split_details.md` | Documentation of train/test split |
| `notebooks/analysis.ipynb` | Full analysis notebook |

---

## 8. Open Questions

> [!IMPORTANT]
> 1. **What does "fine-tuning" mean for XGBoost (PhageHostLearn)?** We propose incremental boosting (Option A above) as the primary approach. Should we also test feature transfer (Option C)?
> 2. **Should we also run with roles reversed?** (i.e., Dataset A = KlebPhaCol, Dataset B = PhageHostLearn). This would test directionality of transfer.
> 3. **Multiple random seeds**: How many replicates? We suggest 5 random seeds for each split to assess variance.
> 4. **Hyperparameter tuning**: Should fine-tuned models use same hyperparameters as from-scratch models, or re-tune on B's validation set?

---

## 9. References

1. Wei, T. et al. (2024). "DeepPBI-KG: A deep learning method for the prediction of phage-bacteria interactions based on key genes." *Briefings in Bioinformatics*.
2. Boeckaerts, D. et al. (2024). "Prediction of Klebsiella phage-host specificity at the strain level." *Nature Communications*.
3. KlebPhaCol Consortium. "KlebPhaCol: An open-source Klebsiella phage collection." [klebphacol.org](http://www.klebphacol.org).
4. Yang, S. et al. (2025). "PhageMind: Meta-learning-based fine-tuning for phage-host prediction."
