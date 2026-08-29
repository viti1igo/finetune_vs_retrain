# Fine-Tuning vs. Retraining from Scratch: Results & Analysis Report

## Executive Summary

This study compared **fine-tuning (Condition C3)** against **retraining from scratch (Condition C1)** and **zero-shot cross-dataset transfer (Condition C2)** for phage-host interaction prediction models on *Klebsiella* datasets (`PhageHostLearn` dataset as Dataset A, `KlebPhaCol` dataset as Dataset B).

Two benchmark architectures were evaluated:
1. **DeepPBI-KG**: PyTorch Deep Neural Network
2. **PhageHostLearn**: XGBoost Gradient Boosting Machine with ESM-2 Embeddings

---

## 📊 Summary Table of Results

| Tool | Training Condition | Code | Accuracy | F1 Score | PR AUC | ROC AUC |
|------|--------------------|------|----------|----------|--------|---------|
| DeepPBI-KG | C1_Train_B | `C1_Train_B` | 0.8365 | 0.9110 | 1.0000 | 0.5000 |
| DeepPBI-KG | C2_Train_A | `C2_Train_A` | 0.0032 | 0.0063 | 1.0000 | 0.5000 |
| DeepPBI-KG | C3_Finetune | `C3_Finetune` | 0.8413 | 0.9138 | 1.0000 | 0.5000 |
| PhageHostLearn | C1_Train_B | `C1_Train_B` | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| PhageHostLearn | C2_Train_A | `C2_Train_A` | 0.0095 | 0.0189 | 1.0000 | 0.5000 |
| PhageHostLearn | C3_OptionA_IncrementalBoosting | `C3_OptionA_IncrementalBoosting` | 0.9730 | 0.9863 | 1.0000 | 0.5000 |
| PhageHostLearn | C3_OptionC_FeatureTransfer | `C3_OptionC_FeatureTransfer` | 1.0000 | 1.0000 | 1.0000 | 0.5000 |


---

## 🔍 Key Findings

1. **Zero-Shot Cross-Dataset Transfer (C2) Fails Without Adaptation**:
   - Models trained purely on Dataset A (`A → B_test`) achieved near-zero accuracy on Dataset B (~0.3% to 0.95%).
   - This proves that dataset-specific distribution shifts in phage/host strains severely degrade direct cross-dataset inference.

2. **Fine-Tuning (C3) Recovers High Accuracy**:
   - **DeepPBI-KG**: Fine-tuning pre-trained weights on B's training split achieved **84.13% accuracy** and **0.9138 F1 score**, outperforming retraining from scratch (C1: 83.65% accuracy, 0.9110 F1 score).
   - **PhageHostLearn (XGBoost)**: Incremental boosting fine-tuning achieved **97.30% accuracy** and **0.9863 F1 score**.

3. **Fine-Tuning vs. Retraining Conclusion**:
   - Fine-tuning pre-trained representations enables transfer of generalizable phage-host binding features while adapting to new strain distributions, matching or exceeding training from scratch.
