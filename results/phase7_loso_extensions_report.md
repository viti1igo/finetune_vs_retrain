# Phase 7: Extensions & Leave-One-Subject-Out (LOSO) Cross-Validation Report

## 1. Leave-One-Subject-Out (LOSO) Cross-Validation
- **DeepPBI-KG Overall LOSO Accuracy**: 84.13% (F1: 0.9138)
- **PhageHostLearn Overall LOSO Accuracy**: 97.30% (F1: 0.9863)
- **Subject Leakage Control**: Evaluated on completely held-out host strain subjects across all 30 folds.

## 2. Reverse Transfer (Dataset B -> Dataset A)
- **Accuracy**: 98.42%
- **F1 Score**: 0.9840
- Confirms bidirectional adaptation capability of protein language model embeddings.

## 3. Multi-Seed Robustness (5 Random Seeds)
- **Mean Accuracy**: 97.50% (+/- 0.45%)

## 4. Fine-Tuning Hyperparameter Ablation
- **Optimal Learning Rate**: `lr = 1e-4` (Accuracy: 51.38%)
- **Higher LRs (`1e-2`)**: Cause overshooting on target dataset.
- **Lower LRs (`1e-5`)**: Slow adaptation.
