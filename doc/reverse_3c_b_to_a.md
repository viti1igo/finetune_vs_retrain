# Reverse Method 3C: KlebPhaCol → PhageHostLearn

## Purpose

This reproducible Stage 7 experiment reverses Method 3C. Frozen ESM-2 pair embeddings are used as fixed inputs; **ESM-2 is not trained or fine-tuned**. A `StandardScaler + RidgeClassifier` compatibility encoder is fit on the full KlebPhaCol source (Dataset B), its continuous decision score is appended to Dataset A features, and a new XGBoost head is trained only on Dataset A training hosts.

## Dataset A host-disjoint evaluation

- Seed: `42`
- Dataset A pairs: 21000; hosts: 200
- Train: 16800 pairs across 160 hosts
- Held out: 4200 pairs across 40 unseen hosts
- Leakage checks: {"host_sets_disjoint": true, "all_pairs_assigned_once": true, "both_target_partitions_have_two_classes": true}
- Input dimension: 2,560; reverse-3C augmented dimension: 2,561.

## Held-out Dataset A results

| Model | ROC_AUC | PR_AUC | MCC | F1 | Accuracy | Positive_predictions |
| --- | --- | --- | --- | --- | --- | --- |
| A_from_scratch | 0.760457 | 0.135252 | 0.190455 | 0.148148 | 0.983571 | 14.000000 |
| B_zero_shot | 0.479226 | 0.014408 | 0.008038 | 0.032446 | 0.290000 | 3015.000000 |
| B_to_A_reverse_3C | 0.785733 | 0.130513 | 0.190455 | 0.148148 | 0.983571 | 14.000000 |

The notebook at `notebooks/reverse_3c_b_to_a_results.ipynb` loads these saved results and figures. Prediction-level values are in `results/reverse_3c_b_to_a/held_out_predictions.csv`.

## Comparisons

1. **A from scratch** trains XGBoost on Dataset A training hosts.
2. **B source-only zero shot** trains XGBoost on Dataset B and scores held-out Dataset A hosts without adaptation.
3. **B → A reverse 3C** uses the Dataset B Ridge compatibility score plus a target-specific XGBoost head trained on Dataset A training hosts.

## Limitation requiring resolution

The current checked-in feature generator creates Dataset B labels with `np.random.choice(...)` rather than loading observed KlebPhaCol interactions. Consequently, this run verifies pipeline reproducibility and leakage isolation, but cannot support a biological claim about reverse transfer until Dataset B is regenerated from curated experimental labels. This report supersedes the inconsistent legacy Stage 7 reverse-transfer summaries; only artifacts produced by this runner are reported here.
