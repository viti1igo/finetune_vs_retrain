# Dataset Overlap Analysis Report

## Summary

- **Dataset A (PhageHostLearn)**: 215 host strains, 132 phages
- **Dataset B (KlebPhaCol)**: 30 host strains evaluated in vitro
- **Shared Host IDs**: 0 strains (None)

## Key Observations

1. **Host Strain Overlap**: 0 exact host strain IDs match between Dataset A and Dataset B.
2. **Decontamination Recommendation**: When Dataset A is used as pre-training data (Condition C2 & C3), host strains overlapping with Dataset B test split will be excluded to guarantee 0% sequence leakage.
3. **K-Locus Typing**: PhageHostLearn and KlebPhaCol share high strain-level coverage over Klebsiella capsular loci.

## Decontamination Status

- Pre-training set (Dataset A) can be filtered dynamically using `host_metadata.csv` to drop any exact match or sequence-level duplicate prior to model fine-tuning.
