# Additional notebooks incorporated in v2.1.0

## notebook67e0991e34.ipynb

Contains two complete workflows:

1. A 15-output ResNet-50/U-Net source model trained with five named scenes held out.
2. The complete four-cell resolution-by-label factorial experiment:
   P1 native/fine, P2 coarse/fine, P3 coarse/coarse, and P4 native/coarse.

It writes per-scene OA/mIoU, cell summaries, two marginal contrasts, an interaction,
and an OA ANOVA table.

## notebook8572cef2e5.ipynb

Contains the robust 16-output GID workflow used by the existing `scripts/train_gid15.py`:
moderate Albumentations, log-smoothed inverse-frequency weights, Dice plus weighted
cross-entropy, decoder warm-up, full-network fine-tuning, mIoU early stopping, and curves.

## yrb-lulc-lobo-comparison.ipynb

Contains the original in-memory dual-backbone LOBO workflow over eight sub-basins.

## yrb-lulc-lobo-comparison22.ipynb

Contains the RAM-safe successor. Basin arrays are loaded on demand, normalization is
estimated from memory-mapped samples, completed folds can be resumed, and arrays are
released after each fold. The refactored executable is `scripts/run_yrb_lobo.py`.

## Important incompatibility

The 15-output and 16-output GID checkpoints use different target encodings. They are
both preserved because both occur in the source materials, but they are not interchangeable.
