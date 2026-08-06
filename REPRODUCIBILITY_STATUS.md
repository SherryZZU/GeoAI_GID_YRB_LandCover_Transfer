# Reproducibility status

## Runs without research data

```bash
python examples/quick_test.py
python -m unittest discover -s tests -v
```

These commands execute the actual GID palette decoding, 15-to-6 remapping, simulated coarsening, normalization, confusion matrix, OA, and mIoU code.

## Requires external data/checkpoints

- GID-15 training
- YRB patch extraction
- Prithvi/ResNet smoke training
- eight-fold LOBO training
- wall-to-wall prediction
- validation-point assessment

The exact historical notebook code is in `extracted/`. The data-dependent CLI that is fully refactored and tested without those inputs is limited to GID training, YRB patch extraction, GID gap evaluation, and the two-arm smoke test. The full LOBO/prediction/assessment notebooks are retained intact because their large-data/GPU execution cannot be validated in this data-free release.

This repository does **not** claim that a full paper run can be reproduced without the external imagery, patch archives, checkpoints, and GPU environment.

## v2.1.0 additions

The newly added factorial calculations and tests were executed locally. The new
data-dependent GID training, model inference, ANOVA, and eight-fold LOBO commands
were syntax-validated but were not run at full scale because the required imagery,
NPZ patch archives, checkpoints, TerraTorch weights, and GPU environment were not
provided with the notebooks.
