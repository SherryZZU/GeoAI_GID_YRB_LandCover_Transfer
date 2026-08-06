# GeoAI GID-to-Yellow River Basin real workflow

This release is reconstructed from twelve uploaded Jupyter notebooks after the previous repository was rejected as non-functional. It contains the **complete code-cell extraction**, sanitized notebook copies, reusable modules, actual command-line workflows, and executable tests.

## What is included

- every code cell from all twelve notebooks, in original order, under `extracted/`;
- sanitized, output-free copies of the notebooks under `notebooks/`;
- real GID-15 training, YRB patch generation, two-arm smoke-test, label remapping, resolution degradation, metrics, and model-builder code;
- the original LOBO, wall-to-wall prediction, BN-fixed assessment, GEE, ERA5, and auxiliary analysis code;
- deterministic tests that perform computations, not print-only placeholders.

## Install the core test environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
pip install -e .
```

## Run real data-free verification

```bash
python examples/quick_test.py
python -m unittest discover -s tests -v
```

The quick test decodes a GID palette mask, introduces a known segmentation error, performs simulated coarsening, remaps 15 classes to six functional classes, computes OA/mIoU, asserts expected values, and writes `outputs/quick_test_result.json`.

## Run a data-dependent workflow

```bash
pip install -r requirements-full.txt
pip install -e .

python scripts/train_gid15.py --base /path/to/gid15-yrb --out outputs/gid15
python scripts/make_yrb_patches.py --source /path/to/TGRS_LULC --out outputs/patches
python scripts/smoke_test_real.py --patch-dir /path/to/yrb-patches --gid-checkpoint /path/to/best_model.pth
```

## Complete historical code

The authoritative full extraction is under:

- `extracted/core/`
- `extracted/auxiliary/TGRS.py`

Those scripts retain the original top-level Kaggle/Colab workflow and hard-coded paths, with notebook commands commented and the exposed CDS credential redacted. Use the refactored `scripts/` for new runs.


## Workflows added in v2.1.0

Four further notebooks were read in full and incorporated:

- scene-held-out 15-output GID training;
- a complete P1-P4 2 x 2 resolution-by-label ablation with marginal effects, interaction, and repeated-measures ANOVA;
- the original in-memory eight-fold YRB LOBO comparison;
- the RAM-safe, resumable YRB LOBO comparison.

New executable entry points:

```bash
python scripts/train_gid15_scene_holdout.py --data-dir /path/to/gid15-yrb
python scripts/run_gid_2x2_ablation.py --scene-dir /path/to/gid15-yrb --weights /path/to/gid15_unet_resnet50_15output.pth
python scripts/run_yrb_lobo.py --patch-dir /path/to/yrb-patches --gid-checkpoint /path/to/best_model.pth
```

### Checkpoint compatibility warning

The uploaded notebooks contain two distinct GID training conventions:

1. `train_gid15.py` and `gid15-training-robust-two-phase.ipynb` use **16 outputs**, with index 0 ignored and classes 1..15 retained.
2. `train_gid15_scene_holdout.py` uses **15 outputs**, shifting reference classes 1..15 to model targets 0..14.

Do not interchange these checkpoints. The P1-P4 ablation script requires the 15-output checkpoint produced by `train_gid15_scene_holdout.py`.

## Honesty boundary

Large imagery, NPZ patch archives, trained checkpoints, and the original GPU environment were not supplied in this turn. Therefore, this package validates core computations locally but does not falsely claim a complete full-scale rerun. See `REPRODUCIBILITY_STATUS.md` and `KNOWN_LIMITATIONS.md`.

## Security

The uploaded TGRS notebook contained a CDS API key. It was redacted. Revoke/rotate it before publishing anything. See `SECURITY.md`.
