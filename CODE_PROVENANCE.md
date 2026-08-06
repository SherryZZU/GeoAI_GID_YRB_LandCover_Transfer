# Code provenance

No placeholder workflow is claimed here. The `extracted/` directory contains every code cell from every uploaded notebook, in original cell order. Notebook installation commands are converted to comments so the `.py` files parse. The `notebooks/` copies are output-free and sanitized.

The reusable files under `src/` and the CLI files under `scripts/` were refactored from those exact code cells. The mapping is:

| Repository component | Source notebook |
|---|---|
| `src/geoai_gid_yrb/gid15.py`, `scripts/train_gid15.py` | `gid15-training.ipynb` |
| `src/geoai_gid_yrb/patches.py`, `scripts/make_yrb_patches.py` | `yrb-patch-generation.ipynb` |
| `scripts/smoke_test_real.py` | `two-arm-smoke-test.ipynb` |
| `src/geoai_gid_yrb/lobo.py` | `lobo-results-part2.ipynb` |
| `src/geoai_gid_yrb/inference.py` | `full-run.ipynb`, `accuracy-mapping.ipynb` |
| `src/geoai_gid_yrb/validation.py` | `accuracy-mapping.ipynb` |
| `src/geoai_gid_yrb/labels.py`, `scripts/evaluate_gid_gap.py` | `downsample-preds.ipynb` |
| `scripts/era5_download.py`, `extracted/auxiliary/TGRS.py` | `TGRS.ipynb` |

`CODE_INVENTORY.json` contains SHA-256 hashes, cell counts, function names, and class names.
