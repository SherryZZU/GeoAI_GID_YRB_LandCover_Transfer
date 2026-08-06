# Required external data

The repository does not redistribute large or licensed imagery.

- GID-15 arranged as `img_dir/{train,val}` and `ann_dir/{train,val}` with integer-coded `*_15label.png` masks.
- Sentinel-2 six-band 30 m GeoTIFFs named `S2_30m_SB*.tif` in band order BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2.
- Functional label rasters named `LABEL_30m_SB*.tif`, classes 0 through 6.
- Generated patch files named `patches_SB*.npz`, containing arrays `X` and `y`.
- GID source checkpoint `best_model.pth` containing the key `model_state`.
- LOBO checkpoints `fold_SB{basin}_{resnet|prithvi}.pth` and fold JSON summaries.
- Optional cleaned reference-point CSV with `lon`, `lat`, `land_type`, and optionally `confidence`.

Update `configs/paths.example.yaml` and do not commit private credentials or raw imagery.
