# Extracted computational workflow

1. **GID-15 source training**: `gid15-training.ipynb` trains a ResNet-50 U-Net with 16 output indices, Dice plus weighted cross-entropy, decoder warm-up, full-network fine-tuning, early stopping, and a saved `best_model.pth`.
2. **YRB raster-to-patch conversion**: `yrb-patch-generation.ipynb` pairs Sentinel-2 and functional WorldCover rasters by sub-basin/spatial overlap and writes 224 x 224 NPZ patches.
3. **Two-arm smoke test**: `two-arm-smoke-test.ipynb` runs Prithvi-EO-2.0 with LoRA and a GID-initialized six-band ResNet/U-Net for two epochs.
4. **LOBO training**: `lobo-results-part2.ipynb` holds out each of eight sub-basins, samples up to 1,500 patches from every other basin, and trains/evaluates both backbones.
5. **Wall-to-wall inference**: `full-run.ipynb` predicts each basin with its held-out fold model using overlapping 224-pixel windows and writes class and entropy GeoTIFFs.
6. **Reference-point assessment**: `accuracy-mapping.ipynb` assigns points to S2 tiles, loads the basin-specific held-out models, recomputes missing ResNet BatchNorm statistics, predicts point classes, and calculates raw/area-weighted accuracy.
7. **GID gap evaluation**: `downsample-preds.ipynb` evaluates native 15-class predictions, simulated coarse 15-class predictions, and a six-class remapping of the coarse GID predictions.
8. **Auxiliary analysis**: `TGRS.ipynb` contains ERA5/GEE acquisition, teleconnection audits, prediction-raster handling, and area-adjusted accuracy calculations. It is retained separately because it is broader than the land-cover transfer paper.

9. **Alternative scene-held-out source training**: `gid15-scene-holdout-and-2x2-ablation.ipynb` trains a 15-output model after excluding five named evaluation scenes.
10. **Complete factorial gap analysis**: the same notebook evaluates P1 native/fine, P2 coarse/fine, P3 coarse/coarse, and P4 native/coarse, then calculates marginal effects and an interaction.
11. **Original LOBO implementation**: `yrb-lobo-comparison-in-memory.ipynb` is retained as the first full eight-fold dual-backbone comparison.
12. **RAM-safe LOBO implementation**: `yrb-lobo-comparison-ram-safe.ipynb` indexes NPZ files by basin, loads basins on demand, supports resume, and saves each completed fold immediately.
