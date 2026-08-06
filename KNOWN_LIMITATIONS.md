# Known limitations and unresolved issues

1. `downsample-preds.ipynb` loads paths for YRB six-class checkpoints but Point 3 actually remaps the Point 2 GID prediction to six classes. The YRB model is not used in that Point 3 calculation. The CLI preserves this definition rather than silently changing the experiment.
2. The GID coarse-label reference in the historical notebook uses nearest-neighbor raster resampling. The data-free utility also provides majority block aggregation, but results are not interchangeable.
3. LOBO ResNet checkpoints saved only trainable parameters, not BatchNorm running buffers. Later assessment code recomputes BatchNorm statistics from training-fold patches.
4. Many model loads use `strict=False`. Users must inspect missing/unexpected keys when changing library versions.
5. The area weights in the assessment notebook are hard-coded WorldCover-derived values. A publication release should archive the script/table used to calculate them.
6. One cell in `TGRS.ipynb` ended with `NameError: os is not defined`; the exact cell remains in the extracted historical script.
7. Notebook dependencies were installed at runtime without pinned versions. `requirements-full.txt` records package names but cannot reconstruct the original Kaggle image exactly.
8. Uploaded `TGRS.ipynb` contained a CDS API key. It has been redacted from this package. The key should be revoked and replaced.

- The source notebooks contain two incompatible GID checkpoint conventions, 15-output shifted targets and 16-output retained targets. The repository documents rather than conceals this difference.
- The full P1-P4 ablation requires the 15-output scene-held-out checkpoint and the five labeled GID scenes.
- The full LOBO comparison requires the YRB NPZ patch archives, the 16-output GID checkpoint, TerraTorch/PEFT, pretrained Prithvi weights, and substantial GPU time.
