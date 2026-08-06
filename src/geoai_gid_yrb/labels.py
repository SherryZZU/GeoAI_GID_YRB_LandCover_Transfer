"""Label palettes and 15-to-6 functional remapping from downsample-preds.ipynb."""
from __future__ import annotations
import numpy as np

GID15_PALETTE = [
    ((200,0,0),1), ((250,0,150),2), ((200,150,150),3), ((250,150,150),4),
    ((0,200,0),5), ((150,250,0),6), ((150,200,150),7), ((200,0,200),8),
    ((150,0,250),9), ((150,150,250),10), ((250,200,0),11), ((200,200,0),12),
    ((0,0,200),13), ((0,150,200),14), ((0,200,250),15),
]

# 1 Crop, 2 Forest, 3 Grassland, 4 Bare, 5 Water, 6 Built-up.
# GID-15 has no explicit Bare class, so output 4 is structurally absent here.
GID15_TO_FUNCTIONAL6 = {
    1:6, 2:6, 3:6, 4:6,
    5:1, 6:1, 7:1,
    8:2, 9:2,
    10:3, 11:3, 12:3,
    13:5, 14:5, 15:5,
}

FUNCTIONAL_CLASS_NAMES = {
    0: "NoData", 1: "Cropland", 2: "Forest", 3: "Grassland",
    4: "Bare", 5: "Water", 6: "Built",
}

VALIDATION_LABEL_MAP = {
    "Cropland":1, "Forest":2, "Grassland":3,
    "Bare land":4, "Bare":4, "Water":5,
    "Built-up":6, "Built":6,
}

def rgb_to_idx15(label: np.ndarray) -> np.ndarray:
    """Convert a GID RGB palette label to integer classes 0..15.

    Accepts either (3,H,W) or (H,W,3).
    """
    arr = np.asarray(label)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D RGB label, got {arr.shape}")
    if arr.shape[0] == 3:
        hwc = np.moveaxis(arr, 0, -1)
    elif arr.shape[-1] == 3:
        hwc = arr
    else:
        raise ValueError(f"Expected RGB channel dimension, got {arr.shape}")
    out = np.zeros(hwc.shape[:2], dtype=np.uint8)
    for color, index in GID15_PALETTE:
        out[np.all(hwc == np.asarray(color, dtype=hwc.dtype), axis=-1)] = index
    return out

def remap_15_to_6(labels: np.ndarray) -> np.ndarray:
    """Map GID classes 1..15 to six functional classes, retaining 0 as NoData."""
    arr = np.asarray(labels)
    out = np.zeros(arr.shape, dtype=np.uint8)
    for source, target in GID15_TO_FUNCTIONAL6.items():
        out[arr == source] = target
    return out
