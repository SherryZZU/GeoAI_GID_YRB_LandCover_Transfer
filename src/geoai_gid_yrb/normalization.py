"""Six-band normalization used by the YRB notebooks."""
from __future__ import annotations
import numpy as np


def compute_band_stats(samples: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Compute per-band mean/std from (N,C,H,W) samples."""
    x = np.asarray(samples)
    if x.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W), got {x.shape}")
    mean = x.mean(axis=(0,2,3)).astype(np.float32)
    std = (x.std(axis=(0,2,3)) + 1e-6).astype(np.float32)
    return mean, std


def normalize_bands(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if arr.ndim == 3:
        return (arr - mean[:,None,None]) / std[:,None,None]
    if arr.ndim == 4:
        return (arr - mean[None,:,None,None]) / std[None,:,None,None]
    raise ValueError(f"Expected (C,H,W) or (N,C,H,W), got {arr.shape}")
