"""Array-only resolution degradation used by the reproducibility quick test."""
from __future__ import annotations
import numpy as np


def block_average(image: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping blocks for CxHxW or HxW imagery."""
    if factor < 1: raise ValueError("factor must be >=1")
    x = np.asarray(image)
    if x.ndim == 2:
        h,w = x.shape; h2=(h//factor)*factor; w2=(w//factor)*factor
        return x[:h2,:w2].reshape(h2//factor,factor,w2//factor,factor).mean((1,3))
    if x.ndim == 3:
        c,h,w = x.shape; h2=(h//factor)*factor; w2=(w//factor)*factor
        return x[:,:h2,:w2].reshape(c,h2//factor,factor,w2//factor,factor).mean((2,4))
    raise ValueError(f"Expected HxW or CxHxW, got {x.shape}")


def block_mode(labels: np.ndarray, factor: int) -> np.ndarray:
    """Majority class in non-overlapping label blocks."""
    y=np.asarray(labels)
    if y.ndim != 2: raise ValueError("labels must be HxW")
    h,w=y.shape; h2=(h//factor)*factor; w2=(w//factor)*factor
    blocks=y[:h2,:w2].reshape(h2//factor,factor,w2//factor,factor).transpose(0,2,1,3)
    out=np.zeros((h2//factor,w2//factor),dtype=y.dtype)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            vals,counts=np.unique(blocks[i,j].ravel(),return_counts=True)
            out[i,j]=vals[counts.argmax()]
    return out
