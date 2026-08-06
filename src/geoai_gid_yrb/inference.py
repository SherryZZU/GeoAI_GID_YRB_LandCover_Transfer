"""Center-crop sliding-window inference from full-run.ipynb."""
from __future__ import annotations
import numpy as np


def predictive_entropy(probabilities, epsilon=1e-6):
    p=np.clip(np.asarray(probabilities,dtype=np.float32),epsilon,1.0)
    return -(p*np.log(p)).sum(axis=0)


def center_crop_indices(window=224, stride=160):
    pad=(window-stride)//2
    if pad<0: raise ValueError('stride must be <= window')
    return pad


def save_raster(array, profile, path, dtype):
    import rasterio
    profile=dict(profile); profile.update(count=1,dtype=dtype,nodata=0,compress='lzw')
    with rasterio.open(path,'w',**profile) as dst:
        dst.write(np.asarray(array).astype(dtype),1)
