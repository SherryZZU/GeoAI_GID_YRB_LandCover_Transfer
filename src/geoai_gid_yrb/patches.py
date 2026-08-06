"""Georeferenced YRB patch extraction from yrb-patch-generation.ipynb."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import re
import numpy as np


def basin_id(path):
    match=re.search(r'SB(\d+)',Path(path).name)
    return int(match.group(1)) if match else None


def find_label(s2_path, label_paths):
    import rasterio
    sb=basin_id(s2_path); candidates=[p for p in label_paths if basin_id(p)==sb]
    if len(candidates)==1: return candidates[0]
    with rasterio.open(s2_path) as src:
        cx=(src.bounds.left+src.bounds.right)/2; cy=(src.bounds.top+src.bounds.bottom)/2
    for path in candidates:
        with rasterio.open(path) as label:
            b=label.bounds
            if b.left<=cx<=b.right and b.bottom<=cy<=b.top: return path
    return candidates[0] if candidates else None


def extract_patches(s2_path, label_path, patch=224, stride=224, min_valid=0.30, nodata=0):
    import rasterio
    from rasterio.windows import Window
    xs=[]; ys=[]
    with rasterio.open(s2_path) as s2src, rasterio.open(label_path) as lsrc:
        for top in range(0,s2src.height-patch+1,stride):
            for left in range(0,s2src.width-patch+1,stride):
                window=Window(left,top,patch,patch)
                x0,y0=s2src.xy(top,left,offset='ul'); lrow,lcol=lsrc.index(x0,y0)
                if lrow<0 or lcol<0 or lrow+patch>lsrc.height or lcol+patch>lsrc.width: continue
                label=lsrc.read(1,window=Window(lcol,lrow,patch,patch))
                if label.shape!=(patch,patch) or (label!=nodata).mean()<min_valid: continue
                image=s2src.read(window=window)
                if image.shape[1:]!=(patch,patch): continue
                xs.append(image.astype(np.int16)); ys.append(label.astype(np.uint8))
    if not xs: return np.empty((0,6,patch,patch),np.int16),np.empty((0,patch,patch),np.uint8)
    return np.stack(xs),np.stack(ys)
