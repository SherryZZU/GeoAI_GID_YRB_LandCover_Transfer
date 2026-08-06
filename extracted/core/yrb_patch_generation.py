#!/usr/bin/env python3
"""Exact code-cell extraction from yrb-patch-generation.ipynb.
Notebook magics/install commands are preserved as comments.
Secrets are redacted. This historical extraction is not the refactored CLI.
"""


# %% [notebook cell 0]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# %% [notebook cell 1]
import numpy as np, rasterio, glob, os, re
from rasterio.windows import Window
from collections import defaultdict

SRC = '/kaggle/input/datasets/sheheryarkhan00/tgrs-lulc' 
OUT = '/kaggle/working/patches'; os.makedirs(OUT, exist_ok=True)
PATCH, STRIDE, MIN_VALID, NODATA = 224, 224, 0.30, 0

def sb_of(p): m=re.search(r'SB(\d+)', os.path.basename(p)); return int(m.group(1)) if m else None

s2_files  = sorted(glob.glob(f'{SRC}/**/*S2_30m_SB*.tif', recursive=True))
lbl_files = sorted(glob.glob(f'{SRC}/**/*LABEL_30m_SB*.tif', recursive=True))

# group labels by sub-basin (list, since a basin may have 1 or several label tiles)
lbl_by_sb = defaultdict(list)
for f in lbl_files: lbl_by_sb[sb_of(f)].append(f)
print(f"found {len(s2_files)} S2 tiles, {len(lbl_files)} label tiles")

def find_label(s2f):
    """Pair an S2 tile to its label by spatial overlap (robust to split mismatch)."""
    sb = sb_of(s2f)
    cands = lbl_by_sb.get(sb, [])
    if len(cands) == 1: return cands[0]          # single label -> use it
    # multiple labels: pick the one whose bounds contain the S2 tile centre
    with rasterio.open(s2f) as s:
        cx, cy = (s.bounds.left+s.bounds.right)/2, (s.bounds.top+s.bounds.bottom)/2
    for lf in cands:
        with rasterio.open(lf) as l:
            b = l.bounds
            if b.left <= cx <= b.right and b.bottom <= cy <= b.top: return lf
    return cands[0] if cands else None

counts = defaultdict(int)
for s2f in s2_files:
    lblf = find_label(s2f)
    if lblf is None: print(f"  no label for {os.path.basename(s2f)}"); continue
    sb = sb_of(s2f)
    with rasterio.open(s2f) as s2src, rasterio.open(lblf) as lsrc:
        # read label by GEOREFERENCED window matching each S2 patch (handles offset grids)
        X_list, Y_list = [], []
        for top in range(0, s2src.height-PATCH+1, STRIDE):
            for left in range(0, s2src.width-PATCH+1, STRIDE):
                win = Window(left, top, PATCH, PATCH)
                # world coords of this S2 patch
                x0, y0 = s2src.xy(top, left, offset='ul')
                lrow, lcol = lsrc.index(x0, y0)            # same coords in label grid
                lwin = Window(lcol, lrow, PATCH, PATCH)
                if (lrow < 0 or lcol < 0 or
                    lrow+PATCH > lsrc.height or lcol+PATCH > lsrc.width): continue
                y = lsrc.read(1, window=lwin)
                if y.shape != (PATCH, PATCH) or (y != NODATA).mean() < MIN_VALID: continue
                x = s2src.read(window=win)
                if x.shape != (6, PATCH, PATCH): continue
                X_list.append(x.astype(np.int16)); Y_list.append(y.astype(np.uint8))
        if X_list:
            tag = re.search(r'(\d{10}-\d{10})', os.path.basename(s2f))
            np.savez_compressed(f'{OUT}/patches_SB{sb}_{tag.group(1) if tag else "0"}.npz',
                                X=np.stack(X_list), y=np.stack(Y_list))
            counts[sb] += len(X_list)
            print(f"  SB{sb} [{os.path.basename(s2f)}]: {len(X_list)} patches")

print("\nper-basin counts:", dict(sorted(counts.items())))
print("total:", sum(counts.values()))

# %% [notebook cell 2]
import shutil
shutil.make_archive('/kaggle/working/yrb_patches', 'zip', '/kaggle/working/patches')
print("zipped -> /kaggle/working/yrb_patches.zip")
