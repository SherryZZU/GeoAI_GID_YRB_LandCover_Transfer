"""Data indexing helpers derived from the GID and YRB notebooks."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np


def basin_id(path: str | Path) -> int:
    match = re.search(r"SB(\d+)", Path(path).name)
    if not match: raise ValueError(f"No basin id in {path}")
    return int(match.group(1))


def deduplicate_by_basename(paths):
    seen=set(); out=[]
    for path in paths:
        name=Path(path).name
        if name not in seen:
            seen.add(name); out.append(Path(path))
    return out


def load_basin_npz(paths) -> tuple[np.ndarray,np.ndarray]:
    xs=[]; ys=[]
    for path in paths:
        data=np.load(path,mmap_mode='r')
        xs.append(np.asarray(data['X'])); ys.append(np.asarray(data['y']))
    if not xs: raise ValueError("No patch files")
    return np.concatenate(xs), np.concatenate(ys)
