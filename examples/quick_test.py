#!/usr/bin/env python3
"""Data-free verification of real label, resolution, and metric code."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from geoai_gid_yrb.labels import GID15_PALETTE,rgb_to_idx15,remap_15_to_6
from geoai_gid_yrb.metrics import segmentation_metrics
from geoai_gid_yrb.resolution import block_mode

# Construct a real GID-palette label, decode it, introduce one prediction error,
# simulate coarse resolution, remap to functional classes, and calculate metrics.
indices=np.array([[1,1,5,5],[1,1,5,5],[9,9,13,13],[9,9,13,13]],dtype=np.uint8)
lookup={idx:color for color,idx in GID15_PALETTE}
rgb=np.array([[lookup[int(v)] for v in row] for row in indices],dtype=np.uint8)
decoded=rgb_to_idx15(rgb)
assert np.array_equal(decoded,indices)
prediction=decoded.copy(); prediction[0,0]=2
native=segmentation_metrics(decoded,prediction,16,0)
coarse_ref=block_mode(decoded,2); coarse_pred=block_mode(prediction,2)
functional=segmentation_metrics(remap_15_to_6(coarse_ref),remap_15_to_6(coarse_pred),7,0)
assert native['overall_accuracy'] == 15/16
assert functional['overall_accuracy'] == 1.0
result={'native':native,'functional_after_coarsening':functional,'decoded_classes':sorted(np.unique(decoded).tolist())}
out=ROOT/'outputs'/'quick_test_result.json'; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2)); print(f'PASS: wrote {out}')
