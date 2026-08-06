#!/usr/bin/env python3
"""Data-driven GID resolution/label-gap evaluation.

This is the CLI equivalent of downsample-preds.ipynb. It deliberately evaluates
P3 by remapping the P2 GID prediction, matching the actual notebook code.
"""
import argparse, json
from pathlib import Path
import numpy as np
from geoai_gid_yrb.labels import remap_15_to_6
from geoai_gid_yrb.metrics import segmentation_metrics
from geoai_gid_yrb.resolution import block_mode

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prediction',required=True,help='Native GID integer prediction .npy')
    p.add_argument('--reference',required=True,help='Native GID integer reference .npy'); p.add_argument('--factor',type=int,default=8); p.add_argument('--out',default='outputs/gid_gap_metrics.json'); a=p.parse_args()
    pred=np.load(a.prediction); ref=np.load(a.reference)
    p1=segmentation_metrics(ref,pred,16,0)
    pred30=block_mode(pred,a.factor); ref30=block_mode(ref,a.factor)
    p2=segmentation_metrics(ref30,pred30,16,0)
    p3=segmentation_metrics(remap_15_to_6(ref30),remap_15_to_6(pred30),7,0)
    result={'P1_native_15class':p1,'P2_simulated_coarse_15class':p2,'P3_simulated_coarse_6class_remap':p3}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(out)
if __name__=='__main__': main()
