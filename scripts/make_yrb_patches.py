#!/usr/bin/env python3
import argparse, glob, re
from pathlib import Path
import numpy as np
from geoai_gid_yrb.patches import basin_id,find_label,extract_patches

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--out',default='outputs/patches'); p.add_argument('--patch',type=int,default=224); p.add_argument('--stride',type=int,default=224); p.add_argument('--min-valid',type=float,default=.30); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    s2=sorted(glob.glob(str(Path(a.source)/'**/*S2_30m_SB*.tif'),recursive=True)); labels=sorted(glob.glob(str(Path(a.source)/'**/*LABEL_30m_SB*.tif'),recursive=True))
    total=0
    for image in s2:
        label=find_label(image,labels)
        if label is None: print('no label:',image); continue
        x,y=extract_patches(image,label,a.patch,a.stride,a.min_valid)
        if len(x):
            tag=re.search(r'(\d{10}-\d{10})',Path(image).name); name=f'patches_SB{basin_id(image)}_{tag.group(1) if tag else "0"}.npz'
            np.savez_compressed(out/name,X=x,y=y); total+=len(x); print(name,len(x))
    print('total patches:',total)
if __name__=='__main__': main()
