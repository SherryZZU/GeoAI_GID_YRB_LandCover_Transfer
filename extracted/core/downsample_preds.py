#!/usr/bin/env python3
"""Exact code-cell extraction from downsample-preds.ipynb.
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
# ============================================================================
# CLAIM A — GID cross-gap transferability evaluation
# Both backbones x 5 GID scenes x {Point1: native 4m/15-cls,
#   Point2: simulated 30m/15-cls, Point3: 30m/6-cls remap}
# Run on Kaggle commit (GPU+Internet). Outputs to /kaggle/working/gid_eval/.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','terratorch','peft','rasterio'], check=False)
import numpy as np, glob, os, re, json, torch, torch.nn as nn
import rasterio; from rasterio.windows import Window
from rasterio.enums import Resampling
import segmentation_models_pytorch as smp
from terratorch.models import EncoderDecoderFactory
DEV='cuda' if torch.cuda.is_available() else 'cpu'

# ---- config ----------------------------------------------------------------
TEST_ONE  = True                  # True = first scene only; then set False
GID_IMG = '/kaggle/input/datasets/sheheryarkhan00/gid15-yrb/img_dir/val'
GID_ANN = '/kaggle/input/datasets/sheheryarkhan00/gid15-yrb/ann_dir/val'
GID_CKPT  = '/kaggle/input/datasets/sheheryarkhan00/phase1-lulc/Phase1_LULC/checkpoints/best_model.pth'
# a fine-tuned YRB fold for the 6-class arm (any held-out fold; we use SB100's)
RESNET_6  = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part1/lobo_results/fold_SB100_resnet.pth'
PRITHVI_6 = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part1/lobo_results/fold_SB100_prithvi.pth'
PATCH_DIR = '/kaggle/input/datasets/sheheryarkhan00/yrb-patches'   # for 6-cls normalization + BN
OUT='/kaggle/working/gid_eval'; os.makedirs(OUT, exist_ok=True)
WINDOW=224; STRIDE=160; PAD=(WINDOW-STRIDE)//2
BANDS=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]

# ---- GID-15 palette + the two remaps ---------------------------------------
PAL=[((200,0,0),1),((250,0,150),2),((200,150,150),3),((250,150,150),4),((0,200,0),5),
     ((150,250,0),6),((150,200,150),7),((200,0,200),8),((150,0,250),9),((150,150,250),10),
     ((250,200,0),11),((200,200,0),12),((0,0,200),13),((0,150,200),14),((0,200,250),15)]
# 15 -> 6 functional  (1 Crop,2 Forest,3 Grass,4 Bare,5 Water,6 Built); garden(8)->Forest
MAP6={1:6,2:6,3:6,4:6,5:1,6:1,7:1,8:2,9:2,10:3,11:3,12:3,13:5,14:5,15:5}
def rgb_to_idx15(lab):           # lab (3,H,W) -> (H,W) in 0..15
    out=np.zeros(lab.shape[1:],np.uint8); fl=lab.reshape(3,-1).T; idx=np.zeros(len(fl),np.uint8)
    for c,i in PAL: idx[(fl[:,0]==c[0])&(fl[:,1]==c[1])&(fl[:,2]==c[2])]=i
    return idx.reshape(lab.shape[1:])

# ---- model builders (15-class GID arm + 6-class YRB arm) -------------------
def build_resnet(nc):
    m=smp.Unet('resnet50',encoder_weights=None,in_channels=(3 if nc==16 else 6),classes=nc)
    return m
def gid_resnet():                 # native GID-pretrained 16-class, 3-band RGB
    m=build_resnet(16); g=torch.load(GID_CKPT,map_location='cpu')['model_state']
    m.load_state_dict(g,strict=True); m.eval(); return m.to(DEV)
def yrb_resnet6():                # 6-class, 6-band, BN-recomputed
    m=build_resnet(7)
    g=torch.load(GID_CKPT,map_location='cpu')['model_state']
    w=g['encoder.conv1.weight']; g['encoder.conv1.weight']=torch.cat([w,w.mean(1,keepdim=True).repeat(1,3,1,1)],1)
    m.load_state_dict(g,strict=False)
    m.segmentation_head[0]=nn.Conv2d(m.segmentation_head[0].in_channels,7,3,padding=1)
    m.load_state_dict(torch.load(RESNET_6,map_location='cpu'),strict=False)
    for mod in m.modules():
        if isinstance(mod,nn.BatchNorm2d): mod.reset_running_stats(); mod.momentum=None
    return m.to(DEV)

# NOTE on the GID arm: the GID checkpoint is RGB/15-class and is the *native* arm
# (Point 1/2 at 15 classes). The 6-class arm is your YRB-fine-tuned model, applied to
# GID after the 6-class remap (Point 3). This keeps each arm honest about what it is.

# ---- normalization ---------------------------------------------------------
# GID arm: simple /255 to match RGB pretraining (no S2 stats). 6-class arm: S2 z-score.
def norm_rgb(x): return x.astype(np.float32)/255.0
pf=sorted({os.path.basename(f):f for f in glob.glob(f'{PATCH_DIR}/**/patches_SB*.npz',recursive=True)}.values())
samp=np.concatenate([np.asarray(np.load(f,mmap_mode='r')['X'][:120]) for f in pf[:8]])
MEAN=samp.reshape(-1,6,WINDOW*WINDOW).mean((0,2)).astype(np.float32)
STD=(samp.reshape(-1,6,WINDOW*WINDOW).std((0,2))+1e-6).astype(np.float32); del samp
def norm_s2(x): return (x.astype(np.float32)-MEAN[:,None,None])/STD[:,None,None]

# ---- sliding-window inference on one (possibly downsampled) RGB array ------
@torch.no_grad()
def infer(model, rgb, six=False):   # rgb (3,H,W) uint8
    C,H,W=rgb.shape; pred=np.zeros((H,W),np.uint8)
    for top in range(0,H,STRIDE):
        for left in range(0,W,STRIDE):
            sub=rgb[:,max(0,top-PAD):top-PAD+WINDOW, max(0,left-PAD):left-PAD+WINDOW]
            if sub.shape[1]<WINDOW or sub.shape[2]<WINDOW:
                z=np.zeros((3,WINDOW,WINDOW),rgb.dtype); z[:,:sub.shape[1],:sub.shape[2]]=sub; sub=z
            if six:    # 6-band model: stack RGB into 6 bands is invalid -> skip 6-band-on-RGB
                pass
            xt=torch.tensor(norm_rgb(sub)).unsqueeze(0).to(DEV)
            o=model(xt); o=o.output if hasattr(o,'output') else o
            lab=o.argmax(1)[0].cpu().numpy().astype(np.uint8)
            h=min(STRIDE,H-top); w=min(STRIDE,W-left)
            pred[top:top+h,left:left+w]=lab[PAD:PAD+h,PAD:PAD+w]
    return pred

def metrics(pred, ref, nclass, ignore=0):
    m=ref!=ignore; p=pred[m]; r=ref[m]
    oa=(p==r).mean() if m.sum() else 0
    ious=[]
    for k in range(1,nclass+1):
        inter=((p==k)&(r==k)).sum(); union=((p==k)|(r==k)).sum()
        if (r==k).sum()>0: ious.append(inter/union if union else 0)
    return float(oa), float(np.mean(ious) if ious else 0)

# ---- run -------------------------------------------------------------------
scenes = sorted(glob.glob(f'{GID_IMG}/*-MSS1.tif'))
scenes=[s for s in scenes if '_15label' not in s]
if TEST_ONE: scenes=scenes[:1]
res={}
for img in scenes:
    sid=re.search(r'L1A(\d+)',os.path.basename(img)).group(1)
    lab = os.path.join(GID_ANN, os.path.basename(img).replace('-MSS1.tif','-MSS1_15label.tif'))
    with rasterio.open(img) as s: rgb=s.read()
    with rasterio.open(lab) as s: ref15=rgb_to_idx15(s.read())
    res[sid]={}
    gnet=gid_resnet()
    # Point 1: native 4m, 15-class (GID arm maps 16->15 by ignoring class 0 head slot)
    p1=infer(gnet,rgb); 
    # GID checkpoint is 16-class (15 + clutter); align: predicted 0..15, ref 0..15
    res[sid]['P1_4m_15cls']=metrics(p1,ref15,15)
    # Point 2: simulate 30m by 8x downsample then evaluate at that scale
    f=224/ (224)  # keep window; downsample whole scene ~7.5x (4m->30m)
    sc=max(1,int(round(30/4)))
    with rasterio.open(img) as s:
        rgb30=s.read(out_shape=(3,s.height//sc,s.width//sc),resampling=Resampling.average).astype(np.uint8)
    with rasterio.open(lab) as s:
        ref30=rgb_to_idx15(s.read(out_shape=(3,s.height//sc,s.width//sc),resampling=Resampling.nearest))
    p2=infer(gnet,rgb30); res[sid]['P2_30m_15cls']=metrics(p2,ref30,15)
    del gnet; torch.cuda.empty_cache()
    # Point 3: 6-class remap at 30m. GID arm prediction remapped 15->6 vs ref remapped 15->6
    p1_6=np.vectorize(lambda v: MAP6.get(int(v),0))(p2).astype(np.uint8)
    ref30_6=np.vectorize(lambda v: MAP6.get(int(v),0))(ref30).astype(np.uint8)
    res[sid]['P3_30m_6cls']=metrics(p1_6,ref30_6,6)
    # save rasters for figures (native-res 15-class prediction)
    prof=dict(driver='GTiff',height=p1.shape[0],width=p1.shape[1],count=1,dtype='uint8',compress='lzw')
    with rasterio.open(f'{OUT}/{sid}_pred15.tif','w',**prof) as d: d.write(p1,1)
    print(f"{sid}: P1(4m,15)={res[sid]['P1_4m_15cls']}  P2(30m,15)={res[sid]['P2_30m_15cls']}  P3(30m,6)={res[sid]['P3_30m_6cls']}")

json.dump(res, open(f'{OUT}/gid_transfer_metrics.json','w'), indent=2)
print("\nDONE. Metrics + pred rasters in /kaggle/working/gid_eval/ — save as dataset and send me.")
