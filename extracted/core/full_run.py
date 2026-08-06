#!/usr/bin/env python3
"""Exact code-cell extraction from full-run.ipynb.
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
# LULC PREDICTION (all basins, both backbones) -> rasters for Figs 3/4/5
# Center-crop overlapping windows = clean edges. Held-out model per basin.
# Run on Kaggle commit (GPU+Internet). Outputs to /kaggle/working/preds/.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','terratorch','peft','rasterio'], check=False)
import numpy as np, glob, os, re, json, torch, torch.nn as nn
import rasterio; from rasterio.windows import Window
import segmentation_models_pytorch as smp
from terratorch.models import EncoderDecoderFactory
DEV='cuda' if torch.cuda.is_available() else 'cpu'

# ---- config ----------------------------------------------------------------
TEST_ONE = False               # True = SB100 only (validate first); then set False
STRIDE     = 160                  # window=224, overlap=64 -> 32px context each side
WINDOW     = 224; PAD = (WINDOW-STRIDE)//2
VAL_CSV   = '/kaggle/input/datasets/sheheryarkhan00/yrb-validation-points-clean-land-type/YRB_validation_points_clean_land_type.csv'      # ← your 450-pt CSV
S2_DIR    = '/kaggle/input/datasets/sheheryarkhan00/tgrs-lulc/TGRS_LULC'                     # ← folder with S2_30m_SB*.tif
PART1_DIR = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part1/lobo_results'
PART2_DIR = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part22/lobo_results'                 # ← latest run, saved as dataset
PATCH_DIR = '/kaggle/input/datasets/sheheryarkhan00/yrb-patches'
GID_CKPT  = '/kaggle/input/datasets/sheheryarkhan00/phase1-lulc/Phase1_LULC/checkpoints/best_model.pth'
OUT='/kaggle/working/preds'; os.makedirs(OUT, exist_ok=True)
NUM_CLASSES=7; BANDS=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]
FOLD_ORDER=[100,89] if TEST_ONE else [100,89,108,60,78,59,109,75]

def model_path(sb,tag):
    for d in (PART1_DIR,PART2_DIR):
        p=f'{d}/fold_SB{sb}_{tag}.pth'
        if os.path.exists(p): return p
    return None

# ---- patches: dedup, normalization, BN-recompute pool ----------------------
def sb_of(p): return int(re.search(r'SB(\d+)',os.path.basename(p)).group(1))
pf=sorted(glob.glob(f'{PATCH_DIR}/**/patches_SB*.npz',recursive=True))
seen,pfiles=set(),[]
for f in pf:
    b=os.path.basename(f)
    if b not in seen: seen.add(b); pfiles.append(f)
byf={}
for f in pfiles: byf.setdefault(sb_of(f),[]).append(f)
def load_sb(sb):
    Xs=[]; 
    for f in byf[sb]:
        Xs.append(np.asarray(np.load(f,mmap_mode='r')['X']))
    return np.concatenate(Xs)
samp=np.concatenate([np.asarray(np.load(byf[sb][0],mmap_mode='r')['X'][:150]) for sb in byf])
MEAN=samp.reshape(-1,6,WINDOW*WINDOW).mean((0,2)).astype(np.float32)
STD=(samp.reshape(-1,6,WINDOW*WINDOW).std((0,2))+1e-6).astype(np.float32); del samp
def norm(x): return (x.astype(np.float32)-MEAN[:,None,None])/STD[:,None,None]
def fold_pool(test_sb, cap=1500, seed=0):
    rng=np.random.default_rng(seed); Xs=[]
    for sb in byf:
        if sb==test_sb: continue
        X=load_sb(sb); Xs.append(X[rng.permutation(len(X))[:cap]].copy()); del X
    return np.concatenate(Xs)

# ---- model builders --------------------------------------------------------
LORA={"method":"LORA","replace_qkv":"qkv","peft_config_kwargs":{
      "target_modules":["qkv.q_linear","qkv.v_linear","mlp.fc1","mlp.fc2"],"lora_alpha":16,"r":16}}
def build_prithvi():
    return EncoderDecoderFactory().build_model(task="segmentation",
        backbone="terratorch_prithvi_eo_v2_300",backbone_pretrained=True,backbone_bands=BANDS,
        necks=[{"name":"SelectIndices","indices":[-1]},{"name":"ReshapeTokensToImage"}],
        decoder="UperNetDecoder",decoder_channels=256,num_classes=NUM_CLASSES,head_dropout=0.1,peft_config=LORA)
def build_resnet():
    m=smp.Unet('resnet50',encoder_weights=None,in_channels=6,classes=16)
    g=torch.load(GID_CKPT,map_location='cpu')['model_state']
    w=g['encoder.conv1.weight']; g['encoder.conv1.weight']=torch.cat([w,w.mean(1,keepdim=True).repeat(1,3,1,1)],1)
    m.load_state_dict(g,strict=False)
    m.segmentation_head[0]=nn.Conv2d(m.segmentation_head[0].in_channels,NUM_CLASSES,3,padding=1)
    return m
def load_model(sb,tag):
    if tag=='prithvi':
        m=build_prithvi().to(DEV); m.load_state_dict(torch.load(model_path(sb,'prithvi'),map_location=DEV),strict=False); m.eval(); return m
    m=build_resnet().to(DEV); m.load_state_dict(torch.load(model_path(sb,'resnet'),map_location=DEV),strict=False)
    for mod in m.modules():
        if isinstance(mod,nn.BatchNorm2d): mod.reset_running_stats(); mod.momentum=None
    X=fold_pool(sb); m.train()
    with torch.no_grad():
        for i in range(0,min(60*16,len(X)),16):
            m(torch.tensor(np.stack([norm(x) for x in X[i:i+16]])).to(DEV))
    m.eval(); del X; return m

# ---- predict one S2 tile with center-crop overlapping windows --------------
@torch.no_grad()
def predict_tile(model, s2_path, want_entropy=False):
    with rasterio.open(s2_path) as s:
        H,W=s.height,s.width; prof=s.profile
        cls=np.zeros((H,W),np.uint8); ent=np.zeros((H,W),np.float32) if want_entropy else None
        for top in range(0,H,STRIDE):
            for left in range(0,W,STRIDE):
                x=s.read(window=Window(left-PAD,top-PAD,WINDOW,WINDOW),boundless=True,fill_value=0).astype(np.int16)
                if x.shape!=(6,WINDOW,WINDOW) or (x!=0).mean()<0.02: continue
                xt=torch.tensor(norm(x)).unsqueeze(0).to(DEV)
                o=model(xt); o=o.output if hasattr(o,'output') else o
                p=torch.softmax(o,1)[0]                       # (7,224,224)
                lab=p.argmax(0).cpu().numpy().astype(np.uint8)
                h=min(STRIDE,H-top); w=min(STRIDE,W-left)
                cls[top:top+h,left:left+w]=lab[PAD:PAD+h,PAD:PAD+w]
                if want_entropy:
                    e=-(p.clamp(min=1e-6)*p.clamp(min=1e-6).log()).sum(0).cpu().numpy()
                    ent[top:top+h,left:left+w]=e[PAD:PAD+h,PAD:PAD+w]
        # mask to S2 valid (any band nonzero) -- read band1 footprint
        with rasterio.open(s2_path) as s2:
            valid=s2.read(1)!=0
        cls[~valid]=0
        return cls, ent, prof

def save_raster(arr, prof, path, dtype):
    prof=prof.copy(); prof.update(count=1,dtype=dtype,nodata=0,compress='lzw')
    with rasterio.open(path,'w',**prof) as d: d.write(arr.astype(dtype),1)

# ---- run: per basin, per S2 tile, both models ------------------------------
for sb in FOLD_ORDER:
    tiles=sorted(glob.glob(f'{S2_DIR}/**/S2_30m_SB{sb}*.tif',recursive=True))
    print(f"\nSB{sb}: {len(tiles)} S2 tile(s)")
    for tag in ('resnet','prithvi'):
        done=all(os.path.exists(f'{OUT}/{os.path.basename(t)[:-4]}_{tag}.tif') for t in tiles)
        if done: print(f"  {tag}: already done"); continue
        m=load_model(sb,tag)
        for t in tiles:
            stem=os.path.basename(t)[:-4]
            cls,ent,prof=predict_tile(m, t, want_entropy=(tag=='resnet'))
            save_raster(cls,prof,f'{OUT}/{stem}_{tag}.tif','uint8')
            if ent is not None: save_raster(ent,prof,f'{OUT}/{stem}_entropy.tif','float32')
            print(f"    {tag} {stem}: classes {np.unique(cls[cls>0]).tolist()}")
        del m; torch.cuda.empty_cache()

# ---- validation-point predictions (both models) -> CSV for Fig 5 -----------
import pandas as pd
from pyproj import Transformer
df=pd.read_csv(VAL_CSV)
LABEL_MAP={'Cropland':1,'Forest':2,'Grassland':3,'Bare land':4,'Bare':4,'Water':5,'Built-up':6,'Built':6}
df['true']=df['land_type'].map(LABEL_MAP)
tiles_all=[(sb_of(t),t) for t in glob.glob(f'{S2_DIR}/**/S2_30m_SB*.tif',recursive=True)]
def predict_point(model, lon, lat):
    for sb,t in tiles_all:
        with rasterio.open(t) as s:
            tr=Transformer.from_crs("EPSG:4326",s.crs,always_xy=True); x,y=tr.transform(lon,lat)
            b=s.bounds
            if b.left<=x<=b.right and b.bottom<=y<=b.top:
                r,c=s.index(x,y)
                arr=s.read(window=Window(c-WINDOW//2,r-WINDOW//2,WINDOW,WINDOW),boundless=True,fill_value=0).astype(np.int16)
                if arr.shape!=(6,WINDOW,WINDOW): return sb,0
                with torch.no_grad():
                    o=model(torch.tensor(norm(arr)).unsqueeze(0).to(DEV)); o=o.output if hasattr(o,'output') else o
                    return sb,int(o.argmax(1)[0,WINDOW//2,WINDOW//2].cpu())
    return None,0
if not TEST_ONE:
    for tag in ('resnet','prithvi'):
        df[f'basin']=None; preds=[]
        # group by basin: load held-out model per basin once
        # (assign basin first)
        bs=[]
        for _,r in df.iterrows():
            sb=None
            for s_,t in tiles_all:
                with rasterio.open(t) as s:
                    tr=Transformer.from_crs("EPSG:4326",s.crs,always_xy=True); x,y=tr.transform(r['lon'],r['lat'])
                    b=s.bounds
                    if b.left<=x<=b.right and b.bottom<=y<=b.top and s.read(1,window=Window(*[int(v) for v in s.index(x,y)][::-1]+[1,1]),boundless=True,fill_value=0).flat[0]!=0:
                        sb=s_; break
            bs.append(sb)
        df['basin']=bs
        out=[]
        for sb in sorted([b for b in df['basin'].dropna().unique()]):
            m=load_model(int(sb),tag)
            for i in df.index[df['basin']==sb]:
                _,p=predict_point(m, df.at[i,'lon'], df.at[i,'lat']); df.at[i,f'pred_{tag}']=p
            del m; torch.cuda.empty_cache()
        print(f"{tag} point preds done")
    df.to_csv(f'{OUT}/validation_predictions.csv',index=False)

print("\nDONE. Rasters in /kaggle/working/preds/ — save as a dataset and send me the basin you want rendered first.")
