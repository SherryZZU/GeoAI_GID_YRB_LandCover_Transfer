#!/usr/bin/env python3
"""Exact code-cell extraction from accuracy-mapping.ipynb.
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
# ACCURACY ASSESSMENT — spatially-honest (held-out model per point).
# Confusion matrix, per-class user's/producer's accuracy, area-weighted OA,
# for BOTH backbones, with a Low-confidence sensitivity check.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','terratorch','peft','rasterio'], check=False)
import numpy as np, pandas as pd, glob, os, re, json, torch, torch.nn as nn
import rasterio; from rasterio.windows import Window
import segmentation_models_pytorch as smp
from terratorch.models import EncoderDecoderFactory
DEV='cuda' if torch.cuda.is_available() else 'cpu'

# ---- PATHS (edit for Kaggle or Colab) --------------------------------------
VAL_CSV   = '/kaggle/input/datasets/sheheryarkhan00/yrb-validation-points-clean-land-type/YRB_validation_points_clean_land_type.csv'      # ← your 450-pt CSV
S2_DIR    = '/kaggle/input/datasets/sheheryarkhan00/tgrs-lulc/TGRS_LULC'                     # ← folder with S2_30m_SB*.tif
PART1_DIR = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part1/lobo_results'
PART2_DIR = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part22/lobo_results'                 # ← latest run, saved as dataset
PATCH_DIR = '/kaggle/input/datasets/sheheryarkhan00/yrb-patches'
GID_CKPT  = '/kaggle/input/datasets/sheheryarkhan00/phase1-lulc/Phase1_LULC/checkpoints/best_model.pth'

NUM_CLASSES, IGNORE, PATCH = 7, 0, 224
BANDS=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]
CLASS={1:'Cropland',2:'Forest',3:'Grassland',4:'Bare',5:'Water',6:'Built'}
LABEL_MAP={'Cropland':1,'Forest':2,'Grassland':3,'Bare land':4,'Bare':4,
           'Water':5,'Built-up':6,'Built':6}
# area weights from WorldCover functional composition x basin areas (whole basin)
W={1:0.194,2:0.094,3:0.538,4:0.136,5:0.0119,6:0.0263}

# ---- dependency check (fail fast) ------------------------------------------
def model_path(sb,tag):
    for d in (PART1_DIR,PART2_DIR):
        p=f'{d}/fold_SB{sb}_{tag}.pth'
        if os.path.exists(p): return p
    return None
problems=[]
if not glob.glob(VAL_CSV): problems.append(f"VAL_CSV not found: {VAL_CSV}")
s2_tifs=glob.glob(f'{S2_DIR}/**/S2_30m_SB*.tif',recursive=True)
if not s2_tifs: problems.append(f"No S2 GeoTIFFs under {S2_DIR}")
for sb in [59,60,75,78,89,100,108,109]:
    for tag in ('prithvi','resnet'):
        if not model_path(sb,tag): problems.append(f"missing model fold_SB{sb}_{tag}.pth")
if not os.path.exists(GID_CKPT): problems.append("GID_CKPT not found")
if problems:
    print("SETUP INCOMPLETE:"); [print("  -",p) for p in problems]; raise SystemExit
print(f"all inputs present | {len(s2_tifs)} S2 tiles")

# ---- normalization (must match training) -----------------------------------
def sb_of(p): return int(re.search(r'SB(\d+)',os.path.basename(p)).group(1))
pf=sorted(glob.glob(f'{PATCH_DIR}/**/patches_SB*.npz',recursive=True))
seen,pfiles={},[]
for f in pf:
    b=os.path.basename(f)
    if b not in seen: seen[b]=1; pfiles.append(f)
byf={}; [byf.setdefault(sb_of(f),[]).append(f) for f in pfiles]
samp=np.concatenate([np.asarray(np.load(byf[sb][0],mmap_mode='r')['X'][:150]) for sb in byf])
MEAN=samp.reshape(-1,6,PATCH*PATCH).mean((0,2)).astype(np.float32)
STD=(samp.reshape(-1,6,PATCH*PATCH).std((0,2))+1e-6).astype(np.float32); del samp
def norm(x): return (x.astype(np.float32)-MEAN[:,None,None])/STD[:,None,None]

# ---- S2 tile bounds, to find which tile contains a point -------------------
tiles=[]
for t in s2_tifs:
    with rasterio.open(t) as s: tiles.append((sb_of(t),t,s.bounds,s.crs))
from pyproj import Transformer
def find_tile(lon,lat):
    for sb,t,b,crs in tiles:
        tr=Transformer.from_crs("EPSG:4326",crs,always_xy=True)
        x,y=tr.transform(lon,lat)
        if b.left<=x<=b.right and b.bottom<=y<=b.top:
            with rasterio.open(t) as s:
                r,c=s.index(x,y)
                v=s.read(1,window=Window(c,r,1,1),boundless=True,fill_value=0)
                if v.size and int(v.flat[0])!=0: return sb,t,x,y   # valid pixel
    return None,None,None,None

# ---- model builders + loader -----------------------------------------------
LORA={"method":"LORA","replace_qkv":"qkv","peft_config_kwargs":{
      "target_modules":["qkv.q_linear","qkv.v_linear","mlp.fc1","mlp.fc2"],"lora_alpha":16,"r":16}}
def build_prithvi():
    return EncoderDecoderFactory().build_model(task="segmentation",
        backbone="terratorch_prithvi_eo_v2_300",backbone_pretrained=True,backbone_bands=BANDS,
        necks=[{"name":"SelectIndices","indices":[-1]},{"name":"ReshapeTokensToImage"}],
        decoder="UperNetDecoder",decoder_channels=256,num_classes=NUM_CLASSES,
        head_dropout=0.1,peft_config=LORA)
def build_resnet():
    m=smp.Unet('resnet50',encoder_weights=None,in_channels=6,classes=16)
    g=torch.load(GID_CKPT,map_location='cpu')['model_state']
    w=g['encoder.conv1.weight']; g['encoder.conv1.weight']=torch.cat([w,w.mean(1,keepdim=True).repeat(1,3,1,1)],1)
    m.load_state_dict(g,strict=False)
    m.segmentation_head[0]=nn.Conv2d(m.segmentation_head[0].in_channels,NUM_CLASSES,3,padding=1)
    return m
def load_model(sb,tag):
    m=(build_prithvi() if tag=='prithvi' else build_resnet()).to(DEV)
    m.load_state_dict(torch.load(model_path(sb,tag),map_location=DEV),strict=False)
    m.eval(); return m

@torch.no_grad()
def predict_point(model,tif,x,y):
    with rasterio.open(tif) as s:
        r,c=s.index(x,y)
        patch=s.read(window=Window(c-PATCH//2,r-PATCH//2,PATCH,PATCH),
                     boundless=True,fill_value=0).astype(np.int16)   # (6,224,224)
    xt=torch.tensor(norm(patch)).unsqueeze(0).to(DEV)
    o=model(xt); o=o.output if hasattr(o,'output') else o
    return int(o.argmax(1)[0,PATCH//2,PATCH//2].cpu())

# ---- load points, assign basin ---------------------------------------------
df=pd.read_csv(VAL_CSV)
df['true']=df['land_type'].map(LABEL_MAP)
assert df['true'].notna().all(), f"unmapped labels: {df[df['true'].isna()]['land_type'].unique()}"
if 'confidence' not in df.columns: df['confidence']='High'
basins=[]; xs=[]; ys=[]; tifs=[]
for _,row in df.iterrows():
    sb,t,x,y=find_tile(row['lon'],row['lat'])
    basins.append(sb); xs.append(x); ys.append(y); tifs.append(t)
df['basin']=basins; df['x']=xs; df['y']=ys; df['tif']=tifs
miss=df['basin'].isna().sum()
print(f"points assigned to a basin: {len(df)-miss}/{len(df)} (dropped {miss} outside valid areas)")
df=df.dropna(subset=['basin']).copy(); df['basin']=df['basin'].astype(int)

# ---- predict (group by basin, load each model once) ------------------------
for tag in ('prithvi','resnet'): df[f'pred_{tag}']=np.nan
for sb in sorted(df['basin'].unique()):
    idx=df.index[df['basin']==sb]
    for tag in ('prithvi','resnet'):
        m=load_model(sb,tag)
        for i in idx:
            df.at[i,f'pred_{tag}']=predict_point(m,df.at[i,'tif'],df.at[i,'x'],df.at[i,'y'])
        del m; torch.cuda.empty_cache()
    print(f"  SB{sb}: predicted {len(idx)} points")

# ---- metrics ---------------------------------------------------------------
def assess(sub,tag):
    cm=np.zeros((6,6),int)   # rows=true(1-6), cols=pred
    for _,r in sub.iterrows():
        t,p=int(r['true']),int(r[f'pred_{tag}'])
        if 1<=p<=6: cm[t-1,p-1]+=1
    tp=np.diag(cm); tot=cm.sum()
    OA=tp.sum()/tot if tot else 0
    PA={CLASS[c+1]:(tp[c]/cm[c].sum() if cm[c].sum() else np.nan) for c in range(6)}  # recall
    UA={CLASS[c+1]:(tp[c]/cm[:,c].sum() if cm[:,c].sum() else np.nan) for c in range(6)}# precision
    OA_area=sum(W[c+1]*(PA[CLASS[c+1]] if not np.isnan(PA[CLASS[c+1]]) else 0) for c in range(6))
    bal=np.nanmean(list(PA.values()))
    return cm,OA,OA_area,bal,UA,PA

print("\n"+"="*70+"\nACCURACY ASSESSMENT\n"+"="*70)
results={}
for label,sub in [('ALL 450',df),('High+Med (drop Low-conf)',df[df['confidence']!='Low'])]:
    print(f"\n########## {label}  (n={len(sub)}) ##########")
    for tag in ('prithvi','resnet'):
        cm,OA,OAa,bal,UA,PA=assess(sub,tag)
        print(f"\n  --- {tag.upper()} ---")
        print(f"  Overall accuracy (raw)          : {OA:.3f}")
        print(f"  Overall accuracy (area-weighted): {OAa:.3f}")
        print(f"  Balanced accuracy (mean recall) : {bal:.3f}")
        print(f"  {'class':10s} {'UserAcc':>8} {'ProdAcc':>8}")
        for c in CLASS.values(): print(f"  {c:10s} {UA[c]:>8.3f} {PA[c]:>8.3f}")
        if label=='ALL 450':
            results[tag]={'OA':round(OA,3),'OA_area':round(OAa,3),'balanced':round(bal,3),
                          'user_acc':{k:round(v,3) for k,v in UA.items()},
                          'prod_acc':{k:round(v,3) for k,v in PA.items()},
                          'confusion':cm.tolist()}
            print("  confusion (rows=true, cols=pred):"); print("   ",cm.tolist())

json.dump(results,open('/kaggle/working/accuracy_assessment.json','w'),indent=2)
df.to_csv('/kaggle/working/validation_with_predictions.csv',index=False)
print("\nsaved -> accuracy_assessment.json + validation_with_predictions.csv")

# %% [notebook cell 2]
# 1. Did the ResNet checkpoint actually save the encoder, and does it load cleanly?
import torch
sd = torch.load(model_path(89,'resnet'), map_location='cpu')
print("keys in ckpt:", len(sd))
print("has encoder keys:", any(k.startswith('encoder') for k in sd))
print("has seg head:", any('segmentation_head' in k for k in sd))

m = build_resnet()
missing, unexpected = m.load_state_dict(sd, strict=False)
print("MISSING (in model, not in ckpt):", len(missing), missing[:6])
print("UNEXPECTED:", len(unexpected), unexpected[:6])

# %% [notebook cell 3]
# ============================================================================
# ACCURACY ASSESSMENT (BN-fixed) — spatially-honest, held-out model per point.
# ResNet BatchNorm running stats are recomputed per fold (they weren't saved).
# Outputs: confusion matrix, user's/producer's accuracy, raw + area-weighted OA,
# for BOTH backbones, with a Low-confidence sensitivity check.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','terratorch','peft','rasterio','pyproj'], check=False)
import numpy as np, pandas as pd, glob, os, re, json, torch, torch.nn as nn
import rasterio; from rasterio.windows import Window
import segmentation_models_pytorch as smp
from terratorch.models import EncoderDecoderFactory
from pyproj import Transformer
DEV='cuda' if torch.cuda.is_available() else 'cpu'

# ---- PATHS (edit for Kaggle or Colab) --------------------------------------
VAL_CSV   = '/kaggle/input/datasets/sheheryarkhan00/yrb-validation-points-clean-land-type/YRB_validation_points_clean_land_type.csv'      # ← your 450-pt CSV
S2_DIR    = '/kaggle/input/datasets/sheheryarkhan00/tgrs-lulc/TGRS_LULC'                     # ← folder with S2_30m_SB*.tif
PART1_DIR = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part1/lobo_results'
PART2_DIR = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part22/lobo_results'                 # ← latest run, saved as dataset
PATCH_DIR = '/kaggle/input/datasets/sheheryarkhan00/yrb-patches'
GID_CKPT  = '/kaggle/input/datasets/sheheryarkhan00/phase1-lulc/Phase1_LULC/checkpoints/best_model.pth'

NUM_CLASSES, IGNORE, PATCH = 7, 0, 224
CAP, VAL_FRAC = 1500, 0.12
BANDS=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]
CLASS={1:'Cropland',2:'Forest',3:'Grassland',4:'Bare',5:'Water',6:'Built'}
LABEL_MAP={'Cropland':1,'Forest':2,'Grassland':3,'Bare land':4,'Bare':4,
           'Water':5,'Built-up':6,'Built':6}
W={1:0.194,2:0.094,3:0.538,4:0.136,5:0.0119,6:0.0263}  # area weights (WorldCover comp x basin area)

# ---- dependency check (fail fast) ------------------------------------------
def model_path(sb,tag):
    for d in (PART1_DIR,PART2_DIR):
        p=f'{d}/fold_SB{sb}_{tag}.pth'
        if os.path.exists(p): return p
    return None
problems=[]
if not glob.glob(VAL_CSV): problems.append(f"VAL_CSV not found: {VAL_CSV}")
s2_tifs=glob.glob(f'{S2_DIR}/**/S2_30m_SB*.tif',recursive=True)
if not s2_tifs: problems.append(f"No S2 GeoTIFFs under {S2_DIR}")
for sb in [59,60,75,78,89,100,108,109]:
    for tag in ('prithvi','resnet'):
        if not model_path(sb,tag): problems.append(f"missing model fold_SB{sb}_{tag}.pth")
if not os.path.exists(GID_CKPT): problems.append("GID_CKPT not found")
if problems:
    print("SETUP INCOMPLETE:"); [print("  -",p) for p in problems]; raise SystemExit
print(f"all inputs present | {len(s2_tifs)} S2 tiles")

# ---- patch index + on-demand basin loader ----------------------------------
def sb_of(p): return int(re.search(r'SB(\d+)',os.path.basename(p)).group(1))
pf=sorted(glob.glob(f'{PATCH_DIR}/**/patches_SB*.npz',recursive=True))
seen,pfiles=set(),[]
for f in pf:
    b=os.path.basename(f)
    if b not in seen: seen.add(b); pfiles.append(f)
by_sb_files={}
for f in pfiles: by_sb_files.setdefault(sb_of(f),[]).append(f)
def load_sb(sb):
    Xs,Ys=[],[]
    for f in by_sb_files[sb]:
        d=np.load(f,mmap_mode='r'); Xs.append(np.asarray(d['X'])); Ys.append(np.asarray(d['y']))
    return np.concatenate(Xs),np.concatenate(Ys)

# ---- normalization (must match training) -----------------------------------
samp=np.concatenate([np.asarray(np.load(by_sb_files[sb][0],mmap_mode='r')['X'][:150]) for sb in by_sb_files])
MEAN=samp.reshape(-1,6,PATCH*PATCH).mean((0,2)).astype(np.float32)
STD=(samp.reshape(-1,6,PATCH*PATCH).std((0,2))+1e-6).astype(np.float32); del samp
def norm(x): return (x.astype(np.float32)-MEAN[:,None,None])/STD[:,None,None]

class DS(torch.utils.data.Dataset):
    def __init__(s,X,Y): s.X,s.Y=X,Y
    def __len__(s): return len(s.X)
    def __getitem__(s,i): return torch.tensor(norm(s.X[i])), torch.tensor(s.Y[i].astype(np.int64))

def make_fold(test_sb, seed=0):       # same logic as training (for BN recompute pool)
    rng=np.random.default_rng(seed); Xs,Ys=[],[]
    for sb in by_sb_files:
        if sb==test_sb: continue
        X,Y=load_sb(sb); idx=rng.permutation(len(X))[:CAP]
        Xs.append(X[idx].copy()); Ys.append(Y[idx].copy()); del X,Y
    X=np.concatenate(Xs); Y=np.concatenate(Ys)
    p=rng.permutation(len(X)); X,Y=X[p],Y[p]; nv=int(len(X)*VAL_FRAC)
    Xte,Yte=load_sb(test_sb)
    return X[nv:],Y[nv:],X[:nv],Y[:nv],Xte,Yte

# ---- S2 tile bounds -> find tile containing a point ------------------------
tiles=[]
for t in s2_tifs:
    with rasterio.open(t) as s: tiles.append((sb_of(t),t,s.bounds,s.crs))
def find_tile(lon,lat):
    for sb,t,b,crs in tiles:
        tr=Transformer.from_crs("EPSG:4326",crs,always_xy=True); x,y=tr.transform(lon,lat)
        if b.left<=x<=b.right and b.bottom<=y<=b.top:
            with rasterio.open(t) as s:
                r,c=s.index(x,y)
                v=s.read(1,window=Window(c,r,1,1),boundless=True,fill_value=0)
                if v.size and int(v.flat[0])!=0: return sb,t,x,y
    return None,None,None,None

# ---- model builders --------------------------------------------------------
LORA={"method":"LORA","replace_qkv":"qkv","peft_config_kwargs":{
      "target_modules":["qkv.q_linear","qkv.v_linear","mlp.fc1","mlp.fc2"],"lora_alpha":16,"r":16}}
def build_prithvi():
    return EncoderDecoderFactory().build_model(task="segmentation",
        backbone="terratorch_prithvi_eo_v2_300",backbone_pretrained=True,backbone_bands=BANDS,
        necks=[{"name":"SelectIndices","indices":[-1]},{"name":"ReshapeTokensToImage"}],
        decoder="UperNetDecoder",decoder_channels=256,num_classes=NUM_CLASSES,
        head_dropout=0.1,peft_config=LORA)
def build_resnet():
    m=smp.Unet('resnet50',encoder_weights=None,in_channels=6,classes=16)
    g=torch.load(GID_CKPT,map_location='cpu')['model_state']
    w=g['encoder.conv1.weight']; g['encoder.conv1.weight']=torch.cat([w,w.mean(1,keepdim=True).repeat(1,3,1,1)],1)
    m.load_state_dict(g,strict=False)
    m.segmentation_head[0]=nn.Conv2d(m.segmentation_head[0].in_channels,NUM_CLASSES,3,padding=1)
    return m

def load_prithvi(sb):
    m=build_prithvi().to(DEV)
    m.load_state_dict(torch.load(model_path(sb,'prithvi'),map_location=DEV),strict=False)
    m.eval(); return m

def load_resnet_fixed(sb):           # recompute BN running stats (not saved in ckpt)
    m=build_resnet().to(DEV)
    m.load_state_dict(torch.load(model_path(sb,'resnet'),map_location=DEV),strict=False)
    for mod in m.modules():
        if isinstance(mod,nn.BatchNorm2d): mod.reset_running_stats(); mod.momentum=None
    Xtr,Ytr,_,_,_,_=make_fold(sb)
    dl=torch.utils.data.DataLoader(DS(Xtr,Ytr),batch_size=16,shuffle=True,num_workers=0)
    m.train()
    with torch.no_grad():
        for i,(x,_) in enumerate(dl):
            m(x.to(DEV))
            if i>=60: break
    m.eval(); del Xtr,Ytr; return m

@torch.no_grad()
def predict_point(model,tif,x,y):
    with rasterio.open(tif) as s:
        r,c=s.index(x,y)
        patch=s.read(window=Window(c-PATCH//2,r-PATCH//2,PATCH,PATCH),
                     boundless=True,fill_value=0).astype(np.int16)
    xt=torch.tensor(norm(patch)).unsqueeze(0).to(DEV)
    o=model(xt); o=o.output if hasattr(o,'output') else o
    return int(o.argmax(1)[0,PATCH//2,PATCH//2].cpu())

# ---- load points, assign basin ---------------------------------------------
df=pd.read_csv(VAL_CSV)
df['true']=df['land_type'].map(LABEL_MAP)
assert df['true'].notna().all(), f"unmapped: {df[df['true'].isna()]['land_type'].unique()}"
if 'confidence' not in df.columns: df['confidence']='High'
b=[];xs=[];ys=[];tf=[]
for _,row in df.iterrows():
    sb,t,x,y=find_tile(row['lon'],row['lat']); b.append(sb);xs.append(x);ys.append(y);tf.append(t)
df['basin']=b; df['x']=xs; df['y']=ys; df['tif']=tf
miss=df['basin'].isna().sum()
print(f"points assigned: {len(df)-miss}/{len(df)} (dropped {miss} outside valid areas)")
df=df.dropna(subset=['basin']).copy(); df['basin']=df['basin'].astype(int)

# ---- predict (per basin, load each model once) -----------------------------
df['pred_prithvi']=np.nan; df['pred_resnet']=np.nan
for sb in sorted(df['basin'].unique()):
    idx=df.index[df['basin']==sb]
    mp=load_prithvi(sb)
    for i in idx: df.at[i,'pred_prithvi']=predict_point(mp,df.at[i,'tif'],df.at[i,'x'],df.at[i,'y'])
    del mp; torch.cuda.empty_cache()
    mr=load_resnet_fixed(sb)
    for i in idx: df.at[i,'pred_resnet']=predict_point(mr,df.at[i,'tif'],df.at[i,'x'],df.at[i,'y'])
    del mr; torch.cuda.empty_cache()
    print(f"  SB{sb}: predicted {len(idx)} points")

# ---- metrics ---------------------------------------------------------------
def assess(sub,tag):
    cm=np.zeros((6,6),int)
    for _,r in sub.iterrows():
        t,p=int(r['true']),int(r[f'pred_{tag}'])
        if 1<=p<=6: cm[t-1,p-1]+=1
    tp=np.diag(cm); tot=cm.sum()
    OA=tp.sum()/tot if tot else 0
    PA={CLASS[c+1]:(tp[c]/cm[c].sum() if cm[c].sum() else np.nan) for c in range(6)}
    UA={CLASS[c+1]:(tp[c]/cm[:,c].sum() if cm[:,c].sum() else np.nan) for c in range(6)}
    OAa=sum(W[c+1]*(PA[CLASS[c+1]] if not np.isnan(PA[CLASS[c+1]]) else 0) for c in range(6))
    bal=np.nanmean(list(PA.values()))
    return cm,OA,OAa,bal,UA,PA

print("\n"+"="*70+"\nACCURACY ASSESSMENT (BN-fixed)\n"+"="*70)
results={}
for label,sub in [('ALL',df),('High+Med (drop Low-conf)',df[df['confidence']!='Low'])]:
    print(f"\n########## {label}  (n={len(sub)}) ##########")
    for tag in ('prithvi','resnet'):
        cm,OA,OAa,bal,UA,PA=assess(sub,tag)
        print(f"\n  --- {tag.upper()} ---")
        print(f"  Overall accuracy (raw)          : {OA:.3f}")
        print(f"  Overall accuracy (area-weighted): {OAa:.3f}")
        print(f"  Balanced accuracy (mean recall) : {bal:.3f}")
        print(f"  {'class':10s} {'UserAcc':>8} {'ProdAcc':>8}")
        for c in CLASS.values(): print(f"  {c:10s} {UA[c]:>8.3f} {PA[c]:>8.3f}")
        if label=='ALL':
            results[tag]={'OA':round(OA,3),'OA_area':round(OAa,3),'balanced':round(bal,3),
                          'user_acc':{k:round(v,3) for k,v in UA.items()},
                          'prod_acc':{k:round(v,3) for k,v in PA.items()},'confusion':cm.tolist()}
            print("  confusion (rows=true 1-6, cols=pred):"); print("   ",cm.tolist())

json.dump(results,open('/kaggle/working/accuracy_assessment.json','w'),indent=2)
df.to_csv('/kaggle/working/validation_with_predictions.csv',index=False)
print("\nsaved -> accuracy_assessment.json + validation_with_predictions.csv")

# %% [notebook cell 4]
# ============================================================================
# FIGURE 4 — generate the functional LULC map for one basin (held-out model).
# Spatially honest: uses the fold model that held out this basin.
# Run on Kaggle/Colab with the same inputs as the accuracy assessment.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','terratorch','peft','rasterio'], check=False)
import numpy as np, glob, os, re, torch, torch.nn as nn
import rasterio; from rasterio.windows import Window
import segmentation_models_pytorch as smp
from terratorch.models import EncoderDecoderFactory
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
DEV='cuda' if torch.cuda.is_available() else 'cpu'

# ---- config ----------------------------------------------------------------
MAP_BASIN = 100        # single-tile basins: 100, 108, 60, 78, 59 (100=orchard story; 108=most cropland)
# ---- PATHS (edit for Kaggle or Colab) --------------------------------------
VAL_CSV   = '/kaggle/input/datasets/sheheryarkhan00/yrb-validation-points-clean-land-type/YRB_validation_points_clean_land_type.csv'      # ← your 450-pt CSV
S2_DIR    = '/kaggle/input/datasets/sheheryarkhan00/tgrs-lulc/TGRS_LULC'                     # ← folder with S2_30m_SB*.tif
PART1_DIR = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part1/lobo_results'
PART2_DIR = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part22/lobo_results'                 # ← latest run, saved as dataset
PATCH_DIR = '/kaggle/input/datasets/sheheryarkhan00/yrb-patches'
GID_CKPT  = '/kaggle/input/datasets/sheheryarkhan00/phase1-lulc/Phase1_LULC/checkpoints/best_model.pth'
PATCH, CAP, VAL_FRAC = 224, 1500, 0.12
BANDS=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]

def model_path(sb,tag):
    for d in (PART1_DIR,PART2_DIR):
        p=f'{d}/fold_SB{sb}_{tag}.pth'
        if os.path.exists(p): return p
    return None

# ---- patches (for normalization + BN recompute) ----------------------------
def sb_of(p): return int(re.search(r'SB(\d+)',os.path.basename(p)).group(1))
pf=sorted(glob.glob(f'{PATCH_DIR}/**/patches_SB*.npz',recursive=True))
seen,pfiles=set(),[]
for f in pf:
    b=os.path.basename(f)
    if b not in seen: seen.add(b); pfiles.append(f)
byf={}
for f in pfiles: byf.setdefault(sb_of(f),[]).append(f)
def load_sb(sb):
    Xs,Ys=[],[]
    for f in byf[sb]:
        d=np.load(f,mmap_mode='r'); Xs.append(np.asarray(d['X'])); Ys.append(np.asarray(d['y']))
    return np.concatenate(Xs),np.concatenate(Ys)
samp=np.concatenate([np.asarray(np.load(byf[sb][0],mmap_mode='r')['X'][:150]) for sb in byf])
MEAN=samp.reshape(-1,6,PATCH*PATCH).mean((0,2)).astype(np.float32)
STD=(samp.reshape(-1,6,PATCH*PATCH).std((0,2))+1e-6).astype(np.float32); del samp
def norm(x): return (x.astype(np.float32)-MEAN[:,None,None])/STD[:,None,None]

# ---- rebuild ResNet + recompute BN (held-out fold) -------------------------
def build_resnet():
    m=smp.Unet('resnet50',encoder_weights=None,in_channels=6,classes=16)
    g=torch.load(GID_CKPT,map_location='cpu')['model_state']
    w=g['encoder.conv1.weight']; g['encoder.conv1.weight']=torch.cat([w,w.mean(1,keepdim=True).repeat(1,3,1,1)],1)
    m.load_state_dict(g,strict=False)
    m.segmentation_head[0]=nn.Conv2d(m.segmentation_head[0].in_channels,7,3,padding=1)
    return m
def make_fold(test_sb, seed=0):
    rng=np.random.default_rng(seed); Xs,Ys=[],[]
    for sb in byf:
        if sb==test_sb: continue
        X,Y=load_sb(sb); idx=rng.permutation(len(X))[:CAP]
        Xs.append(X[idx].copy()); del X,Y
    return np.concatenate(Xs)
print(f"loading held-out ResNet for SB{MAP_BASIN} + recomputing BN...")
model=build_resnet().to(DEV)
model.load_state_dict(torch.load(model_path(MAP_BASIN,'resnet'),map_location=DEV),strict=False)
for mod in model.modules():
    if isinstance(mod,nn.BatchNorm2d): mod.reset_running_stats(); mod.momentum=None
Xtr=make_fold(MAP_BASIN)
model.train()
with torch.no_grad():
    for i in range(0,min(60*16,len(Xtr)),16):
        xb=torch.tensor(np.stack([norm(x) for x in Xtr[i:i+16]])).to(DEV); model(xb)
model.eval(); del Xtr

# ---- find the basin S2 tile, predict in sliding windows --------------------
tif=[t for t in glob.glob(f'{S2_DIR}/**/S2_30m_SB{MAP_BASIN}*.tif',recursive=True)]
assert len(tif)==1, f"expected 1 tile for SB{MAP_BASIN}, found {len(tif)} (multi-tile basin; pick a single-tile one)"
tif=tif[0]
with rasterio.open(tif) as src:
    H,W=src.height,src.width; prof=src.profile
    out=np.zeros((H,W),np.uint8)
    print(f"mapping {H}x{W} in {PATCH}px tiles...")
    for top in range(0,H,PATCH):
        for left in range(0,W,PATCH):
            win=Window(left,top,PATCH,PATCH)
            x=src.read(window=win,boundless=True,fill_value=0).astype(np.int16)
            if x.shape!=(6,PATCH,PATCH): continue
            if (x!=0).mean()<0.05: continue                 # skip empty
            with torch.no_grad():
                xt=torch.tensor(norm(x)).unsqueeze(0).to(DEV)
                o=model(xt); o=o.output if hasattr(o,'output') else o
                pred=o.argmax(1)[0].cpu().numpy().astype(np.uint8)
            h=min(PATCH,H-top); w=min(PATCH,W-left)
            out[top:top+h,left:left+w]=pred[:h,:w]
        print(f"  row {top}/{H}",end="\r")

# ---- save georeferenced GeoTIFF + colorized PNG ----------------------------
prof.update(count=1,dtype='uint8',nodata=0)
with rasterio.open(f'/kaggle/working/LULC_SB{MAP_BASIN}.tif','w',**prof) as dst:
    dst.write(out,1)

# 6-class palette (0=nodata transparent)
COLORS=['#000000','#E8B84B','#2E7D32','#9CCC65','#B59A82','#4A90D9','#C0392B']
cmap=ListedColormap(COLORS)
# downsample for a viewable PNG
step=max(1,max(H,W)//2500)
small=out[::step,::step]
fig,ax=plt.subplots(figsize=(8,8*small.shape[0]/small.shape[1]))
ax.imshow(small,cmap=cmap,vmin=0,vmax=6,interpolation='nearest')
ax.axis('off')
from matplotlib.patches import Patch
names=['Cropland','Forest','Grassland','Bare','Water','Built']
ax.legend(handles=[Patch(color=COLORS[i+1],label=names[i]) for i in range(6)],
          loc='lower right',fontsize=8,framealpha=0.9)
ax.set_title(f'Functional LULC — SB{MAP_BASIN} (GID ResNet-50, held-out)',fontsize=11)
fig.savefig(f'/kaggle/working/fig4_LULC_SB{MAP_BASIN}.png',dpi=200,bbox_inches='tight')
print(f"\nsaved -> LULC_SB{MAP_BASIN}.tif (georeferenced) + fig4_LULC_SB{MAP_BASIN}.png")

# %% [notebook cell 5]
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
TEST_ONE   = True                 # True = SB100 only (validate first); then set False
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
