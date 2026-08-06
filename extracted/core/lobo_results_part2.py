#!/usr/bin/env python3
"""Exact code-cell extraction from lobo-results-part2.ipynb.
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
import glob, json
hits = glob.glob('/kaggle/input/**/fold_SB*.json', recursive=True)
print("found:", hits)
for f in sorted(hits):
    r = json.load(open(f))
    sb = r['test_basin']
    print(f"\nSB{sb}: Prithvi mIoU={r['prithvi']['test_mIoU']:.3f}  "
          f"ResNet mIoU={r['resnet']['test_mIoU']:.3f}")
    print(f"   Prithvi per-class: {r['prithvi']['per_class_IoU']}")
    print(f"   ResNet  per-class: {r['resnet']['per_class_IoU']}")

# %% [notebook cell 2]
# ============================================================================
# RESUME — remaining 4 folds (SB78, SB59, SB109, SB75) with deadlock fix.
# num_workers=0, reads completed folds from RESUME_DIR and skips them.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','terratorch','peft'], check=False)

import numpy as np, glob, os, re, json, torch, torch.nn as nn, gc
import segmentation_models_pytorch as smp
from terratorch.models import EncoderDecoderFactory
torch.manual_seed(42); np.random.seed(42)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- config ----------------------------------------------------------------
PATCH_DIR = '/kaggle/input/datasets/sheheryarkhan00/yrb-patches'
GID_CKPT  = '/kaggle/input/datasets/sheheryarkhan00/phase1-lulc/Phase1_LULC/checkpoints/best_model.pth'
RESUME_DIR = '/kaggle/input/datasets/sheheryarkhan00/lobo-results-part1/lobo_results'
OUT = '/kaggle/working/lobo_results'; os.makedirs(OUT, exist_ok=True)

NUM_CLASSES, IGNORE = 7, 0
BANDS = ["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]
CLASS_NAMES = {1:'Cropland',2:'Forest',3:'Grassland',4:'Bare',5:'Water',6:'Built'}
CAP, VAL_FRAC, MAX_EP, PATIENCE, BATCH, LR = 1500, 0.12, 25, 5, 16, 1e-4
FOLD_ORDER = [78, 59, 109, 75]              # ← only the remaining folds

# ---- index patches by basin (paths only) -----------------------------------
all_npz = sorted(glob.glob(f'{PATCH_DIR}/**/patches_SB*.npz', recursive=True))
seen, files = set(), []
for f in all_npz:
    b=os.path.basename(f)
    if b not in seen: seen.add(b); files.append(f)
def sb_of(p): return int(re.search(r'SB(\d+)', os.path.basename(p)).group(1))
by_sb_files = {}
for f in files: by_sb_files.setdefault(sb_of(f), []).append(f)

def load_sb(sb):
    Xs, Ys = [], []
    for f in by_sb_files[sb]:
        d = np.load(f, mmap_mode='r')
        Xs.append(np.asarray(d['X'])); Ys.append(np.asarray(d['y']))
    return np.concatenate(Xs), np.concatenate(Ys)

sizes = {sb: sum(np.load(f, mmap_mode='r')['X'].shape[0] for f in fl)
         for sb,fl in by_sb_files.items()}
print("basins:", sizes, "| total:", sum(sizes.values()))

# ---- normalization (memory-mapped samples) ---------------------------------
samp = np.concatenate([np.asarray(np.load(by_sb_files[sb][0], mmap_mode='r')['X'][:150])
                       for sb in by_sb_files])
MEAN = samp.reshape(-1,6,224*224).mean((0,2)).astype(np.float32)
STD  = (samp.reshape(-1,6,224*224).std((0,2))+1e-6).astype(np.float32)
del samp
def norm(x): return (x.astype(np.float32)-MEAN[:,None,None])/STD[:,None,None]

class DS(torch.utils.data.Dataset):
    def __init__(s,X,Y): s.X,s.Y=X,Y
    def __len__(s): return len(s.X)
    def __getitem__(s,i): return torch.tensor(norm(s.X[i])), torch.tensor(s.Y[i].astype(np.int64))

def make_fold(test_sb, seed=0):
    rng=np.random.default_rng(seed); Xs,Ys=[],[]
    for sb in by_sb_files:
        if sb==test_sb: continue
        X,Y=load_sb(sb); idx=rng.permutation(len(X))[:CAP]
        Xs.append(X[idx].copy()); Ys.append(Y[idx].copy()); del X,Y
    X=np.concatenate(Xs); Y=np.concatenate(Ys)
    p=rng.permutation(len(X)); X,Y=X[p],Y[p]; nv=int(len(X)*VAL_FRAC)
    Xte,Yte=load_sb(test_sb)
    return X[nv:],Y[nv:],X[:nv],Y[:nv],Xte,Yte

dice=smp.losses.DiceLoss(mode='multiclass',ignore_index=IGNORE)
ce=nn.CrossEntropyLoss(ignore_index=IGNORE)
def lossf(o,y): return 0.5*dice(o,y)+0.5*ce(o,y)

def confmat(model,dl):
    model.eval(); cm=torch.zeros(NUM_CLASSES,NUM_CLASSES,dtype=torch.long)
    with torch.no_grad():
        for x,y in dl:
            o=model(x.to(DEV)); o=o.output if hasattr(o,'output') else o
            p=o.argmax(1).cpu().flatten(); t=y.flatten(); m=t!=IGNORE
            k=t[m]*NUM_CLASSES+p[m]
            cm+=torch.bincount(k,minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
    return cm
def cm_metrics(cm):
    cm=cm.float(); tp=cm.diag(); un=cm.sum(0)+cm.sum(1)-tp
    iou=(tp/un.clamp(min=1)); v=(un>0).clone(); v[IGNORE]=False
    pc={CLASS_NAMES[c]:round(iou[c].item(),3) for c in CLASS_NAMES if un[c]>0}
    acc=(tp.sum()/(cm.sum()-cm[IGNORE].sum()).clamp(min=1)).item()
    return round(iou[v].mean().item(),4), round(acc,4), pc

LORA={"method":"LORA","replace_qkv":"qkv","peft_config_kwargs":{
      "target_modules":["qkv.q_linear","qkv.v_linear","mlp.fc1","mlp.fc2"],"lora_alpha":16,"r":16}}
def build_prithvi():
    return EncoderDecoderFactory().build_model(
        task="segmentation", backbone="terratorch_prithvi_eo_v2_300",
        backbone_pretrained=True, backbone_bands=BANDS,
        necks=[{"name":"SelectIndices","indices":[-1]},{"name":"ReshapeTokensToImage"}],
        decoder="UperNetDecoder", decoder_channels=256,
        num_classes=NUM_CLASSES, head_dropout=0.1, peft_config=LORA)
def build_resnet():
    m=smp.Unet('resnet50',encoder_weights=None,in_channels=6,classes=16)
    g=torch.load(GID_CKPT,map_location='cpu')['model_state']
    w=g['encoder.conv1.weight']; g['encoder.conv1.weight']=torch.cat([w,w.mean(1,keepdim=True).repeat(1,3,1,1)],1)
    m.load_state_dict(g,strict=False)
    m.segmentation_head[0]=nn.Conv2d(m.segmentation_head[0].in_channels,NUM_CLASSES,3,padding=1)
    return m

def train_model(build_fn, tag, tr_dl, val_dl, ckpt_path):
    model=build_fn().to(DEV)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=1e-2)
    sc=torch.cuda.amp.GradScaler(); best=-1; wait=0
    for ep in range(1,MAX_EP+1):
        model.train(); tot=0
        for x,y in tr_dl:
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast():
                o=model(x); o=o.output if hasattr(o,'output') else o; loss=lossf(o,y)
            sc.scale(loss).backward(); sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            sc.step(opt); sc.update(); tot+=loss.item()
        vmi,_,_=cm_metrics(confmat(model,val_dl))
        print(f"    {tag} ep{ep:02d} loss={tot/len(tr_dl):.3f} val_mIoU={vmi:.3f}",flush=True)
        if vmi>best:
            best=vmi; wait=0
            torch.save({n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},ckpt_path)
        else:
            wait+=1
            if wait>=PATIENCE: print(f"    {tag} early stop @ep{ep}",flush=True); break
    model.load_state_dict(torch.load(ckpt_path),strict=False)
    return model, best

def done(sb):
    for d in [OUT, RESUME_DIR]:
        if d and os.path.exists(f'{d}/fold_SB{sb}.json'): return f'{d}/fold_SB{sb}.json'
    return None

for sb in FOLD_ORDER:
    if done(sb):
        print(f"\n=== SB{sb}: already done — skipping ==="); continue
    print(f"\n{'='*60}\nFOLD — hold out SB{sb}\n{'='*60}",flush=True)
    Xtr,Ytr,Xv,Yv,Xte,Yte=make_fold(sb)
    print(f"  train={len(Xtr)} val={len(Xv)} test(SB{sb})={len(Xte)}",flush=True)
    tr_dl=torch.utils.data.DataLoader(DS(Xtr,Ytr),batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True)
    val_dl=torch.utils.data.DataLoader(DS(Xv,Yv),batch_size=BATCH,num_workers=0)
    te_dl=torch.utils.data.DataLoader(DS(Xte,Yte),batch_size=BATCH,num_workers=0)
    res={'test_basin':sb,'n_test':int(len(Xte))}
    for tag,build_fn in [('prithvi',build_prithvi),('resnet',build_resnet)]:
        print(f"  --- {tag} ---",flush=True)
        model,bval=train_model(build_fn,tag,tr_dl,val_dl,f'{OUT}/fold_SB{sb}_{tag}.pth')
        cm=confmat(model,te_dl); mi,acc,pc=cm_metrics(cm)
        res[tag]={'test_mIoU':mi,'test_acc':acc,'per_class_IoU':pc,'best_val_mIoU':round(bval,4),'confusion':cm.tolist()}
        print(f"    {tag} TEST mIoU={mi} acc={acc} per-class={pc}",flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(res,open(f'{OUT}/fold_SB{sb}.json','w'),indent=2)
    print(f"  >>> SB{sb} saved. Prithvi={res['prithvi']['test_mIoU']} ResNet={res['resnet']['test_mIoU']}",flush=True)
    del Xtr,Ytr,Xv,Yv,Xte,Yte; gc.collect()

# ---- aggregate ALL 8 (completed + new) -------------------------------------
print(f"\n{'='*60}\nSUMMARY — all folds\n{'='*60}")
rows=[]
for sb in [100,89,108,60,78,59,109,75]:
    j=done(sb)
    if not j: print(f"SB{sb}: not done"); continue
    r=json.load(open(j))
    rows.append((sb,r['prithvi']['test_mIoU'],r['resnet']['test_mIoU']))
    print(f"SB{sb:>3}: Prithvi={r['prithvi']['test_mIoU']:.3f}  ResNet={r['resnet']['test_mIoU']:.3f}")
if rows:
    pm=np.mean([p for _,p,_ in rows]); rm=np.mean([r for _,_,r in rows])
    print(f"\nMEAN over {len(rows)} folds:  Prithvi={pm:.3f}  ResNet={rm:.3f}  Δ={pm-rm:+.3f}")
