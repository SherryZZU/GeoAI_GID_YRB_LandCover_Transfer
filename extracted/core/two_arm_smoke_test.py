#!/usr/bin/env python3
"""Exact code-cell extraction from two-arm-smoke-test.ipynb.
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
# SMOKE TEST — validate both arms before the 8-fold commit.
# Prithvi-EO-2.0 (LoRA) + GID ResNet (6-band, 7-class adapted). 1 fold, 2 epochs.
# ============================================================================
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','terratorch','peft'], check=False)

import numpy as np, glob, os, re, torch, torch.nn as nn
import segmentation_models_pytorch as smp
from terratorch.models import EncoderDecoderFactory
torch.manual_seed(42); np.random.seed(42)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- paths -----------------------------------------------------------------
PATCH_DIR = '/kaggle/input/datasets/sheheryarkhan00/yrb-patches'
GID_CKPT  = '/kaggle/input/datasets/sheheryarkhan00/phase1-lulc/Phase1_LULC/checkpoints/best_model.pth'
NUM_CLASSES, IGNORE = 7, 0          # labels: 0=nodata(ignore), 1-6=functional
BANDS = ["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]

# ---- load patches, DEDUP by filename, group by basin -----------------------
all_npz = sorted(glob.glob(f'{PATCH_DIR}/**/patches_SB*.npz', recursive=True))
seen, files = set(), []
for f in all_npz:
    b = os.path.basename(f)
    if b not in seen: seen.add(b); files.append(f)
print(f"{len(all_npz)} paths -> {len(files)} unique files")

def sb_of(p): return int(re.search(r'SB(\d+)', os.path.basename(p)).group(1))
by_sb = {}
for f in files:
    d = np.load(f); sb = sb_of(f)
    by_sb.setdefault(sb,[]).append((d['X'], d['y']))
by_sb = {sb: (np.concatenate([x for x,_ in v]), np.concatenate([y for _,y in v]))
         for sb,v in by_sb.items()}
print("basins:", {sb: x.shape[0] for sb,(x,_) in by_sb.items()})
print("total patches:", sum(x.shape[0] for x,_ in by_sb.values()))

# ---- smoke fold: test=SB100, train=SB59+SB108 (small subsets) --------------
Xtr = np.concatenate([by_sb[59][0][:300], by_sb[108][0][:300]])
Ytr = np.concatenate([by_sb[59][1][:300], by_sb[108][1][:300]])
Xte, Yte = by_sb[100][0][:150], by_sb[100][1][:150]

# ---- per-band z-score normalization (same for both models) -----------------
mean = Xtr.reshape(-1,6,224*224).mean(axis=(0,2)).astype(np.float32)
std  = (Xtr.reshape(-1,6,224*224).std(axis=(0,2))+1e-6).astype(np.float32)
def norm(x): return (x.astype(np.float32)-mean[:,None,None])/std[:,None,None]

class DS(torch.utils.data.Dataset):
    def __init__(s,X,Y): s.X,s.Y=X,Y
    def __len__(s): return len(s.X)
    def __getitem__(s,i):
        return torch.tensor(norm(s.X[i])), torch.tensor(s.Y[i].astype(np.int64))
tr_dl = torch.utils.data.DataLoader(DS(Xtr,Ytr), batch_size=8, shuffle=True)
te_dl = torch.utils.data.DataLoader(DS(Xte,Yte), batch_size=8)

# ---- mIoU helper -----------------------------------------------------------
def miou(model, dl):
    model.eval(); cm=torch.zeros(NUM_CLASSES,NUM_CLASSES,dtype=torch.long)
    with torch.no_grad():
        for x,y in dl:
            out=model(x.to(DEV)); out=out.output if hasattr(out,'output') else out
            p=out.argmax(1).cpu().flatten(); t=y.flatten(); m=t!=IGNORE
            k=t[m]*NUM_CLASSES+p[m]
            cm+=torch.bincount(k,minlength=NUM_CLASSES**2).reshape(NUM_CLASSES,NUM_CLASSES)
    tp=cm.diag().float(); un=cm.sum(0)+cm.sum(1)-cm.diag()
    iou=tp/un.clamp(min=1); v=(un>0); v[IGNORE]=False
    return iou[v].mean().item()

def train2(model, tag):
    model.to(DEV)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    lossf=nn.CrossEntropyLoss(ignore_index=IGNORE); sc=torch.cuda.amp.GradScaler()
    for ep in range(2):
        model.train(); tot=0
        for x,y in tr_dl:
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast():
                out=model(x); out=out.output if hasattr(out,'output') else out
                loss=lossf(out,y)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); tot+=loss.item()
        print(f"  {tag} ep{ep+1} loss={tot/len(tr_dl):.3f}")
    print(f"  {tag} test mIoU={miou(model,te_dl):.3f}")

# ---- ARM 1: Prithvi-EO-2.0 + LoRA (verified IBM config) --------------------
print("\n[Prithvi-EO-2.0 + LoRA]")
LORA = {"method":"LORA","replace_qkv":"qkv",
        "peft_config_kwargs":{"target_modules":["qkv.q_linear","qkv.v_linear","mlp.fc1","mlp.fc2"],
                              "lora_alpha":16,"r":16}}
prithvi = EncoderDecoderFactory().build_model(
    task="segmentation", backbone="terratorch_prithvi_eo_v2_300",
    backbone_pretrained=True, backbone_bands=BANDS,
    necks=[{"name":"SelectIndices","indices":[-1]},{"name":"ReshapeTokensToImage"}],
    decoder="UperNetDecoder", decoder_channels=256,
    num_classes=NUM_CLASSES, head_dropout=0.1, peft_config=LORA)
tp=sum(p.numel() for p in prithvi.parameters() if p.requires_grad)
print(f"  trainable params: {tp/1e6:.2f}M (LoRA -> small fraction of 310M)")
train2(prithvi, "Prithvi")

# ---- ARM 2: GID ResNet, 6-band inflated, 7-class head ----------------------
print("\n[GID ResNet-50 UNet — adapted]")
resnet = smp.Unet('resnet50', encoder_weights=None, in_channels=6, classes=16)
gid = torch.load(GID_CKPT, map_location='cpu')['model_state']
w = gid['encoder.conv1.weight']                       # [64,3,7,7] RGB
gid['encoder.conv1.weight'] = torch.cat(
    [w, w.mean(1,keepdim=True).repeat(1,3,1,1)], dim=1)   # inflate 3->6
resnet.load_state_dict(gid, strict=False)
resnet.segmentation_head[0] = nn.Conv2d(
    resnet.segmentation_head[0].in_channels, NUM_CLASSES, 3, padding=1)  # head 16->7
print("  GID weights loaded (conv1 inflated 3->6, head 16->7)")
train2(resnet, "ResNet")

print("\nSMOKE TEST DONE — both arms should show decreasing loss + a test mIoU.")
