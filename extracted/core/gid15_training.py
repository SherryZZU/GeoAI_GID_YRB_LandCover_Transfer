#!/usr/bin/env python3
"""Exact code-cell extraction from gid15-training.ipynb.
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
# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — GID-15 Transfer Learning (robust, anti-overfit/underfit)
# Cell 1/3 — Setup
# ════════════════════════════════════════════════════════════════════════════
import subprocess
subprocess.run(['pip', 'install', 'segmentation-models-pytorch',
                'albumentations', '-q'], check=True)

import os, random, warnings
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
BASE        = Path('/kaggle/input/datasets/sheheryarkhan00/gid15-yrb')
OUT_DIR     = Path('/kaggle/working/Phase1_LULC')
CKPT_DIR    = OUT_DIR / 'checkpoints'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE  = 512
BATCH_SIZE  = 6           # AMP lets us fit 6 on T4 16GB
NUM_CLASSES = 16          # indices 0–15, 0 = unlabeled (ignored)
IGNORE_IDX  = 0
SEED        = 42

# Warmup = train decoder only (encoder frozen). Finetune = unfreeze all.
WARMUP_EPOCHS   = 8
FINETUNE_EPOCHS = 60
PATIENCE        = 12      # early stop if val mIoU doesn't improve

CLASS_NAMES = {
    0:'Unlabeled',       1:'Industrial area',  2:'Paddy field',
    3:'Irrigated farmland', 4:'Dry cropland',  5:'Garden land',
    6:'Arbor forest',    7:'Shrub forest',     8:'Natural meadow',
    9:'Artificial meadow', 10:'River',         11:'Urban residential',
    12:'Lake',           13:'Pond',            14:'Fish pond',
    15:'Sea',
}

def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# %% [notebook cell 2]
# ════════════════════════════════════════════════════════════════════════════
# Cell 2/3 — Dataset, class weights, combined loss, mIoU metric  (VAL-MASK FIX)
# ════════════════════════════════════════════════════════════════════════════

def load_mask(path):
    """GID-15 masks: the .png is index-coded (0-15) and CORRECT.
    The .tif is RGB color-coded and must never be used directly.
    Always read the .png version."""
    path = Path(path)
    png  = path.with_suffix('.png')          # force the index-coded PNG
    src  = png if png.exists() else path
    arr  = np.array(Image.open(src))
    if arr.ndim == 3:                         # safety: collapse if 3-channel
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)

class GID15Dataset(Dataset):
    def __init__(self, split, patch_size, transform, crops_per_img):
        self.patch_size = patch_size
        self.transform  = transform
        self.crops      = crops_per_img

        img_dir, ann_dir = BASE/'img_dir'/split, BASE/'ann_dir'/split
        self.pairs = []
        for img_path in sorted(img_dir.glob('*.tif')):
            mp = ann_dir / f'{img_path.stem}_15label.png'   # ← .png for BOTH
            if mp.exists():
                self.pairs.append((img_path, mp))

        # Verify mask values on first pair (prints in commit log)
        sample = load_mask(self.pairs[0][1])
        print(f"  {split}: {len(self.pairs)} pairs → "
              f"{len(self.pairs)*self.crops} patches/epoch | "
              f"mask vals {np.unique(sample)[:8]}...")

    def __len__(self):
        return len(self.pairs) * self.crops

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx % len(self.pairs)]
        img  = np.array(Image.open(img_path).convert('RGB'))
        mask = load_mask(mask_path)
        h, w = img.shape[:2]

        for _ in range(10):                              # prefer labeled crops
            top  = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            mcrop = mask[top:top+self.patch_size, left:left+self.patch_size]
            if (mcrop != IGNORE_IDX).mean() > 0.10:
                break
        icrop = img[top:top+self.patch_size, left:left+self.patch_size]
        mcrop = np.clip(mcrop, 0, NUM_CLASSES-1).astype(np.int64)

        aug = self.transform(image=icrop, mask=mcrop.astype(np.uint8))
        return aug['image'], aug['mask'].long()

# ── Augmentation ──────────────────────────────────────────────────────────────
NORM = A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
    A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.4),
    A.RandomBrightnessContrast(p=0.3),
    NORM, ToTensorV2(),
])
val_tf = A.Compose([NORM, ToTensorV2()])

# ── Class weights (log-smoothed inverse frequency) ───────────────────────────
def compute_class_weights(pairs, subsample=8):
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, mp in pairs:
        m = load_mask(mp)[::subsample, ::subsample]
        counts += np.bincount(np.clip(m, 0, NUM_CLASSES-1).ravel(),
                              minlength=NUM_CLASSES)
    freq = counts / counts.sum()
    w = 1.0 / np.log(1.02 + freq)
    w[IGNORE_IDX] = 0.0
    return torch.tensor(w, dtype=torch.float32)

print("Building datasets...")
train_ds = GID15Dataset('train', PATCH_SIZE, train_tf, crops_per_img=12)
val_ds   = GID15Dataset('val',   PATCH_SIZE, val_tf,   crops_per_img=20)
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True)
val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                      num_workers=2, pin_memory=True)

print("Computing class weights...")
class_weights = compute_class_weights(train_ds.pairs).to(DEVICE)
present = [f"{CLASS_NAMES[i]}={class_weights[i]:.2f}"
           for i in range(NUM_CLASSES) if class_weights[i] > 0]
print("  " + " | ".join(present))

# ── Combined Dice + weighted CrossEntropy loss ────────────────────────────────
dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=IGNORE_IDX)
ce   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_IDX)
def criterion(logits, target):
    return 0.5 * dice(logits, target) + 0.5 * ce(logits, target)

# ── mIoU via confusion matrix ─────────────────────────────────────────────────
def update_confmat(cm, pred, tgt):
    valid = tgt != IGNORE_IDX
    p, t  = pred[valid], tgt[valid]
    k = t * NUM_CLASSES + p
    cm += torch.bincount(k, minlength=NUM_CLASSES**2)\
            .reshape(NUM_CLASSES, NUM_CLASSES)
    return cm

def miou_from_confmat(cm):
    cm = cm.float()
    tp = cm.diag(); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    union = tp + fp + fn
    iou = tp / union.clamp(min=1)
    valid = (union > 0); valid[IGNORE_IDX] = False
    return iou[valid].mean().item()

scaler = torch.cuda.amp.GradScaler()

def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=DEVICE)
    with torch.set_grad_enabled(train):
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, masks)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            total_loss += loss.item()
            cm = update_confmat(cm, logits.argmax(1).flatten(), masks.flatten())
    acc = (cm.diag().sum().float() /
           (cm.sum() - cm[IGNORE_IDX].sum()).clamp(min=1)).item()
    return total_loss/len(loader), miou_from_confmat(cm), acc

# %% [notebook cell 3]
# ════════════════════════════════════════════════════════════════════════════
# Cell 3/3 — Two-phase training with early stopping on val mIoU
# ════════════════════════════════════════════════════════════════════════════

model = smp.Unet(
    encoder_name          = 'resnet50',
    encoder_weights       = 'imagenet',
    in_channels           = 3,
    classes               = NUM_CLASSES,
    decoder_use_batchnorm = True,
).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")

def set_encoder(trainable):
    for p in model.encoder.parameters():
        p.requires_grad = trainable

history       = []
best_miou     = -1.0
patience_ctr  = 0

def train_phase(name, epochs, lr, monitor_early_stop):
    global best_miou, patience_ctr
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n=== {name} | lr={lr:.0e} | {epochs} epochs ===")
    print(f"{'Ep':>4} {'TrLoss':>9} {'TrIoU':>7} {'VlLoss':>9} "
          f"{'VlIoU':>7} {'VlAcc':>7} {'LR':>9}")
    print("─"*60)

    for ep in range(1, epochs+1):
        tr_loss, tr_iou, _      = run_epoch(model, train_dl, optimizer)
        vl_loss, vl_iou, vl_acc = run_epoch(model, val_dl)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        history.append(dict(phase=name, ep=ep, tr_loss=tr_loss, tr_iou=tr_iou,
                            vl_loss=vl_loss, vl_iou=vl_iou, vl_acc=vl_acc))
        flag = ""
        if vl_iou > best_miou:
            best_miou, patience_ctr = vl_iou, 0
            torch.save(dict(model_state=model.state_dict(),
                            val_miou=vl_iou, val_acc=vl_acc,
                            classes=CLASS_NAMES, phase=name, epoch=ep),
                       CKPT_DIR/'best_model.pth')
            flag = " ✓ best"
        elif monitor_early_stop:
            patience_ctr += 1
            flag = f" patience {patience_ctr}/{PATIENCE}"

        print(f"{ep:>4} {tr_loss:>9.4f} {tr_iou:>7.4f} {vl_loss:>9.4f} "
              f"{vl_iou:>7.4f} {vl_acc:>7.4f} {lr_now:>9.1e}{flag}")

        if monitor_early_stop and patience_ctr >= PATIENCE:
            print(f"\n⚡ Early stopping — best val mIoU = {best_miou:.4f}")
            return True
    return False

# ── Phase 1: freeze encoder, warm up the decoder (higher LR, no early stop) ──
set_encoder(False)
train_phase("WARMUP (decoder only)", WARMUP_EPOCHS, lr=1e-3,
            monitor_early_stop=False)

# ── Phase 2: unfreeze everything, fine-tune (low LR, early stopping) ─────────
set_encoder(True)
train_phase("FINETUNE (full network)", FINETUNE_EPOCHS, lr=3e-5,
            monitor_early_stop=True)

# ── Report best ───────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT_DIR/'best_model.pth')
print(f"\nBest checkpoint: {ckpt['phase']} epoch {ckpt['epoch']}")
print(f"  val mIoU : {ckpt['val_miou']:.4f}")
print(f"  val acc  : {ckpt['val_acc']:.4f}")
print(f"  saved → {CKPT_DIR/'best_model.pth'}")

# ── Curves ────────────────────────────────────────────────────────────────────
ep_axis = range(1, len(history)+1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14,5))
a1.plot(ep_axis, [h['tr_loss'] for h in history], label='Train')
a1.plot(ep_axis, [h['vl_loss'] for h in history], label='Val')
a1.axvline(WARMUP_EPOCHS+0.5, color='gray', ls=':', label='Unfreeze')
a1.set_title('Loss'); a1.set_xlabel('Epoch'); a1.legend(); a1.grid(True)
a2.plot(ep_axis, [h['tr_iou'] for h in history], label='Train mIoU')
a2.plot(ep_axis, [h['vl_iou'] for h in history], label='Val mIoU')
a2.axvline(WARMUP_EPOCHS+0.5, color='gray', ls=':')
a2.set_title('mIoU'); a2.set_xlabel('Epoch'); a2.legend(); a2.grid(True)
plt.tight_layout(); plt.savefig(OUT_DIR/'training_curve.png', dpi=150); plt.show()
print(f"Curve → {OUT_DIR/'training_curve.png'}")

# %% [notebook cell 4]
import numpy as np
from PIL import Image
from pathlib import Path

BASE = Path('/kaggle/input/datasets/sheheryarkhan00/gid15-yrb')

val_png = sorted((BASE/'ann_dir'/'val').glob('*.png'))
print(f"Val .png masks found: {len(val_png)}")
if val_png:
    m = np.array(Image.open(val_png[0]))
    print(f"  Sample : {val_png[0].name}")
    print(f"  Shape  : {m.shape}")
    print(f"  Unique : {np.unique(m)}   ← expect [0..15]")
