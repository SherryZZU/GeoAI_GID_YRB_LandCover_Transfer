"""
Complete code-cell extraction from notebook67e0991e34.ipynb.

Notebook shell commands are retained as comments so this file is valid Python.
The original cell order is preserved. Prefer the refactored scripts/ entry points
for new runs; this file is the full provenance extraction.
"""


# ============================================================================
# SOURCE NOTEBOOK CELL 0 | CODE CELL 1
# ============================================================================
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


# ============================================================================
# SOURCE NOTEBOOK CELL 1 | CODE CELL 2
# ============================================================================
"""
TRAIN the GID-15 segmentation model (U-Net + ResNet-50 encoder).
Run on KAGGLE (GPU T4). Produces the weights consumed by kaggle_2x2_ablation.py.

Fixed: _load indentation; no scene cache + num_workers=0 (prevents OOM);
modern torch.amp API (no deprecation warnings).
"""
# NOTEBOOK_SHELL_COMMAND: pip install -q segmentation-models-pytorch==0.3.3 rasterio
import os, glob, random, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import rasterio

# ----------------------------------------------------------------------------- CONFIG
DATA_DIR   = "/kaggle/input/datasets/sheheryarkhan00/gid15-yrb"  # images in img_dir/, labels in ann_dir/
OUT_DIR    = "/kaggle/working"
ENCODER    = "resnet50"
N_CLASSES  = 15
PATCH      = 512
BATCH      = 6
EPOCHS     = 40
LR         = 3e-4
VAL_FRAC   = 0.12
STEPS_PER_EPOCH = 400
SPATIAL_HOLDOUT = False
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

EVAL_SCENES = ["0000575925", "0001064454", "0001118839", "0001395956", "0001680858"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

PALETTE = {
    (200,0,0):1,    (250,0,150):2,   (200,150,150):3, (250,150,150):4,
    (0,200,0):5,    (150,250,0):6,   (150,200,150):7, (200,0,200):8,
    (150,0,250):9,  (150,150,250):10,(250,200,0):11,  (200,200,0):12,
    (0,0,200):13,   (0,150,200):14,  (0,200,250):15,
}
IGNORE = 255

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# --------------------------------------------------------------------- label decoding
def rgb_to_index(lab):
    """GID label -> (H,W) int class index 1..15, nodata->IGNORE.
    Handles 3-band RGB-coded AND single-band indexed labels."""
    lab = np.asarray(lab)
    nb = lab.shape[0] if lab.ndim == 3 else 1
    if nb >= 3:
        rgb = np.transpose(lab[:3], (1, 2, 0))
        out = np.full(rgb.shape[:2], IGNORE, np.int64)
        for (r, g, b), idx in PALETTE.items():
            out[(rgb[..., 0] == r) & (rgb[..., 1] == g) & (rgb[..., 2] == b)] = idx
        return out
    band = lab[0] if lab.ndim == 3 else lab
    out = np.full(band.shape, IGNORE, np.int64)
    valid = (band >= 1) & (band <= 15)
    out[valid] = band[valid].astype(np.int64)
    return out

# ------------------------------------------------------------------------ scene index
def scene_id(path):
    base = os.path.basename(path)
    return base.split("L1A")[1][:10] if "L1A" in base else base

def find_pairs():
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.*"), recursive=True)
    imgs, labs = {}, {}
    for f in all_files:
        b = os.path.basename(f).lower()
        if not (b.endswith(".tif") or b.endswith(".tiff") or b.endswith(".png")):
            continue
        if "label" in b or "/ann_dir/" in f.replace(os.sep, "/").lower():
            labs[scene_id(f)] = f
        elif "/img_dir/" in f.replace(os.sep, "/").lower() or "mss" in b:
            if "label" not in b:
                imgs[scene_id(f)] = f
    pairs = [(sid, imgs[sid], labs[sid]) for sid in imgs if sid in labs]
    only_img = [s for s in imgs if s not in labs]
    only_lab = [s for s in labs if s not in imgs]
    if only_img: print(f"  (note) {len(only_img)} image(s) with no label")
    if only_lab: print(f"  (note) {len(only_lab)} label(s) with no image")
    return pairs

# -------------------------------------------------------------------------- dataset
class GIDPatches(Dataset):
    """Random PATCHxPATCH crops from full GID scenes, ImageNet-normalized,
    flip/rot90 augmentation. No scene cache (avoids OOM); rereads per crop."""
    def __init__(self, pairs, n_samples, train=True):
        self.pairs = pairs
        self.n = n_samples
        self.train = train

    def __len__(self):
        return self.n

    def _load(self, im, lab):
        with rasterio.open(im) as s:
            img = s.read([1, 2, 3])
        with rasterio.open(lab) as s:
            lb = rgb_to_index(s.read())
        return img, lb

    def __getitem__(self, i):
        sid, im, lab = random.choice(self.pairs)
        img, lb = self._load(im, lab)
        _, H, W = img.shape
        if SPATIAL_HOLDOUT:
            cut = int(0.8 * H)
            r0lo, r0hi = (0, cut - PATCH) if self.train else (cut, H - PATCH)
        else:
            r0lo, r0hi = 0, H - PATCH
        for _ in range(10):
            r0 = random.randint(max(0, r0lo), max(0, r0hi))
            c0 = random.randint(0, W - PATCH)
            lcrop = lb[r0:r0+PATCH, c0:c0+PATCH]
            if (lcrop != IGNORE).mean() > 0.25:
                break
        icrop = img[:, r0:r0+PATCH, c0:c0+PATCH].astype(np.float32)
        if self.train:
            if random.random() < 0.5:
                icrop = icrop[:, :, ::-1]; lcrop = lcrop[:, ::-1]
            if random.random() < 0.5:
                icrop = icrop[:, ::-1, :]; lcrop = lcrop[::-1, :]
            k = random.randint(0, 3)
            if k:
                icrop = np.rot90(icrop, k, axes=(1, 2)).copy()
                lcrop = np.rot90(lcrop, k).copy()
        x = (icrop/255.0 - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        t = lcrop.copy()
        t[t != IGNORE] -= 1
        return torch.from_numpy(x.copy()).float(), torch.from_numpy(t.copy()).long()

# --------------------------------------------------------------------------- metrics
@torch.no_grad()
def val_miou(model, loader):
    model.eval()
    inter = np.zeros(N_CLASSES); union = np.zeros(N_CLASSES); correct = 0; total = 0
    for x, t in loader:
        x = x.to(DEVICE)
        pred = model(x).argmax(1).cpu().numpy()
        t = t.numpy()
        m = t != IGNORE
        correct += (pred[m] == t[m]).sum(); total += m.sum()
        for c in range(N_CLASSES):
            pi = (pred == c) & m; gi = (t == c) & m
            inter[c] += np.logical_and(pi, gi).sum()
            union[c] += np.logical_or(pi, gi).sum()
    ious = inter[union > 0] / union[union > 0]
    return (correct/max(1, total)), float(np.mean(ious)) if len(ious) else 0.0

# ----------------------------------------------------------------------------- train
def main():
    pairs = find_pairs()
    train_pairs = [p for p in pairs if p[0] not in EVAL_SCENES]
    eval_pairs  = [p for p in pairs if p[0] in EVAL_SCENES]
    print(f"found {len(pairs)} scene/label pairs")
    print(f"  training scenes ({len(train_pairs)})")
    print(f"  held-out eval   ({len(eval_pairs)}): {[p[0] for p in eval_pairs]}")
    if not train_pairs:
        if SPATIAL_HOLDOUT:
            print("no non-eval scenes -> SPATIAL_HOLDOUT active.")
            train_pairs = eval_pairs
        else:
            raise SystemExit("ERROR: no training scenes after excluding EVAL_SCENES.")

    tr_ds = GIDPatches(train_pairs, int(STEPS_PER_EPOCH*BATCH*(1-VAL_FRAC)), train=True)
    va_ds = GIDPatches(train_pairs, int(STEPS_PER_EPOCH*BATCH*VAL_FRAC),     train=False)
    tr = DataLoader(tr_ds, batch_size=BATCH, num_workers=0, drop_last=True)
    va = DataLoader(va_ds, batch_size=BATCH, num_workers=0)

    model = smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet",
                     in_channels=3, classes=N_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE)
    use_amp = (DEVICE == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = -1; history = []
    for ep in range(1, EPOCHS+1):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for x, t in tr:
            x, t = x.to(DEVICE), t.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = lossf(model(x), t)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item(); nb += 1
        sched.step()
        oa, miou = val_miou(model, va)
        history.append(dict(epoch=ep, loss=run/max(1, nb), val_OA=oa, val_mIoU=miou))
        print(f"ep {ep:02d}/{EPOCHS} loss {run/max(1,nb):.3f} | val OA {oa:.3f} mIoU {miou:.3f} | {time.time()-t0:.0f}s")
        if miou > best:
            best = miou
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "gid15_unet_resnet50.pth"))
            print(f"   saved best (val mIoU {miou:.3f})")

    with open(os.path.join(OUT_DIR, "gid15_training_history.json"), "w") as f:
        json.dump(dict(history=history, best_val_mIoU=best,
                       arch="smp.Unet(resnet50, classes=15)",
                       train_scenes=[p[0] for p in train_pairs],
                       eval_scenes=EVAL_SCENES), f, indent=2)
    print(f"\nDONE. best val mIoU = {best:.3f}")
    print(f"weights -> {OUT_DIR}/gid15_unet_resnet50.pth")

if __name__ == "__main__":
    main()


# ============================================================================
# SOURCE NOTEBOOK CELL 2 | CODE CELL 3
# ============================================================================
"""
ITEMS 2 + 6 — Full 2x2 transfer-degradation ablation (resolution x label).
Run on KAGGLE (GPU T4) where the trained GID ResNet-50 weights + scenes live.
This SUPERSEDES kaggle_regenerate_gid_degradation.py (adds the 4th cell + interaction).

The 2x2 design (factors: spatial resolution x label granularity):

                    15-class (fine)        6-class (coarse)
    4 m  (native)   P1                     P4   <- the cell you were missing
    30 m (coarse)   P2                     P3

  P1  native 4 m,  15-class
  P2  30 m (4 m averaged x8), 15-class
  P3  30 m, remapped to 6 functional classes
  P4  native 4 m, remapped to 6 functional classes   (NEW — native pred, 6-class scored)

Why P4 matters: with all four corners you can compute the MAIN EFFECT of resolution
(averaged over both label schemes) and of labels (averaged over both resolutions),
plus the INTERACTION (do the two gaps compound additively?). Three corners can only
give you each effect at one level of the other factor — a confound.

The five EVALUABLE scenes (image + 15-class label both present):
    575925, 1064454, 1118839, 1395956, 1680858
(Scene 564539 has imagery but NO label -> not evaluable -> correctly excluded -> n=5.)

Outputs (save to /kaggle/working, then download all three):
    gid_2x2_per_scene.csv        <- per-scene OA & mIoU for all 4 cells (source-of-truth)
    gid_2x2_summary.json         <- cell means/sd + main effects + interaction
    gid_2x2_anova.txt            <- 2-way repeated-measures ANOVA table
"""
import os, glob, json
import numpy as np, pandas as pd
import rasterio
import torch, torch.nn.functional as F
import segmentation_models_pytorch as smp   # install: segmentation-models-pytorch

# ----------------------------------------------------------------------------
# CONFIG — point these at your Kaggle dataset paths
SCENE_DIR = "/kaggle/input/datasets/sheheryarkhan00/gid15-yrb"  # images in img_dir/, labels in ann_dir/
WEIGHTS   = "/kaggle/input/datasets/sheheryarkhan00/gid15-unet"   # produced by kaggle_train_gid15.py
# ^ if you saved the trained .pth as a separate Kaggle dataset, point this there instead,
#   e.g. "/kaggle/input/<your-weights-dataset>/gid15_unet_resnet50.pth"
OUT_DIR   = "/kaggle/working"
DOWNSCALE = 8
PATCH     = 512
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

SCENES = ["0000575925", "0001064454", "0001118839", "0001395956", "0001680858"]

MAP6 = {1:6,2:6,3:6,4:6, 5:1,6:1,7:1, 8:2,9:2, 10:3,11:3,12:3, 13:5,14:5,15:5}

# GID-15 labels are RGB-CODED 3-band images (verified), NOT single-band indices.
# Decode via the official palette. IGNORE marks nodata / unmapped pixels.
PALETTE = {
    (200,0,0):1,    (250,0,150):2,   (200,150,150):3, (250,150,150):4,
    (0,200,0):5,    (150,250,0):6,   (150,200,150):7, (200,0,200):8,
    (150,0,250):9,  (150,150,250):10,(250,200,0):11,  (200,200,0):12,
    (0,0,200):13,   (0,150,200):14,  (0,200,250):15,
}
IGNORE = 255
def rgb_to_index(lab):
    """GID label -> (H,W) uint8 class index 1..15, nodata->IGNORE.
    Handles BOTH 3-band RGB-coded labels (decode via PALETTE) and single-band
    indexed labels (0=nodata, 1..15=class). Accepts rasterio (bands,H,W)."""
    lab = np.asarray(lab)
    nb = lab.shape[0] if lab.ndim == 3 else 1
    if nb >= 3:
        rgb = np.transpose(lab[:3], (1, 2, 0))
        out = np.full(rgb.shape[:2], IGNORE, np.uint8)
        for (r, g, b), idx in PALETTE.items():
            out[(rgb[...,0]==r)&(rgb[...,1]==g)&(rgb[...,2]==b)] = idx
        return out
    band = lab[0] if lab.ndim == 3 else lab
    out = np.full(band.shape, IGNORE, np.uint8)
    valid = (band >= 1) & (band <= 15)
    out[valid] = band[valid].astype(np.uint8)
    return out

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

def find(scene, label=False):
    """Locate an image (img_dir/) or label (ann_dir/) for a scene id.
    Sensor-agnostic (MSS1/MSS2); tolerant of subfolders and .tif/.png."""
    hits = []
    for f in glob.glob(os.path.join(SCENE_DIR, "**", "*.*"), recursive=True):
        b = os.path.basename(f).lower()
        if scene not in os.path.basename(f):
            continue
        if not (b.endswith(".tif") or b.endswith(".tiff") or b.endswith(".png")):
            continue
        is_label = ("label" in b) or ("/ann_dir/" in f.replace(os.sep, "/").lower())
        if is_label == label:
            hits.append(f)
    return hits[0] if hits else None

def load_model():
    m = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=15)
    sd = torch.load(WEIGHTS, map_location=DEVICE); sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.to(DEVICE).eval()

@torch.no_grad()
def infer(model, rgb):
    H, W, _ = rgb.shape
    x = (rgb.astype(np.float32)/255.0 - IMAGENET_MEAN)/IMAGENET_STD
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(DEVICE)
    out = np.zeros((H, W), np.uint8)
    for r in range(0, H, PATCH):
        for c in range(0, W, PATCH):
            r2, c2 = min(r+PATCH, H), min(c+PATCH, W); ph, pw = r2-r, c2-c
            tile = F.pad(x[:, :, r:r2, c:c2], (0, PATCH-pw, 0, PATCH-ph))
            pred = model(tile).argmax(1)[0, :ph, :pw].cpu().numpy().astype(np.uint8) + 1
            out[r:r2, c:c2] = pred
    return out

def metrics(pred, gt, n_classes, valid):
    p, g = pred[valid], gt[valid]
    oa = float((p == g).mean())
    ious = []
    for c in range(1, n_classes+1):
        inter = np.logical_and(p == c, g == c).sum()
        union = np.logical_or(p == c, g == c).sum()
        if union > 0: ious.append(inter/union)
    return oa, (float(np.mean(ious)) if ious else float("nan"))

def downsample(arr, factor, is_label):
    H, W = arr.shape[:2]; h, w = H//factor, W//factor
    arr = arr[:h*factor, :w*factor]
    if is_label:
        blk = arr.reshape(h, factor, w, factor); out = np.zeros((h, w), arr.dtype)
        for i in range(h):
            for j in range(w):
                vals, cnts = np.unique(blk[i,:,j,:], return_counts=True)
                out[i,j] = vals[np.argmax(cnts)]
        return out
    blk = arr.reshape(h, factor, w, factor, arr.shape[2]).astype(np.float32)
    return blk.mean(axis=(1,3)).astype(np.uint8)

def remap6(a):
    out = np.zeros_like(a)
    for k, v in MAP6.items(): out[a == k] = v
    return out

def main():
    model = load_model()
    rows = []
    for s in SCENES:
        img_f, lab_f = find(s, False), find(s, True)
        assert img_f and lab_f, f"scene {s}: missing image or label"
        with rasterio.open(img_f) as src: rgb = np.transpose(src.read([1,2,3]), (1,2,0))
        with rasterio.open(lab_f) as src: lab = rgb_to_index(src.read())
        Hc, Wc = min(rgb.shape[0], lab.shape[0]), min(rgb.shape[1], lab.shape[1])
        rgb, lab = rgb[:Hc,:Wc], lab[:Hc,:Wc]

        # --- native-resolution prediction (shared by P1 and P4) ---
        pred15 = infer(model, rgb); valid = (lab >= 1) & (lab <= 15)
        oa1, mi1 = metrics(pred15, lab, 15, valid)                       # P1
        oa4, mi4 = metrics(remap6(pred15), remap6(lab), 6, valid)        # P4 (NEW)

        # --- 30 m prediction (shared by P2 and P3) ---
        rgb_ds, lab_ds = downsample(rgb, DOWNSCALE, False), downsample(lab, DOWNSCALE, True)
        valid_ds = (lab_ds >= 1) & (lab_ds <= 15)
        pred15_ds = infer(model, rgb_ds)
        oa2, mi2 = metrics(pred15_ds, lab_ds, 15, valid_ds)             # P2
        oa3, mi3 = metrics(remap6(pred15_ds), remap6(lab_ds), 6, valid_ds)  # P3

        rows.append(dict(scene=s,
            P1_OA=oa1, P1_mIoU=mi1, P2_OA=oa2, P2_mIoU=mi2,
            P3_OA=oa3, P3_mIoU=mi3, P4_OA=oa4, P4_mIoU=mi4))
        print(f"scene {s}: P1={oa1:.3f} P2={oa2:.3f} P3={oa3:.3f} P4={oa4:.3f} (OA)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "gid_2x2_per_scene.csv"), index=False)

    # ---- cell means + factorial effects (on OA and mIoU) ----
    summ = {}
    for met in ["OA", "mIoU"]:
        c = {p: df[f"{p}_{met}"] for p in ["P1","P2","P3","P4"]}
        m = {p: float(c[p].mean()) for p in c}
        sd = {p: float(c[p].std(ddof=1)) for p in c}
        # factor coding: resolution (native=P1,P4 / coarse=P2,P3); label (fine=P1,P2 / coarse=P3,P4)
        main_res = ((m["P1"]+m["P4"]) - (m["P2"]+m["P3"]))/2.0   # native - coarse, avg over labels
        main_lbl = ((m["P1"]+m["P2"]) - (m["P3"]+m["P4"]))/2.0   # fine - coarse, avg over res
        interaction = (m["P1"] - m["P2"]) - (m["P4"] - m["P3"])  # res effect@fine - res effect@coarse
        summ[met] = dict(cell_mean=m, cell_sd=sd,
                         main_effect_resolution=main_res,
                         main_effect_label=main_lbl,
                         interaction=interaction)
        print(f"\n[{met}] cell means: " + ", ".join(f"{k}={v:.3f}" for k,v in m.items()))
        print(f"[{met}] main effect RESOLUTION (native-coarse, avg) = {main_res:+.3f}")
        print(f"[{met}] main effect LABEL      (fine-coarse, avg)   = {main_lbl:+.3f}")
        print(f"[{met}] interaction                                 = {interaction:+.3f}")
        dom = "resolution" if abs(main_res) > abs(main_lbl) else "label"
        print(f"[{met}] dominant gap = {dom}  (|res|={abs(main_res):.3f} vs |lbl|={abs(main_lbl):.3f})")

    with open(os.path.join(OUT_DIR, "gid_2x2_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)

    # ---- 2-way repeated-measures ANOVA (statsmodels) on OA ----
    try:
        import statsmodels.formula.api as smf
        long = []
        for _, r in df.iterrows():
            for p, res, lbl in [("P1","native","fine"),("P2","coarse","fine"),
                                 ("P3","coarse","coarse"),("P4","native","coarse")]:
                long.append(dict(scene=r["scene"], resolution=res, label=lbl, OA=r[f"{p}_OA"]))
        L = pd.DataFrame(long)
        model_an = smf.ols("OA ~ C(resolution)*C(label) + C(scene)", data=L).fit()
        import statsmodels.api as sm
        aov = sm.stats.anova_lm(model_an, typ=2)
        with open(os.path.join(OUT_DIR, "gid_2x2_anova.txt"), "w") as f:
            f.write(str(aov))
        print("\n=== 2-way ANOVA (OA) ===\n", aov)
    except Exception as e:
        print("ANOVA skipped:", e)

    print(f"\nn scenes = {len(df)} (expect 5). Saved gid_2x2_per_scene.csv / _summary.json / _anova.txt")

if __name__ == "__main__":
    main()
