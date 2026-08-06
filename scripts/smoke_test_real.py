#!/usr/bin/env python3
"""Actual two-arm 2-epoch smoke test, refactored from smoke-test.ipynb.

Requires YRB patch NPZ files, a GID checkpoint, GPU, and full dependencies.
"""
import argparse,glob,re,os
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--patch-dir',required=True); p.add_argument('--gid-checkpoint',required=True); p.add_argument('--epochs',type=int,default=2); a=p.parse_args()
    import torch,torch.nn as nn
    from geoai_gid_yrb.data import basin_id,deduplicate_by_basename
    from geoai_gid_yrb.normalization import compute_band_stats,normalize_bands
    from geoai_gid_yrb.models import build_prithvi,build_yrb_resnet,unwrap_output
    paths=deduplicate_by_basename(glob.glob(f'{a.patch_dir}/**/patches_SB*.npz',recursive=True)); by={}
    for path in paths:
        d=np.load(path); by.setdefault(basin_id(path),[]).append((d['X'],d['y']))
    by={sb:(np.concatenate([x for x,_ in v]),np.concatenate([y for _,y in v])) for sb,v in by.items()}
    required={59,108,100}; missing=required-set(by)
    if missing: raise RuntimeError(f'Missing smoke-test basins: {sorted(missing)}')
    xtr=np.concatenate([by[59][0][:300],by[108][0][:300]]); ytr=np.concatenate([by[59][1][:300],by[108][1][:300]])
    xte,yte=by[100][0][:150],by[100][1][:150]; mean,std=compute_band_stats(xtr)
    class DS(torch.utils.data.Dataset):
        def __init__(self,x,y): self.x,self.y=x,y
        def __len__(self): return len(self.x)
        def __getitem__(self,i): return torch.tensor(normalize_bands(self.x[i],mean,std)),torch.tensor(self.y[i].astype(np.int64))
    tr=torch.utils.data.DataLoader(DS(xtr,ytr),batch_size=8,shuffle=True); te=torch.utils.data.DataLoader(DS(xte,yte),batch_size=8)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def run(model,name):
        model.to(device); opt=torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],lr=1e-4); lossf=nn.CrossEntropyLoss(ignore_index=0)
        for ep in range(a.epochs):
            model.train(); total=0
            for x,y in tr:
                x,y=x.to(device),y.to(device); opt.zero_grad(); out=unwrap_output(model(x)); loss=lossf(out,y); loss.backward(); opt.step(); total+=loss.item()
            print(name,'epoch',ep+1,'loss',total/len(tr))
        model.eval(); cm=torch.zeros(7,7,dtype=torch.long)
        with torch.no_grad():
            for x,y in te:
                pred=unwrap_output(model(x.to(device))).argmax(1).cpu().flatten(); truth=y.flatten(); valid=truth!=0; encoded=truth[valid]*7+pred[valid]; cm+=torch.bincount(encoded,minlength=49).reshape(7,7)
        tp=cm.diag().float(); union=cm.sum(0)+cm.sum(1)-tp; valid=union>0; valid[0]=False; print(name,'test mIoU',float((tp/union.clamp(min=1))[valid].mean()))
    run(build_prithvi(7),'Prithvi'); run(build_yrb_resnet(a.gid_checkpoint,7),'ResNet')
if __name__=='__main__': main()
