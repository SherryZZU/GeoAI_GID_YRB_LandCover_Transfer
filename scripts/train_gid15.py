#!/usr/bin/env python3
"""Run the actual GID-15 two-phase training workflow.

This CLI is a refactoring of gid15-training.ipynb. It requires GID-15 data.
"""
import argparse, random
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--base',required=True); parser.add_argument('--out',default='outputs/gid15')
    parser.add_argument('--patch-size',type=int,default=512); parser.add_argument('--batch-size',type=int,default=6)
    parser.add_argument('--warmup-epochs',type=int,default=8); parser.add_argument('--finetune-epochs',type=int,default=60)
    parser.add_argument('--patience',type=int,default=12); parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from geoai_gid_yrb.gid15 import make_gid_dataset_class,compute_class_weights,GID_CLASS_NAMES
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=Path(args.out); ckpt=out/'checkpoints'; ckpt.mkdir(parents=True,exist_ok=True)
    norm=A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
    train_tf=A.Compose([A.HorizontalFlip(p=.5),A.VerticalFlip(p=.5),A.RandomRotate90(p=.5),A.ColorJitter(.2,.2,.2,.1,p=.4),A.RandomBrightnessContrast(p=.3),norm,ToTensorV2()])
    val_tf=A.Compose([norm,ToTensorV2()]); Dataset=make_gid_dataset_class(args.base)
    train_ds=Dataset('train',args.patch_size,train_tf,12); val_ds=Dataset('val',args.patch_size,val_tf,20)
    train_dl=DataLoader(train_ds,args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
    val_dl=DataLoader(val_ds,args.batch_size,shuffle=False,num_workers=2,pin_memory=True)
    weights=compute_class_weights(train_ds.pairs).to(device)
    dice=smp.losses.DiceLoss(mode='multiclass',ignore_index=0); ce=nn.CrossEntropyLoss(weight=weights,ignore_index=0)
    model=smp.Unet('resnet50',encoder_weights='imagenet',in_channels=3,classes=16,decoder_use_batchnorm=True).to(device)
    scaler=torch.cuda.amp.GradScaler(); best=-1.; wait=0
    def epoch(loader,opt=None):
        training=opt is not None; model.train(training); total=0.; cm=torch.zeros(16,16,dtype=torch.long,device=device)
        with torch.set_grad_enabled(training):
            for image,label in loader:
                image,label=image.to(device),label.to(device)
                with torch.cuda.amp.autocast(): logits=model(image); loss=.5*dice(logits,label)+.5*ce(logits,label)
                if training:
                    opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
                total+=loss.item(); prediction=logits.argmax(1); valid=label!=0; encoded=label[valid]*16+prediction[valid]
                cm += torch.bincount(encoded,minlength=256).reshape(16,16)
        tp=cm.diag().float(); union=cm.sum(0)+cm.sum(1)-tp; valid=union>0; valid[0]=False
        return total/len(loader),float((tp/union.clamp(min=1))[valid].mean())
    phases=[('warmup',args.warmup_epochs,1e-3,False),('finetune',args.finetune_epochs,3e-5,True)]
    for name,count,lr,early in phases:
        for parameter in model.encoder.parameters(): parameter.requires_grad=(name=='finetune')
        opt=optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=1e-2); scheduler=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=count)
        for ep in range(1,count+1):
            train_loss,train_iou=epoch(train_dl,opt); val_loss,val_iou=epoch(val_dl); scheduler.step()
            print(name,ep,train_loss,train_iou,val_loss,val_iou,flush=True)
            if val_iou>best:
                best=val_iou; wait=0; torch.save({'model_state':model.state_dict(),'val_miou':val_iou,'classes':GID_CLASS_NAMES,'phase':name,'epoch':ep},ckpt/'best_model.pth')
            elif early:
                wait+=1
                if wait>=args.patience: break
    print(f'best val mIoU={best:.4f}; checkpoint={ckpt/"best_model.pth"}')
if __name__=='__main__': main()
