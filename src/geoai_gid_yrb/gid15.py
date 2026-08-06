"""GID-15 dataset and training primitives extracted from gid15-training.ipynb."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np

GID_CLASS_NAMES = {
    0:'Unlabeled',1:'Industrial area',2:'Paddy field',3:'Irrigated farmland',
    4:'Dry cropland',5:'Garden land',6:'Arbor forest',7:'Shrub forest',
    8:'Natural meadow',9:'Artificial meadow',10:'River',11:'Urban residential',
    12:'Lake',13:'Pond',14:'Fish pond',15:'Sea',
}


def load_mask(path):
    from PIL import Image
    path=Path(path); png=path.with_suffix('.png'); src=png if png.exists() else path
    arr=np.array(Image.open(src))
    if arr.ndim==3: arr=arr[:,:,0]
    return arr.astype(np.uint8)


def make_gid_dataset_class(base, ignore_index=0, num_classes=16):
    import torch
    from PIL import Image
    class GID15Dataset(torch.utils.data.Dataset):
        def __init__(self, split, patch_size, transform, crops_per_img):
            self.patch_size=patch_size; self.transform=transform; self.crops=crops_per_img
            img_dir,ann_dir=Path(base)/'img_dir'/split,Path(base)/'ann_dir'/split
            self.pairs=[]
            for image_path in sorted(img_dir.glob('*.tif')):
                mask_path=ann_dir/f'{image_path.stem}_15label.png'
                if mask_path.exists(): self.pairs.append((image_path,mask_path))
            if not self.pairs: raise FileNotFoundError(f'No GID pairs in {img_dir}')
        def __len__(self): return len(self.pairs)*self.crops
        def __getitem__(self,index):
            image_path,mask_path=self.pairs[index%len(self.pairs)]
            image=np.array(Image.open(image_path).convert('RGB')); mask=load_mask(mask_path)
            h,w=image.shape[:2]
            if h<self.patch_size or w<self.patch_size:
                raise ValueError('Image smaller than requested patch')
            for _ in range(10):
                top=random.randint(0,h-self.patch_size); left=random.randint(0,w-self.patch_size)
                mask_crop=mask[top:top+self.patch_size,left:left+self.patch_size]
                if (mask_crop!=ignore_index).mean()>0.10: break
            image_crop=image[top:top+self.patch_size,left:left+self.patch_size]
            mask_crop=np.clip(mask_crop,0,num_classes-1).astype(np.int64)
            aug=self.transform(image=image_crop,mask=mask_crop.astype(np.uint8))
            return aug['image'],aug['mask'].long()
    return GID15Dataset


def compute_class_weights(pairs, num_classes=16, ignore_index=0, subsample=8):
    import torch
    counts=np.zeros(num_classes,dtype=np.float64)
    for _,mask_path in pairs:
        mask=load_mask(mask_path)[::subsample,::subsample]
        counts += np.bincount(np.clip(mask,0,num_classes-1).ravel(),minlength=num_classes)
    frequency=counts/counts.sum(); weights=1.0/np.log(1.02+frequency); weights[ignore_index]=0.0
    return torch.tensor(weights,dtype=torch.float32)
