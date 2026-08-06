#!/usr/bin/env python3
"""Train the 15-output GID ResNet-50/U-Net with scene-level holdout.

This is a command-line refactoring of the training workflow in
notebook67e0991e34.ipynb. It keeps the source notebook's 15-output convention:
reference classes 1..15 are shifted to model targets 0..14; 255 is ignored.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np


PALETTE = {
    (200, 0, 0): 1, (250, 0, 150): 2, (200, 150, 150): 3, (250, 150, 150): 4,
    (0, 200, 0): 5, (150, 250, 0): 6, (150, 200, 150): 7, (200, 0, 200): 8,
    (150, 0, 250): 9, (150, 150, 250): 10, (250, 200, 0): 11, (200, 200, 0): 12,
    (0, 0, 200): 13, (0, 150, 200): 14, (0, 200, 250): 15,
}
IGNORE = 255
DEFAULT_EVAL_SCENES = [
    "0000575925", "0001064454", "0001118839", "0001395956", "0001680858"
]
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def rgb_to_index(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label)
    channels = label.shape[0] if label.ndim == 3 else 1
    if channels >= 3:
        rgb = np.transpose(label[:3], (1, 2, 0))
        output = np.full(rgb.shape[:2], IGNORE, dtype=np.int64)
        for colour, index in PALETTE.items():
            output[np.all(rgb == np.asarray(colour), axis=-1)] = index
        return output
    band = label[0] if label.ndim == 3 else label
    output = np.full(band.shape, IGNORE, dtype=np.int64)
    valid = (band >= 1) & (band <= 15)
    output[valid] = band[valid].astype(np.int64)
    return output


def scene_id(path: str) -> str:
    name = os.path.basename(path)
    return name.split("L1A")[1][:10] if "L1A" in name else name


def find_pairs(data_dir: str) -> list[tuple[str, str, str]]:
    images, labels = {}, {}
    for filename in glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True):
        lower = os.path.basename(filename).lower()
        if not lower.endswith((".tif", ".tiff", ".png")):
            continue
        normalized = filename.replace(os.sep, "/").lower()
        if "label" in lower or "/ann_dir/" in normalized:
            labels[scene_id(filename)] = filename
        elif "/img_dir/" in normalized or "mss" in lower:
            images[scene_id(filename)] = filename
    return [(sid, images[sid], labels[sid]) for sid in images if sid in labels]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="outputs/gid15_scene_holdout")
    parser.add_argument("--eval-scenes", nargs="+", default=DEFAULT_EVAL_SCENES)
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--spatial-holdout", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import rasterio
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader, Dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.data_dir)
    training_pairs = [pair for pair in pairs if pair[0] not in set(args.eval_scenes)]
    held_out_pairs = [pair for pair in pairs if pair[0] in set(args.eval_scenes)]
    if not training_pairs:
        if args.spatial_holdout and held_out_pairs:
            training_pairs = held_out_pairs
        else:
            raise RuntimeError("No training scene/label pairs remain after scene holdout.")

    class GIDPatches(Dataset):
        def __init__(self, pairs_, samples: int, training: bool):
            self.pairs = pairs_
            self.samples = samples
            self.training = training

        def __len__(self):
            return self.samples

        def _load(self, image_path, label_path):
            with rasterio.open(image_path) as source:
                image = source.read([1, 2, 3])
            with rasterio.open(label_path) as source:
                label = rgb_to_index(source.read())
            return image, label

        def __getitem__(self, index):
            _ = index
            _, image_path, label_path = random.choice(self.pairs)
            image, label = self._load(image_path, label_path)
            _, height, width = image.shape
            if height < args.patch or width < args.patch:
                raise ValueError(f"Scene smaller than patch: {(height, width)} < {args.patch}")

            if args.spatial_holdout:
                cut = int(0.8 * height)
                low, high = ((0, cut - args.patch) if self.training
                             else (cut, height - args.patch))
            else:
                low, high = 0, height - args.patch

            row = col = 0
            for _attempt in range(10):
                row = random.randint(max(0, low), max(0, high))
                col = random.randint(0, width - args.patch)
                label_crop = label[row:row + args.patch, col:col + args.patch]
                if (label_crop != IGNORE).mean() > 0.25:
                    break

            image_crop = image[:, row:row + args.patch, col:col + args.patch].astype(np.float32)
            label_crop = label[row:row + args.patch, col:col + args.patch]
            if self.training:
                if random.random() < 0.5:
                    image_crop = image_crop[:, :, ::-1]
                    label_crop = label_crop[:, ::-1]
                if random.random() < 0.5:
                    image_crop = image_crop[:, ::-1, :]
                    label_crop = label_crop[::-1, :]
                rotation = random.randint(0, 3)
                if rotation:
                    image_crop = np.rot90(image_crop, rotation, axes=(1, 2)).copy()
                    label_crop = np.rot90(label_crop, rotation).copy()

            features = (
                image_crop / 255.0 - IMAGENET_MEAN[:, None, None]
            ) / IMAGENET_STD[:, None, None]
            target = label_crop.copy()
            target[target != IGNORE] -= 1
            return (
                torch.from_numpy(features.copy()).float(),
                torch.from_numpy(target.copy()).long(),
            )

    train_samples = int(args.steps_per_epoch * args.batch * (1.0 - args.validation_fraction))
    validation_samples = int(args.steps_per_epoch * args.batch * args.validation_fraction)
    train_loader = DataLoader(
        GIDPatches(training_pairs, train_samples, True),
        batch_size=args.batch, num_workers=args.num_workers, drop_last=True,
    )
    validation_loader = DataLoader(
        GIDPatches(training_pairs, validation_samples, False),
        batch_size=args.batch, num_workers=args.num_workers,
    )

    model = smp.Unet(
        encoder_name="resnet50", encoder_weights="imagenet",
        in_channels=3, classes=15,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    amp_enabled = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    @torch.no_grad()
    def validation_metrics():
        model.eval()
        intersection = np.zeros(15, dtype=np.float64)
        union = np.zeros(15, dtype=np.float64)
        correct = total = 0
        for features, target in validation_loader:
            prediction = model(features.to(device)).argmax(1).cpu().numpy()
            target_np = target.numpy()
            valid = target_np != IGNORE
            correct += int((prediction[valid] == target_np[valid]).sum())
            total += int(valid.sum())
            for class_index in range(15):
                predicted = (prediction == class_index) & valid
                reference = (target_np == class_index) & valid
                intersection[class_index] += np.logical_and(predicted, reference).sum()
                union[class_index] += np.logical_or(predicted, reference).sum()
        iou = intersection[union > 0] / union[union > 0]
        return correct / max(1, total), float(iou.mean()) if iou.size else 0.0

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        loss_total = batches = 0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = criterion(model(features), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.item())
            batches += 1
        scheduler.step()
        oa, miou = validation_metrics()
        record = {
            "epoch": epoch, "loss": loss_total / max(1, batches),
            "validation_oa": oa, "validation_miou": miou,
            "seconds": time.time() - started,
        }
        history.append(record)
        print(record, flush=True)
        if miou > best:
            best = miou
            torch.save(model.state_dict(), out_dir / "gid15_unet_resnet50_15output.pth")

    metadata = {
        "history": history,
        "best_validation_miou": best,
        "architecture": "smp.Unet(resnet50, in_channels=3, classes=15)",
        "target_encoding": "GID classes 1..15 shifted to model indices 0..14; 255 ignored",
        "training_scenes": [pair[0] for pair in training_pairs],
        "held_out_scenes": [pair[0] for pair in held_out_pairs],
    }
    (out_dir / "gid15_training_history.json").write_text(json.dumps(metadata, indent=2))
    print(f"Best validation mIoU: {best:.4f}")


if __name__ == "__main__":
    main()
