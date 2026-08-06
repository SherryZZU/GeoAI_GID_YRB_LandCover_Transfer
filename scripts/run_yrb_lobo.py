#!/usr/bin/env python3
"""Run the RAM-safe, resumable eight-fold YRB LOBO backbone comparison.

This is a command-line refactoring of yrb-lulc-lobo-comparison22.ipynb.
It trains Prithvi-EO-2.0 with LoRA and a six-band GID-initialized
ResNet-50/U-Net, writes a JSON result and two checkpoints per completed fold,
and creates summary.csv over all available folds.
"""
from __future__ import annotations
import argparse
import csv
import gc
import glob
import json
import os
import re
from pathlib import Path

import numpy as np


CLASS_NAMES = {
    1: "Cropland", 2: "Forest", 3: "Grassland",
    4: "Bare", 5: "Water", 6: "Built",
}
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


def basin_from_path(path: str) -> int:
    match = re.search(r"SB(\d+)", os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot infer basin identifier from {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--gid-checkpoint", required=True)
    parser.add_argument("--out-dir", default="outputs/lobo_results")
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--fold-order", nargs="+", type=int,
                        default=[100, 89, 108, 60, 78, 59, 109, 75])
    parser.add_argument("--cap", type=int, default=1500)
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--normalization-samples", type=int, default=150)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader, Dataset
    from geoai_gid_yrb.models import build_prithvi, build_yrb_resnet, unwrap_output

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_dir = Path(args.resume_dir) if args.resume_dir else None

    all_paths = sorted(
        glob.glob(os.path.join(args.patch_dir, "**", "patches_SB*.npz"), recursive=True)
    )
    unique_paths = []
    seen_names = set()
    for path in all_paths:
        name = os.path.basename(path)
        if name not in seen_names:
            seen_names.add(name)
            unique_paths.append(path)
    if not unique_paths:
        raise FileNotFoundError(f"No patches_SB*.npz files below {args.patch_dir}")

    by_basin = {}
    for path in unique_paths:
        by_basin.setdefault(basin_from_path(path), []).append(path)

    def load_basin(basin: int):
        features, labels = [], []
        for filename in by_basin[basin]:
            archive = np.load(filename, mmap_mode="r")
            features.append(np.asarray(archive["X"]))
            labels.append(np.asarray(archive["y"]))
        return np.concatenate(features), np.concatenate(labels)

    sizes = {
        basin: sum(np.load(path, mmap_mode="r")["X"].shape[0] for path in paths)
        for basin, paths in by_basin.items()
    }
    print({"basin_sizes": sizes, "total": sum(sizes.values())})

    samples = []
    for basin, paths in by_basin.items():
        archive = np.load(paths[0], mmap_mode="r")
        samples.append(np.asarray(archive["X"][: args.normalization_samples]))
    normalization_sample = np.concatenate(samples)
    channels = normalization_sample.shape[1]
    patch_height, patch_width = normalization_sample.shape[-2:]
    mean = normalization_sample.reshape(-1, channels, patch_height * patch_width).mean((0, 2)).astype(np.float32)
    std = (
        normalization_sample.reshape(-1, channels, patch_height * patch_width).std((0, 2))
        + 1e-6
    ).astype(np.float32)
    del normalization_sample, samples

    def normalize(features):
        return (
            features.astype(np.float32) - mean[:, None, None]
        ) / std[:, None, None]

    class PatchDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, index):
            return (
                torch.tensor(normalize(self.features[index])),
                torch.tensor(self.labels[index].astype(np.int64)),
            )

    def make_fold(test_basin: int):
        rng = np.random.default_rng(args.seed)
        training_features, training_labels = [], []
        for basin in by_basin:
            if basin == test_basin:
                continue
            features, labels = load_basin(basin)
            indices = rng.permutation(len(features))[: args.cap]
            training_features.append(features[indices].copy())
            training_labels.append(labels[indices].copy())
            del features, labels
        features = np.concatenate(training_features)
        labels = np.concatenate(training_labels)
        order = rng.permutation(len(features))
        features, labels = features[order], labels[order]
        n_validation = int(len(features) * args.validation_fraction)
        test_features, test_labels = load_basin(test_basin)
        return (
            features[n_validation:], labels[n_validation:],
            features[:n_validation], labels[:n_validation],
            test_features, test_labels,
        )

    num_classes = 7
    ignore_index = 0
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
    cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def loss_function(logits, target):
        return 0.5 * dice(logits, target) + 0.5 * cross_entropy(logits, target)

    @torch.no_grad()
    def confusion(model, loader):
        model.eval()
        matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for features, target in loader:
            logits = unwrap_output(model(features.to(device)))
            prediction = logits.argmax(1).cpu().flatten()
            target = target.flatten()
            valid = target != ignore_index
            encoded = target[valid] * num_classes + prediction[valid]
            matrix += torch.bincount(
                encoded, minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)
        return matrix

    def matrix_metrics(matrix):
        matrix = matrix.float()
        true_positive = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - true_positive
        iou = true_positive / union.clamp(min=1)
        valid = (union > 0).clone()
        valid[ignore_index] = False
        per_class = {
            CLASS_NAMES[index]: round(iou[index].item(), 3)
            for index in CLASS_NAMES if union[index] > 0
        }
        denominator = (matrix.sum() - matrix[ignore_index].sum()).clamp(min=1)
        accuracy = (true_positive.sum() / denominator).item()
        return round(iou[valid].mean().item(), 4), round(accuracy, 4), per_class

    def train_one(model_builder, tag, train_loader, validation_loader, checkpoint_path):
        model = model_builder().to(device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate, weight_decay=1e-2,
        )
        amp_enabled = str(device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        best = -1.0
        wait = 0
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            total = 0.0
            for features, target in train_loader:
                features, target = features.to(device), target.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = unwrap_output(model(features))
                    loss = loss_function(logits, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.item())
            validation_miou, _, _ = matrix_metrics(confusion(model, validation_loader))
            print({
                "model": tag, "epoch": epoch,
                "loss": total / max(1, len(train_loader)),
                "validation_miou": validation_miou,
            }, flush=True)
            if validation_miou > best:
                best = validation_miou
                wait = 0
                trainable_state = {
                    name: parameter.detach().cpu()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                torch.save(trainable_state, checkpoint_path)
            else:
                wait += 1
                if wait >= args.patience:
                    break
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        return model, best

    def completed_result(basin: int) -> Path | None:
        candidates = [out_dir]
        if resume_dir:
            candidates.append(resume_dir)
        for directory in candidates:
            path = directory / f"fold_SB{basin}.json"
            if path.exists():
                return path
        return None

    for basin in args.fold_order:
        if basin not in by_basin:
            raise KeyError(f"Requested basin SB{basin} is absent from patch files")
        previous = completed_result(basin)
        if previous:
            print(f"SB{basin} already complete at {previous}; skipping")
            continue

        arrays = make_fold(basin)
        x_train, y_train, x_validation, y_validation, x_test, y_test = arrays
        train_loader = DataLoader(
            PatchDataset(x_train, y_train), batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True,
        )
        validation_loader = DataLoader(
            PatchDataset(x_validation, y_validation), batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            PatchDataset(x_test, y_test), batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        result = {"test_basin": basin, "n_test": int(len(x_test))}

        builders = [
            ("prithvi", lambda: build_prithvi(num_classes=num_classes, bands=BANDS)),
            ("resnet", lambda: build_yrb_resnet(args.gid_checkpoint, num_classes=num_classes)),
        ]
        for tag, builder in builders:
            checkpoint = out_dir / f"fold_SB{basin}_{tag}.pth"
            model, best_validation = train_one(
                builder, tag, train_loader, validation_loader, checkpoint
            )
            matrix = confusion(model, test_loader)
            miou, accuracy, per_class = matrix_metrics(matrix)
            result[tag] = {
                "test_mIoU": miou,
                "test_acc": accuracy,
                "per_class_IoU": per_class,
                "best_val_mIoU": round(best_validation, 4),
                "confusion": matrix.tolist(),
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        (out_dir / f"fold_SB{basin}.json").write_text(json.dumps(result, indent=2))
        del x_train, y_train, x_validation, y_validation, x_test, y_test
        gc.collect()

    rows = []
    for basin in args.fold_order:
        result_path = completed_result(basin)
        if not result_path:
            continue
        result = json.loads(result_path.read_text())
        rows.append((
            basin,
            result["prithvi"]["test_mIoU"],
            result["resnet"]["test_mIoU"],
        ))
    with (out_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["basin", "prithvi_mIoU", "resnet_mIoU"])
        writer.writerows(rows)
    if rows:
        prithvi_mean = float(np.mean([row[1] for row in rows]))
        resnet_mean = float(np.mean([row[2] for row in rows]))
        print({
            "completed_folds": len(rows),
            "mean_prithvi_mIoU": prithvi_mean,
            "mean_resnet_mIoU": resnet_mean,
            "difference": prithvi_mean - resnet_mean,
        })


if __name__ == "__main__":
    main()
