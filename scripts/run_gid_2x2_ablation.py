#!/usr/bin/env python3
"""Run the complete four-cell GID resolution-by-label factorial ablation.

Refactored from notebook67e0991e34.ipynb. This script performs model inference
at native imagery and at simulated coarse resolution, scores fine and functional
labels, writes per-scene metrics, factorial contrasts, and an optional ANOVA.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from geoai_gid_yrb.factorial import summarize_scene_cells
from geoai_gid_yrb.labels import GID15_PALETTE, remap_15_to_6


IGNORE = 255
DEFAULT_SCENES = [
    "0000575925", "0001064454", "0001118839", "0001395956", "0001680858"
]
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def decode_label(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label)
    if label.ndim == 3 and label.shape[0] >= 3:
        rgb = np.transpose(label[:3], (1, 2, 0))
        output = np.full(rgb.shape[:2], IGNORE, dtype=np.uint8)
        for colour, index in GID15_PALETTE:
            output[np.all(rgb == np.asarray(colour), axis=-1)] = index
        return output
    band = label[0] if label.ndim == 3 else label
    output = np.full(band.shape, IGNORE, dtype=np.uint8)
    valid = (band >= 1) & (band <= 15)
    output[valid] = band[valid].astype(np.uint8)
    return output


def locate_scene(data_dir: str, scene: str, label: bool) -> str | None:
    matches = []
    for filename in glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True):
        lower = os.path.basename(filename).lower()
        if scene not in os.path.basename(filename):
            continue
        if not lower.endswith((".tif", ".tiff", ".png")):
            continue
        normalized = filename.replace(os.sep, "/").lower()
        is_label = "label" in lower or "/ann_dir/" in normalized
        if is_label == label:
            matches.append(filename)
    return sorted(matches)[0] if matches else None


def downsample_rgb(image: np.ndarray, factor: int) -> np.ndarray:
    height, width, channels = image.shape
    h = height // factor
    w = width // factor
    trimmed = image[: h * factor, : w * factor]
    blocks = trimmed.reshape(h, factor, w, factor, channels).astype(np.float32)
    return blocks.mean(axis=(1, 3)).astype(np.uint8)


def downsample_mode(label: np.ndarray, factor: int) -> np.ndarray:
    height, width = label.shape
    h = height // factor
    w = width // factor
    trimmed = label[: h * factor, : w * factor]
    blocks = trimmed.reshape(h, factor, w, factor).transpose(0, 2, 1, 3)
    output = np.zeros((h, w), dtype=label.dtype)
    for row in range(h):
        for column in range(w):
            values, counts = np.unique(blocks[row, column].ravel(), return_counts=True)
            output[row, column] = values[counts.argmax()]
    return output


def score(prediction: np.ndarray, reference: np.ndarray, classes: int, valid: np.ndarray) -> dict:
    predicted = prediction[valid]
    observed = reference[valid]
    oa = float((predicted == observed).mean())
    iou = []
    for class_index in range(1, classes + 1):
        intersection = np.logical_and(predicted == class_index, observed == class_index).sum()
        union = np.logical_or(predicted == class_index, observed == class_index).sum()
        if union > 0:
            iou.append(intersection / union)
    return {"oa": oa, "miou": float(np.mean(iou)) if iou else float("nan")}


def load_checkpoint(model, path: str, device: str):
    import torch
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--weights", required=True, help="15-output ResNet-50/U-Net checkpoint")
    parser.add_argument("--out-dir", default="outputs/gid_2x2")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--downscale", type=int, default=8)
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import pandas as pd
    import rasterio
    import torch
    import torch.nn.functional as functional
    import segmentation_models_pytorch as smp

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    model = smp.Unet(
        encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=15
    )
    load_checkpoint(model, args.weights, device)
    model = model.to(device).eval()

    @torch.no_grad()
    def infer(rgb: np.ndarray) -> np.ndarray:
        height, width, _ = rgb.shape
        normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
        prediction = np.zeros((height, width), dtype=np.uint8)
        for row in range(0, height, args.patch):
            for column in range(0, width, args.patch):
                r2 = min(row + args.patch, height)
                c2 = min(column + args.patch, width)
                ph, pw = r2 - row, c2 - column
                tile = functional.pad(
                    tensor[:, :, row:r2, column:c2],
                    (0, args.patch - pw, 0, args.patch - ph),
                )
                logits = model(tile)
                classes = logits.argmax(1)[0, :ph, :pw].cpu().numpy().astype(np.uint8) + 1
                prediction[row:r2, column:c2] = classes
        return prediction

    records = []
    metric_scene_maps = {"oa": {}, "miou": {}}
    for scene in args.scenes:
        image_file = locate_scene(args.scene_dir, scene, label=False)
        label_file = locate_scene(args.scene_dir, scene, label=True)
        if not image_file or not label_file:
            raise FileNotFoundError(f"Missing image or label for scene {scene}")

        with rasterio.open(image_file) as source:
            rgb = np.transpose(source.read([1, 2, 3]), (1, 2, 0))
        with rasterio.open(label_file) as source:
            reference = decode_label(source.read())

        height = min(rgb.shape[0], reference.shape[0])
        width = min(rgb.shape[1], reference.shape[1])
        rgb = rgb[:height, :width]
        reference = reference[:height, :width]

        native_prediction = infer(rgb)
        native_valid = (reference >= 1) & (reference <= 15)
        p1 = score(native_prediction, reference, 15, native_valid)
        p4 = score(
            remap_15_to_6(native_prediction), remap_15_to_6(reference), 6, native_valid
        )

        coarse_rgb = downsample_rgb(rgb, args.downscale)
        coarse_reference = downsample_mode(reference, args.downscale)
        coarse_valid = (coarse_reference >= 1) & (coarse_reference <= 15)
        coarse_prediction = infer(coarse_rgb)
        p2 = score(coarse_prediction, coarse_reference, 15, coarse_valid)
        p3 = score(
            remap_15_to_6(coarse_prediction),
            remap_15_to_6(coarse_reference),
            6,
            coarse_valid,
        )

        record = {
            "scene": scene,
            "P1_OA": p1["oa"], "P1_mIoU": p1["miou"],
            "P2_OA": p2["oa"], "P2_mIoU": p2["miou"],
            "P3_OA": p3["oa"], "P3_mIoU": p3["miou"],
            "P4_OA": p4["oa"], "P4_mIoU": p4["miou"],
        }
        records.append(record)
        for metric, suffix in (("oa", "OA"), ("miou", "mIoU")):
            metric_scene_maps[metric][scene] = {
                cell: record[f"{cell}_{suffix}"] for cell in ("P1", "P2", "P3", "P4")
            }
        print(record, flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(output_directory / "gid_2x2_per_scene.csv", index=False)
    summary = {
        metric: summarize_scene_cells(scene_values)
        for metric, scene_values in metric_scene_maps.items()
    }
    (output_directory / "gid_2x2_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        long_records = []
        design = [
            ("P1", "native", "fine"), ("P2", "coarse", "fine"),
            ("P3", "coarse", "coarse"), ("P4", "native", "coarse"),
        ]
        for _, row in frame.iterrows():
            for cell, resolution, labels in design:
                long_records.append({
                    "scene": row["scene"], "resolution": resolution,
                    "labels": labels, "OA": row[f"{cell}_OA"],
                })
        long_frame = pd.DataFrame(long_records)
        fitted = smf.ols("OA ~ C(resolution) * C(labels) + C(scene)", data=long_frame).fit()
        anova = sm.stats.anova_lm(fitted, typ=2)
        (output_directory / "gid_2x2_anova.txt").write_text(str(anova))
    except Exception as error:
        (output_directory / "gid_2x2_anova.txt").write_text(
            f"ANOVA was not generated: {type(error).__name__}: {error}\n"
        )

    print(f"Wrote outputs to {output_directory}")


if __name__ == "__main__":
    main()
