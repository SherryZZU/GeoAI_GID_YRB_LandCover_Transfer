"""Segmentation metrics derived from the GID and LOBO notebooks."""
from __future__ import annotations
from typing import Iterable, Mapping
import numpy as np


def confusion_matrix(reference: np.ndarray, prediction: np.ndarray, num_classes: int,
                     ignore_index: int | None = 0) -> np.ndarray:
    ref = np.asarray(reference).astype(np.int64).ravel()
    pred = np.asarray(prediction).astype(np.int64).ravel()
    if ref.shape != pred.shape:
        raise ValueError("reference and prediction must have identical shapes")
    valid = (ref >= 0) & (ref < num_classes) & (pred >= 0) & (pred < num_classes)
    if ignore_index is not None:
        valid &= ref != ignore_index
    encoded = ref[valid] * num_classes + pred[valid]
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm: np.ndarray, ignore_index: int | None = 0) -> dict:
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    valid = union > 0
    if ignore_index is not None and 0 <= ignore_index < len(valid):
        valid[ignore_index] = False
    denominator = cm.sum()
    if ignore_index is not None and 0 <= ignore_index < cm.shape[0]:
        denominator -= cm[ignore_index].sum()
    oa = float(tp.sum() / denominator) if denominator > 0 else float('nan')
    return {
        "overall_accuracy": oa,
        "mean_iou": float(np.nanmean(iou[valid])) if np.any(valid) else float('nan'),
        "per_class_iou": [float(v) if np.isfinite(v) else None for v in iou],
        "confusion": cm.astype(int).tolist(),
    }


def segmentation_metrics(reference: np.ndarray, prediction: np.ndarray, num_classes: int,
                         ignore_index: int | None = 0) -> dict:
    return metrics_from_confusion(
        confusion_matrix(reference, prediction, num_classes, ignore_index), ignore_index
    )


def user_producer_accuracy(cm: np.ndarray, class_names: Mapping[int,str] | None = None,
                           class_offset: int = 0) -> dict:
    """Return user accuracy (precision) and producer accuracy (recall)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    ua = np.divide(tp, cm.sum(axis=0), out=np.full_like(tp, np.nan), where=cm.sum(axis=0)>0)
    pa = np.divide(tp, cm.sum(axis=1), out=np.full_like(tp, np.nan), where=cm.sum(axis=1)>0)
    names = [class_names.get(i+class_offset, str(i+class_offset)) if class_names else str(i+class_offset)
             for i in range(cm.shape[0])]
    return {
        "user_accuracy": dict(zip(names, ua.tolist())),
        "producer_accuracy": dict(zip(names, pa.tolist())),
        "balanced_accuracy": float(np.nanmean(pa)),
    }


def area_weighted_oa(producer_accuracy: Mapping[str,float], area_weights: Mapping[str,float]) -> float:
    total = 0.0
    for key, weight in area_weights.items():
        value = producer_accuracy.get(key, float('nan'))
        if np.isfinite(value):
            total += float(weight) * float(value)
    return float(total)
