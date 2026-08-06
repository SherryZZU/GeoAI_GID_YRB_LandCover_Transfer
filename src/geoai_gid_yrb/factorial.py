"""Utilities for the complete 2 x 2 resolution-by-label ablation.

Cell definitions from notebook67e0991e34.ipynb:
P1 = native resolution, fine labels
P2 = coarse resolution, fine labels
P3 = coarse resolution, coarse labels
P4 = native resolution, coarse labels
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np


CELL_DESIGN = {
    "P1": {"resolution": "native", "labels": "fine"},
    "P2": {"resolution": "coarse", "labels": "fine"},
    "P3": {"resolution": "coarse", "labels": "coarse"},
    "P4": {"resolution": "native", "labels": "coarse"},
}


def factorial_effects(cell_values: Mapping[str, float]) -> dict:
    """Calculate cell means, two marginal contrasts, and the interaction.

    The signs follow the source notebook:
    resolution = native minus coarse
    labels = fine minus coarse
    interaction = resolution effect at fine labels minus resolution effect at coarse labels
    """
    missing = set(CELL_DESIGN) - set(cell_values)
    if missing:
        raise KeyError(f"Missing factorial cells: {sorted(missing)}")
    m = {key: float(cell_values[key]) for key in CELL_DESIGN}
    resolution = ((m["P1"] + m["P4"]) - (m["P2"] + m["P3"])) / 2.0
    labels = ((m["P1"] + m["P2"]) - (m["P3"] + m["P4"])) / 2.0
    interaction = (m["P1"] - m["P2"]) - (m["P4"] - m["P3"])
    return {
        "cell_values": m,
        "main_effect_resolution_native_minus_coarse": float(resolution),
        "main_effect_labels_fine_minus_coarse": float(labels),
        "interaction": float(interaction),
        "dominant_absolute_main_effect": (
            "resolution" if abs(resolution) > abs(labels) else "labels"
        ),
    }


def summarize_scene_cells(per_scene: Mapping[str, Mapping[str, float]]) -> dict:
    """Aggregate P1-P4 values over scenes and return mean, SD, and effects."""
    if not per_scene:
        raise ValueError("per_scene cannot be empty")
    summary = {}
    means = {}
    for cell in CELL_DESIGN:
        values = np.asarray([scene[cell] for scene in per_scene.values()], dtype=float)
        means[cell] = float(values.mean())
        summary[cell] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n": int(values.size),
        }
    return {"cells": summary, "effects": factorial_effects(means)}
