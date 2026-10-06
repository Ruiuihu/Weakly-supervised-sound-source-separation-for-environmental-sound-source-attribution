# -*- coding: utf-8 -*-
"""Green (low energy) -> multi-color gradient -> red (high) for feature/energy maps."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def feature_map_colormap_lut(size: int = 256) -> np.ndarray:
    """
    RGB LUT: green (0) -> cyan -> yellow -> orange -> red (1).
    Returns uint8 array [size, 3].
    """
    lut = np.zeros((size, 3), dtype=np.uint8)
    for i in range(size):
        x = i / max(size - 1, 1)
        if x < 0.25:
            u = x / 0.25
            r, g, b = 0, 255, int(255 * u)
        elif x < 0.5:
            u = (x - 0.25) / 0.25
            r, g, b = int(255 * u), 255, int(255 * (1 - u))
        elif x < 0.75:
            u = (x - 0.5) / 0.25
            r, g, b = 255, int(255 * (1 - 0.5 * u)), 0
        else:
            u = (x - 0.75) / 0.25
            r, g, b = 255, int(128 * (1 - u)), 0
        lut[i] = [r, g, b]
    return lut


def normalize_energy_01(
    energy: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Robust percentile normalize 2D energy to [0, 1]."""
    flat = energy.astype(np.float32).reshape(-1)
    lo = np.percentile(flat, p_low)
    hi = np.percentile(flat, p_high)
    return np.clip((energy.astype(np.float32) - lo) / max(hi - lo, eps), 0.0, 1.0)


def intensity_from_rgb_image(rgb: np.ndarray) -> np.ndarray:
    """Extract scalar energy proxy from a pseudo-colored spectrogram RGB image."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb must be HxWx3")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def apply_feature_colormap(
    energy_01: np.ndarray,
    lut: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Map normalized energy [0,1] to RGB uint8 HxWx3.
    """
    if lut is None:
        lut = feature_map_colormap_lut()
    t = np.clip(energy_01, 0.0, 1.0)
    idx = (t * (len(lut) - 1)).astype(np.uint8)
    return lut[idx]


def remap_spectrogram_image_to_feature_map(
    rgb: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remap a heat-style spectrogram PNG to green-low / red-high feature colors.

    Returns:
        feature_rgb (HxWx3 uint8), energy_01 (HxW float32)
    """
    intensity = intensity_from_rgb_image(rgb)
    energy_01 = normalize_energy_01(intensity, p_low=p_low, p_high=p_high)
    feature_rgb = apply_feature_colormap(energy_01)
    return feature_rgb, energy_01


def energy_tensor_to_rgb(
    energy_hw: "torch.Tensor",
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """[H,W] or [1,H,W] tensor in [0,1] or raw -> feature RGB uint8."""
    if torch is None:
        raise ImportError("torch required for energy_tensor_to_rgb")
    e = energy_hw.detach().float().cpu()
    if e.dim() == 3:
        e = e.squeeze(0)
    arr = e.numpy()
    if arr.max() > 1.0 + 1e-3 or arr.min() < -1e-3:
        arr = normalize_energy_01(arr, p_low=p_low, p_high=p_high)
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return apply_feature_colormap(arr)
