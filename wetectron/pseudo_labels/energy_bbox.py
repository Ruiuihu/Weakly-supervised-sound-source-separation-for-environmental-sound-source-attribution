# -*- coding: utf-8 -*-
"""Energy-map connected-component bboxes seeded at click points (spectrogram WSOD)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

def _flood_component_mask(mask: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """4-connected component containing (cx, cy) on a boolean mask."""
    h, w = mask.shape
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))
    if not mask[cy, cx]:
        return np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=bool)
    stack = [(cy, cx)]
    out[cy, cx] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                stack.append((ny, nx))
    return out


def _connected_bbox_from_seed(
    mask: np.ndarray,
    cx: int,
    cy: int,
    min_side: int = 20,
) -> Optional[Tuple[float, float, float, float]]:
    """Return xyxy bbox for the connected component containing (cx, cy), or None."""
    h, w = mask.shape
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))
    comp = _flood_component_mask(mask, cx, cy)
    if not comp.any():
        return None
    ys, xs = np.where(comp)
    if ys.size == 0:
        return None
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if (xmax - xmin + 1) < min_side:
        pad = (min_side - (xmax - xmin + 1)) / 2.0
        xmin = max(0.0, xmin - pad)
        xmax = min(w - 1.0, xmax + pad)
    if (ymax - ymin + 1) < min_side:
        pad = (min_side - (ymax - ymin + 1)) / 2.0
        ymin = max(0.0, ymin - pad)
        ymax = min(h - 1.0, ymax + pad)
    return xmin, ymin, xmax, ymax


def _adaptive_threshold_mask(
    energy_hw: np.ndarray,
    cx: int,
    cy: int,
    init_thresh: float = 0.5,
    min_thresh: float = 0.15,
    step: float = 0.05,
) -> np.ndarray:
    """Binary mask: largest connected component at click with decreasing threshold."""
    h, w = energy_hw.shape
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))
    t = init_thresh
    while t >= min_thresh:
        mask = energy_hw >= t
        if mask[cy, cx]:
            return _flood_component_mask(mask, cx, cy)
        t -= step
    # fallback: small square around click
    r = 30
    out = np.zeros((h, w), dtype=bool)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    out[y0:y1, x0:x1] = True
    return out


def _downscale_for_cc(e: np.ndarray, max_side: int = 512) -> Tuple[np.ndarray, float, float]:
    """Return downscaled energy and (scale_x, scale_y) to map bboxes back."""
    h, w = e.shape
    scale_x, scale_y = 1.0, 1.0
    long_side = max(h, w)
    if long_side <= max_side:
        return e, scale_x, scale_y
    r = max_side / float(long_side)
    nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
    et = torch.from_numpy(e).unsqueeze(0).unsqueeze(0)
    et = torch.nn.functional.interpolate(et, size=(nh, nw), mode="bilinear", align_corners=False)
    scale_x = w / float(nw)
    scale_y = h / float(nh)
    return et.squeeze().numpy(), scale_x, scale_y


def boxes_from_energy_and_clicks(
    energy_1hw: torch.Tensor,
    clicks_xy: torch.Tensor,
    click_labels: torch.Tensor,
    image_size: Tuple[int, int],
    min_box_size: int = 20,
    energy_thresh: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One bbox per (click, class) via energy CC at click.

    Args:
        energy_1hw: [1, H, W] or [H, W] in [0, 1]
        clicks_xy: [N, 2] (x, y)
        click_labels: [N] class ids (1..C)
        image_size: (width, height)

    Returns:
        boxes [M, 4], labels [M], scores [M] (energy at center)
    """
    if energy_1hw.dim() == 3:
        energy_1hw = energy_1hw.squeeze(0)
    e = energy_1hw.detach().cpu().float().numpy()
    w, h = image_size
    if e.shape != (h, w):
        import torch.nn.functional as F

        et = torch.from_numpy(e).unsqueeze(0).unsqueeze(0)
        et = F.interpolate(et, size=(h, w), mode="bilinear", align_corners=False)
        e = et.squeeze().numpy()

    e_small, sx, sy = _downscale_for_cc(e)
    boxes_list: List[List[float]] = []
    labels_list: List[int] = []
    scores_list: List[float] = []

    if clicks_xy.numel() == 0:
        z4 = torch.zeros((0, 4), dtype=torch.float32)
        z1 = torch.zeros(0, dtype=torch.int64)
        zf = torch.zeros(0, dtype=torch.float32)
        return z4, z1, zf

    for i in range(clicks_xy.shape[0]):
        cx = int(clicks_xy[i, 0].item())
        cy = int(clicks_xy[i, 1].item())
        cls_id = int(click_labels[i].item())
        cx_s = int(np.clip(cx / sx, 0, e_small.shape[1] - 1))
        cy_s = int(np.clip(cy / sy, 0, e_small.shape[0] - 1))
        mask = _adaptive_threshold_mask(e_small, cx_s, cy_s, init_thresh=energy_thresh)
        bbox = _connected_bbox_from_seed(mask, cx_s, cy_s, min_side=max(4, int(min_box_size / max(sx, sy))))
        if bbox is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        boxes_list.append([xmin * sx, ymin * sy, xmax * sx, ymax * sy])
        labels_list.append(cls_id)
        scores_list.append(float(e[cy, cx]))

    if not boxes_list:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros(0, dtype=torch.int64),
            torch.zeros(0, dtype=torch.float32),
        )
    return (
        torch.tensor(boxes_list, dtype=torch.float32),
        torch.tensor(labels_list, dtype=torch.int64),
        torch.tensor(scores_list, dtype=torch.float32),
    )
