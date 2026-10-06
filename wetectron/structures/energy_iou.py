# -*- coding: utf-8 -*-
"""Spectrogram energy-weighted IoU for eval-only mAP / diagnosis (not VOC standard)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from wetectron.modeling.proposal_filters.energy_gate import build_energy_probability_map_n11hw

IOU_MODES = ("geom", "energy_mass", "energy_geom")


def energy_map_from_image_tensors(image_tensors: torch.Tensor, cfg) -> torch.Tensor:
    """
    Args:
        image_tensors: [N, C, H, W] (ImageList.tensors)
    Returns:
        [N, H, W] energy in [0, 1]
    """
    emap = build_energy_probability_map_n11hw(image_tensors, cfg)
    return emap[:, 0, :, :]


def _box_slices(box: torch.Tensor, h: int, w: int) -> Tuple[int, int, int, int]:
    x1 = int(torch.floor(box[0]).clamp(0, w - 1).item())
    y1 = int(torch.floor(box[1]).clamp(0, h - 1).item())
    x2 = int(torch.ceil(box[2]).clamp(0, w).item()) - 1
    y2 = int(torch.ceil(box[3]).clamp(0, h).item()) - 1
    x2 = max(x2, x1)
    y2 = max(y2, y1)
    return x1, y1, x2, y2


def energy_mass_in_box(energy_2d: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Sum of energy inside box (xyxy, same coords as energy_2d)."""
    h, w = energy_2d.shape[-2], energy_2d.shape[-1]
    x1, y1, x2, y2 = _box_slices(box, h, w)
    patch = energy_2d[y1 : y2 + 1, x1 : x2 + 1]
    if patch.numel() == 0:
        return energy_2d.new_zeros(())
    return patch.sum()


def mean_energy_in_box(energy_2d: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    h, w = energy_2d.shape[-2], energy_2d.shape[-1]
    x1, y1, x2, y2 = _box_slices(box, h, w)
    patch = energy_2d[y1 : y2 + 1, x1 : x2 + 1]
    if patch.numel() == 0:
        return energy_2d.new_zeros(())
    return patch.mean()


def geom_iou_tensor(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Standard axis-aligned IoU; box [4], boxes [N, 4]."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=box.device)
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter_w = (x2 - x1 + 1).clamp(min=0)
    inter_h = (y2 - y1 + 1).clamp(min=0)
    inter = inter_w * inter_h
    area_box = (box[2] - box[0] + 1).clamp(min=0) * (box[3] - box[1] + 1).clamp(min=0)
    area_boxes = (boxes[:, 2] - boxes[:, 0] + 1).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1] + 1
    ).clamp(min=0)
    union = area_box + area_boxes - inter
    return inter / union.clamp(min=1e-8)


def energy_mass_iou_single(
    energy_2d: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8
) -> float:
    """E-IoU_mass = sum(e in inter) / sum(e in union)."""
    h, w = energy_2d.shape[-2], energy_2d.shape[-1]
    px1, py1, px2, py2 = _box_slices(pred, h, w)
    gx1, gy1, gx2, gy2 = _box_slices(gt, h, w)
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    inter_e = energy_2d.new_zeros(())
    if ix1 <= ix2 and iy1 <= iy2:
        inter_e = energy_2d[iy1 : iy2 + 1, ix1 : ix2 + 1].sum()
    mass_p = energy_mass_in_box(energy_2d, pred)
    mass_g = energy_mass_in_box(energy_2d, gt)
    union_e = mass_p + mass_g - inter_e
    return float((inter_e / union_e.clamp(min=eps)).item())


def energy_geom_iou_single(
    energy_2d: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8
) -> float:
    """geom IoU * sqrt(mean_e_pred * mean_e_gt)."""
    giou = float(geom_iou_tensor(pred, gt.unsqueeze(0)).item())
    mp = mean_energy_in_box(energy_2d, pred)
    mg = mean_energy_in_box(energy_2d, gt)
    factor = torch.sqrt(mp.clamp(min=0) * mg.clamp(min=0)).item()
    return giou * factor


def pairwise_iou_tensor(
    pred: torch.Tensor,
    gts: torch.Tensor,
    energy_2d: Optional[torch.Tensor],
    iou_mode: str = "geom",
) -> torch.Tensor:
    """
    IoU between one pred and N GT boxes.
    energy_2d required for energy_mass / energy_geom.
    """
    if iou_mode not in IOU_MODES:
        raise ValueError(f"iou_mode must be one of {IOU_MODES}, got {iou_mode!r}")
    if gts.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=pred.device)

    if iou_mode == "geom":
        return geom_iou_tensor(pred, gts)

    assert energy_2d is not None, f"energy_2d required for iou_mode={iou_mode}"
    out = []
    for j in range(gts.shape[0]):
        if iou_mode == "energy_mass":
            out.append(energy_mass_iou_single(energy_2d, pred, gts[j]))
        else:
            out.append(energy_geom_iou_single(energy_2d, pred, gts[j]))
    return torch.tensor(out, dtype=torch.float32, device=pred.device)


def max_iou_xyxy_mode(
    pred_box: torch.Tensor,
    gt_boxes: torch.Tensor,
    energy_2d: Optional[torch.Tensor],
    iou_mode: str = "geom",
) -> torch.Tensor:
    """Max IoU of pred vs gt_boxes (for diagnose_eval)."""
    if gt_boxes.numel() == 0:
        return torch.tensor(0.0)
    ious = pairwise_iou_tensor(pred_box, gt_boxes, energy_2d, iou_mode=iou_mode)
    return ious.max()
