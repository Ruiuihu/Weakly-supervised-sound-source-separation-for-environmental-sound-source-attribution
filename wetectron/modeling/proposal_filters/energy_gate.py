# -*- coding: utf-8 -*-
"""
Spectrogram energy map (same construction as RPN ATTN_PRIOR) for class-agnostic
proposal gating at inference.
"""
from typing import List

import torch
import torch.nn.functional as F

from wetectron.structures.bounding_box import BoxList


def robust_normalize_2d(x2d: torch.Tensor, p_low: float, p_high: float, eps: float = 1e-6) -> torch.Tensor:
    flat = x2d.reshape(-1)
    n = flat.numel()
    if n == 0:
        return x2d
    k_low = max(1, int(round(p_low * (n - 1))) + 1)
    k_high = max(1, int(round(p_high * (n - 1))) + 1)
    k_low = min(k_low, n)
    k_high = min(max(k_high, k_low), n)
    v_low = torch.kthvalue(flat, k_low).values
    v_high = torch.kthvalue(flat, k_high).values
    denom = (v_high - v_low).clamp(min=eps)
    return ((x2d - v_low) / denom).clamp(0.0, 1.0)


def build_energy_probability_map_n11hw(
    image_tensors: torch.Tensor,
    cfg,
    size_hw=None,
) -> torch.Tensor:
    """
    Same energy definition as GeneralizedRCNN._make_energy_atten_logits, but returns
    probabilities in (0,1) instead of logits.

    Args:
        image_tensors: [N, C, H, W] (typically ImageList.tensors)
        cfg: full config (reads MODEL.RPN.ATTN_PRIOR)
        size_hw: optional (H, W) to interpolate to; default is input tensor H,W

    Returns:
        Tensor [N, 1, H, W] with values in [0, 1]
    """
    if image_tensors.ndim != 4:
        raise ValueError("image_tensors must be NCHW")
    atten_cfg = cfg.MODEL.RPN.ATTN_PRIOR
    p_low = float(atten_cfg.P_LOW)
    p_high = float(atten_cfg.P_HIGH)
    smooth_k = int(atten_cfg.SMOOTH_K)

    energy = image_tensors.float().mean(dim=1)  # [N, H, W]
    normed = []
    for i in range(energy.shape[0]):
        e = energy[i]
        e = robust_normalize_2d(e, p_low=p_low, p_high=p_high)
        if smooth_k and smooth_k > 1:
            e = F.avg_pool2d(
                e.unsqueeze(0).unsqueeze(0),
                kernel_size=smooth_k,
                stride=1,
                padding=smooth_k // 2,
            ).squeeze(0).squeeze(0)
        normed.append(e)
    energy_map = torch.stack(normed, dim=0).unsqueeze(1)  # [N,1,H,W]
    if size_hw is not None:
        energy_map = F.interpolate(energy_map, size=size_hw, mode="bilinear", align_corners=False)
    return energy_map.clamp(0.0, 1.0)


def energy_logits_from_probability(emap: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """emap in [0,1] -> logits for RPN attention branch."""
    p = emap.clamp(min=eps, max=1.0 - eps)
    return torch.log(p / (1.0 - p))


def _mean_in_box_energy(energy_1hw: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """
    energy_1hw: [1, H, W]
    boxes_xyxy: [N, 4] same device
    """
    H, W = energy_1hw.shape[-2], energy_1hw.shape[-1]
    n = boxes_xyxy.shape[0]
    if n == 0:
        return boxes_xyxy.new_zeros((0,))
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    xi1 = torch.clamp(torch.floor(x1).long(), 0, W - 1)
    yi1 = torch.clamp(torch.floor(y1).long(), 0, H - 1)
    xi2 = torch.clamp(torch.ceil(x2).long() - 1, 0, W - 1)
    yi2 = torch.clamp(torch.ceil(y2).long() - 1, 0, H - 1)
    xi2 = torch.maximum(xi2, xi1)
    yi2 = torch.maximum(yi2, yi1)
    means = []
    e0 = energy_1hw[0]
    for i in range(n):
        patch = e0[yi1[i] : yi2[i] + 1, xi1[i] : xi2[i] + 1]
        means.append(patch.mean() if patch.numel() > 0 else e0.new_zeros(()))
    return torch.stack(means, dim=0)


def filter_proposals_by_energy_inference(
    proposals: List[BoxList],
    image_tensors: torch.Tensor,
    cfg,
) -> List[BoxList]:
    """
    Per-image: score each proposal by mean energy inside its box; apply threshold,
    optional TOPK, MAX cap, and MIN_KEEP fallback. Class-agnostic.

    Proposals must live in the same (W,H) coordinate system as image_tensors spatial dims.
    """
    gate = cfg.MODEL.ROI_WEAK_HEAD.INFERENCE_ENERGY_GATE
    if not bool(getattr(gate, "ENABLED", False)):
        return proposals

    min_mean = float(getattr(gate, "MIN_MEAN_ENERGY", 0.0))
    topk = int(getattr(gate, "TOPK_PER_IMAGE", 0))
    max_props = int(getattr(gate, "MAX_PROPOSALS_PER_IMAGE", 512))
    min_keep = int(getattr(gate, "MIN_KEEP_PER_IMAGE", 32))

    Ht, Wt = image_tensors.shape[-2], image_tensors.shape[-1]
    energy_n = build_energy_probability_map_n11hw(image_tensors, cfg, size_hw=(Ht, Wt))

    out = []
    for i, prop in enumerate(proposals):
        if prop is None or len(prop) == 0:
            out.append(prop)
            continue
        dev = prop.bbox.device
        e1 = energy_n[i : i + 1].to(device=dev, dtype=torch.float32)
        boxes = prop.bbox
        scores = _mean_in_box_energy(e1, boxes)

        n = scores.shape[0]
        order_full = torch.argsort(scores, descending=True)

        sorted_idx = order_full
        if min_mean > 1e-8:
            sorted_idx = order_full[scores[order_full] >= min_mean]
            if sorted_idx.numel() == 0:
                sorted_idx = order_full[: min(min_keep, n)]

        if topk > 0:
            sorted_idx = sorted_idx[: min(topk, sorted_idx.numel())]

        if sorted_idx.numel() > max_props:
            sorted_idx = sorted_idx[:max_props]

        if sorted_idx.numel() < min_keep and n > 0:
            sorted_idx = order_full[: min(min_keep, n)]
            if sorted_idx.numel() > max_props:
                sorted_idx = sorted_idx[:max_props]

        out.append(prop[sorted_idx.long()])
    return out
