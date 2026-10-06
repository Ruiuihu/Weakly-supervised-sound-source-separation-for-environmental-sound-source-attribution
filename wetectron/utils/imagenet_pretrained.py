# --------------------------------------------------------
# Load torchvision ImageNet-1K ResNet-50 weights into wetectron R-50 backbone.body
# (stem + layer1..layer4). RPN / ROI heads stay randomly initialized.
# --------------------------------------------------------
from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_resnet_body(backbone: nn.Module) -> nn.Module:
    if hasattr(backbone, "body"):
        return backbone.body
    return backbone[0]


def _torchvision_resnet50_state_dict() -> Dict[str, torch.Tensor]:
    import torchvision.models as tvm

    try:
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1
        net = tvm.resnet50(weights=weights)
    except Exception:
        net = tvm.resnet50(pretrained=True)
    return net.state_dict()


def _map_tv_key_to_body_key(tv_key: str) -> str:
    if tv_key.startswith("conv1."):
        return "stem." + tv_key
    if tv_key.startswith("bn1."):
        return "stem." + tv_key
    return tv_key


def load_imagenet_resnet50_backbone(model: nn.Module) -> Tuple[int, int]:
    """
    Copy torchvision ResNet-50 ImageNet weights into ``model.backbone`` body.

    Returns:
        (num_tensors_loaded, num_body_params_not_overwritten)
    """
    cfg = model.cfg
    conv_body = cfg.MODEL.BACKBONE.CONV_BODY
    if "R-50" not in str(conv_body):
        raise ValueError(
            "ImageNet ResNet-50 preload is only valid for R-50 backbones; got CONV_BODY=%r"
            % (conv_body,)
        )

    body = _get_resnet_body(model.backbone)
    body_sd = body.state_dict()
    tv_sd = _torchvision_resnet50_state_dict()

    mapped: Dict[str, torch.Tensor] = {}
    for tv_k, tv_v in tv_sd.items():
        if tv_k.startswith("fc."):
            continue
        body_k = _map_tv_key_to_body_key(tv_k)
        if body_k not in body_sd:
            continue
        if body_sd[body_k].shape != tv_v.shape:
            logger.warning(
                "Skip %s -> %s: shape tv %s vs body %s",
                tv_k,
                body_k,
                tuple(tv_v.shape),
                tuple(body_sd[body_k].shape),
            )
            continue
        mapped[body_k] = tv_v

    incompatible = body.load_state_dict(mapped, strict=False)
    n_loaded = len(mapped)
    n_missing = len(incompatible.missing_keys)
    if incompatible.unexpected_keys:
        logger.warning("Unexpected keys when loading ImageNet weights: %s", incompatible.unexpected_keys)
    logger.info(
        "ImageNet ResNet-50: applied %d tensors to backbone body; %d body keys left uninitialized by this load",
        n_loaded,
        n_missing,
    )
    return n_loaded, n_missing
