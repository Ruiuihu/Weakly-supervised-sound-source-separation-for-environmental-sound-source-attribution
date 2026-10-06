# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from wetectron.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import torchvision.transforms.functional as T_F
from ..roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
import torchvision.models as models
from ..cam.cam import Compute_Cam_Loss
from ..proposal_filters.energy_gate import (
    build_energy_probability_map_n11hw,
    energy_logits_from_probability,
    filter_proposals_by_energy_inference,
)
from wetectron.utils.roi_inputs import uses_precomputed_proposals

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        ### adjust stride for R50-C5 combination###
        if 'R-50-C5' in cfg.MODEL.BACKBONE['CONV_BODY']:
            self.backbone[0].layer4[0].downsample[0].stride = (1,1)
            self.backbone[0].layer4[0].conv1.stride = (1,1)
        elif 'R-101-C5' in cfg.MODEL.BACKBONE['CONV_BODY']:
            self.backbone[0].layer4[0].downsample[0].stride = (1,1)
            self.backbone[0].layer4[0].conv1.stride = (1,1)
        elif 'R-18-C5' in cfg.MODEL.BACKBONE['CONV_BODY']:
            self.backbone[0].layer4[0].downsample[0].stride = (1,1)
            self.backbone[0].layer4[0].conv1.stride = (1,1)

        if cfg.MODEL.FASTER_RCNN:
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.use_rpn = True
        else:
            self.use_rpn = False

        self.use_cam = False
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)


    def _energy_atten_out_hw(self, features):
        """
        Spatial size for spectrogram energy attention (RPN ATTN_PRIOR).
        FPN: use P4 (index 2, stride 16) to match legacy R-50-C4 resolution.
        """
        if (
            self.use_rpn
            and getattr(self.cfg.MODEL.RPN, "USE_FPN", False)
            and len(features) >= 3
        ):
            return features[2].shape[-2:]
        return features[0].shape[-2:]

    def _make_energy_atten_logits(self, images, out_hw, eps=1e-4):
        """
        Build attention logits from input spectrogram brightness/energy.
        Returns Tensor[N, 1, H_out, W_out]
        """
        x = images.tensors
        if x.ndim != 4:
            raise ValueError("images.tensors must be NCHW")
        energy_map = build_energy_probability_map_n11hw(x, self.cfg, size_hw=out_hw)
        return energy_logits_from_probability(energy_map, eps=eps)

    def forward(self, images, targets=None, rois=None, model_cdb=None, iteration=None, atten_logits=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        features = self.backbone(images.tensors)

        if uses_precomputed_proposals(rois):
            # use pre-computed proposals
            proposals = rois
            proposal_losses = {}
        else:
            use_attn = (
                hasattr(self.cfg.MODEL.RPN, "ATTN_PRIOR")
                and bool(self.cfg.MODEL.RPN.ATTN_PRIOR.ENABLED)
            )
            if use_attn and atten_logits is None:
                out_hw = self._energy_atten_out_hw(features)
                atten_logits = self._make_energy_atten_logits(images, out_hw)
            proposals, proposal_losses = self.rpn(images, features, targets, atten_logits=atten_logits)
            if not self.training:
                max_pi = int(getattr(self.cfg.TEST, "MAX_PROPOSALS_PER_IMAGE", 512))
                if max_pi > 0:
                    proposals = [
                        p if len(p) <= max_pi else p[:max_pi] for p in proposals
                    ]

        if (
            not self.training
            and self.roi_heads is not None
            and not uses_precomputed_proposals(rois)
            and hasattr(self.cfg.MODEL.ROI_WEAK_HEAD, "INFERENCE_ENERGY_GATE")
            and bool(self.cfg.MODEL.ROI_WEAK_HEAD.INFERENCE_ENERGY_GATE.ENABLED)
        ):
            proposals = filter_proposals_by_energy_inference(
                proposals, images.tensors, self.cfg
            )

        if self.roi_heads:
            x, result, detector_losses, accuracy = self.roi_heads(features, proposals, targets, model_cdb, iteration)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            return losses, accuracy

        return result

    def backbone_forward(self, images):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed

        Returns:
            features (list[Tensor]): the output from the backbone.
        """
        return self.backbone(images.tensors)

    def neck_head_forward(self, features, targets=None, rois=None, model_cdb=None):
        """
        Arguments:
            features (list[Tensor]): the output from the backbone.
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            the same as `forward`
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # use pre-computed proposals
        assert rois is not None
        assert rois[0] is not None
        x, result, detector_losses, accuracy = self.roi_heads(features, rois, targets, model_cdb)

        if self.training:
            return detector_losses, accuracy

        return result

    def resnet_head(self):
        net = models.resnet50()

        for i in range(2, 4):
            getattr(net, 'layer%d'%i)[0].conv1.stride = (2,2)
            getattr(net, 'layer%d'%i)[0].conv2.stride = (1,1)
            # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
            net.layer4[0].conv2.stride = (1,1)
            net.layer4[0].downsample[0].stride = (1,1)
        del net.avgpool, net.fc
        base = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
        neck = net.layer4
        return base, neck
