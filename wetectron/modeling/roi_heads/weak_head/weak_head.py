# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
from torch import nn
import numpy as np
import os

from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from ..box_head.loss import make_roi_box_loss_evaluator
from ..box_head.roi_box_predictors import make_roi_box_predictor
from ..box_head.inference import make_roi_box_post_processor as strong_roi_box_post_processor

from .roi_weak_predictors import make_roi_weak_predictor
from .inference import make_roi_box_post_processor as weak_roi_box_post_processor
from .loss import make_roi_weak_loss_evaluator, generate_img_label
from .roi_sampler import make_roi_sampler

from wetectron.modeling.utils import cat
from wetectron.structures.boxlist_ops import cat_boxlist
from wetectron.modeling.roi_heads.sim_head.sim_net import Sim_Net

class ROIWeakHead(torch.nn.Module):
    """
    Weak head without bounding-box regression (REGRESS_ON=False).
    Outputs classification scores on RPN proposals; boxes stay at proposal locations.
    """

    def __init__(self, cfg, in_channels):
        super(ROIWeakHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_weak_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = weak_roi_box_post_processor(cfg)
        loss_name = cfg.MODEL.ROI_WEAK_HEAD.LOSS
        if loss_name == "RoIRegLoss":
            from wetectron.modeling.registry import registry

            self.loss_evaluator = registry.ROI_WEAK_LOSS["RoILoss"](cfg)
        else:
            self.loss_evaluator = make_roi_weak_loss_evaluator(cfg)
        self.roi_sampler = (
            make_roi_sampler(cfg) if cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS != "none" else None
        )
        self.DB_METHOD = cfg.DB.METHOD

    def go_through_cdb(self, pooled_feats, proposals, model_cdb):
        if not self.training or self.DB_METHOD == "none":
            return pooled_feats
        if self.DB_METHOD == "concrete":
            return model_cdb(pooled_feats)
        if self.DB_METHOD == "dropblock":
            return self.feature_extractor.forward_dropblock(pooled_feats, proposals)
        if self.DB_METHOD == "attention":
            return self.feature_extractor.forward_attention_dropblock(pooled_feats, proposals)
        raise ValueError(self.DB_METHOD)

    def _predictor_outputs(self, x, proposals):
        out = self.predictor(x, proposals)
        if len(out) == 4:
            cls_score, det_score, ref_scores, _bbox_preds = out
            return cls_score, det_score, ref_scores
        return out[0], out[1], out[2]

    def forward(self, features, proposals, targets=None, model_cdb=None, iteration=None):
        """
        Same call signature as ROIWeakRegHead (iteration ignored).
        """
        if self.roi_sampler is not None and self.training:
            with torch.no_grad():
                proposals = self.roi_sampler(proposals, targets)

        if self.training and self.DB_METHOD != "none":
            _roi_feats, pooled_feats = self.feature_extractor.forward(features, proposals)
            aug_pooled_feats = self.go_through_cdb(pooled_feats, proposals, model_cdb=model_cdb)
            x = self.feature_extractor.forward_neck(aug_pooled_feats)
        else:
            x, _pooled_feats = self.feature_extractor(features, proposals)

        cls_score, det_score, ref_scores = self._predictor_outputs(x, proposals)
        if not self.training:
            if ref_scores is None:
                final_score = cls_score * det_score
            else:
                final_score = torch.mean(torch.stack(ref_scores), dim=0)
            result = self.post_processor(final_score, proposals)
            return x, result, {}, {}

        loss_img, accuracy_img = self.loss_evaluator(
            [cls_score], [det_score], ref_scores, proposals, targets
        )

        return x, proposals, loss_img, accuracy_img


class ROIWeakRegHead(torch.nn.Module):
    """ Generic Box Head class w/ regression. """
    def __init__(self, cfg, in_channels):
        super(ROIWeakRegHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_weak_predictor(cfg, self.feature_extractor.out_channels)
        self.loss_evaluator = make_roi_weak_loss_evaluator(cfg)
        self.weak_post_processor = weak_roi_box_post_processor(cfg)
        self.strong_post_processor = strong_roi_box_post_processor(cfg)

        self.HEUR = cfg.MODEL.ROI_WEAK_HEAD.REGRESS_HEUR
        self.roi_sampler = make_roi_sampler(cfg) if cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS != "none" else None
        self.DB_METHOD = cfg.DB.METHOD
        self.model_sim = Sim_Net(cfg, self.feature_extractor.out_channels)

    def go_through_cdb(self, pooled_feats, proposals, model_cdb):
        if not self.training or self.DB_METHOD == "none":
            return pooled_feats
        elif self.DB_METHOD == "concrete":
            return model_cdb(pooled_feats)
        elif self.DB_METHOD == "dropblock":
            return self.feature_extractor.forward_dropblock(pooled_feats, proposals)
        elif self.DB_METHOD == "attention":
            return self.feature_extractor.forward_attention_dropblock(pooled_feats, proposals)
        else:
            raise ValueError

    def forward(self, features, proposals, targets=None, model_cdb=None, iteration=None):
        debug_nan = os.environ.get("WSCL_DEBUG_NAN_TRACE", "0") == "1"

        def _check_tensor(name, tensor):
            if not debug_nan:
                return
            finite_mask = torch.isfinite(tensor)
            if finite_mask.all():
                print(
                    "[NaNTrace] {} OK shape={} min={:.6e} max={:.6e}".format(
                        name, tuple(tensor.shape), tensor.min().item(), tensor.max().item()
                    )
                )
                return
            finite_vals = tensor[finite_mask]
            if finite_vals.numel() > 0:
                finite_min = finite_vals.min().item()
                finite_max = finite_vals.max().item()
            else:
                finite_min = float("nan")
                finite_max = float("nan")
            raise RuntimeError(
                "[NaNTrace] {} BAD shape={} finite_ratio={:.6f} finite_min={:.6e} finite_max={:.6e}".format(
                    name,
                    tuple(tensor.shape),
                    finite_mask.float().mean().item(),
                    finite_min,
                    finite_max,
                )
            )

        def _check_module_params(name, module):
            if not debug_nan:
                return
            for p_name, p_val in module.named_parameters():
                if not torch.isfinite(p_val).all():
                    raise RuntimeError(
                        "[NaNTrace] {} parameter {} has non-finite values".format(name, p_name)
                    )

        # for partial labels
        if self.roi_sampler is not None and self.training:
            with torch.no_grad():
                proposals = self.roi_sampler(proposals, targets)

        clean_roi_feats, clean_pooled_feats = self.feature_extractor.forward(features, proposals)
        _check_tensor("clean_pooled_feats", clean_pooled_feats)
        _check_tensor("clean_roi_feats", clean_roi_feats)

        if self.training:
            sim_feature = self.model_sim(clean_roi_feats)
            _check_tensor("sim_feature", sim_feature)
            aug_pooled_feats = self.go_through_cdb(clean_pooled_feats, proposals, model_cdb=model_cdb)
            _check_tensor("aug_pooled_feats", aug_pooled_feats)
            aug_roi_feats = self.feature_extractor.forward_neck(aug_pooled_feats)
            _check_tensor("aug_roi_feats", aug_roi_feats)
            _check_module_params("predictor", self.predictor)
            cls_score, det_score, ref_scores, ref_bbox_preds = self.predictor(aug_roi_feats, proposals)
            _check_tensor("cls_score_logits", cls_score)
            _check_tensor("det_score_logits", det_score)
            for ridx, r in enumerate(ref_scores):
                _check_tensor("ref_scores[{}]_logits".format(ridx), r)
            for ridx, r in enumerate(ref_bbox_preds):
                _check_tensor("ref_bbox_preds[{}]".format(ridx), r)

        if not self.training:
            cls_score, det_score, ref_scores, ref_bbox_preds = self.predictor(clean_roi_feats, proposals)
            result = self.testing_forward(cls_score, det_score, proposals, ref_scores, ref_bbox_preds)
            return clean_roi_feats, result, {}, {}

        loss_img, accuracy_img = self.loss_evaluator([cls_score], [det_score], ref_scores, ref_bbox_preds, sim_feature, clean_pooled_feats, self.feature_extractor, self.model_sim, proposals, targets)

        return (aug_roi_feats, proposals, loss_img, accuracy_img)

    def testing_forward(self, cls_score, det_score, proposals, ref_scores=None, ref_bbox_preds=None):
        if self.HEUR == "WSDDN":
            final_score = cls_score * det_score
            result = self.weak_post_processor(final_score, proposals)
        elif self.HEUR == "CLS-AVG":
            final_score = torch.mean(torch.stack(ref_scores), dim=0)
            result = self.weak_post_processor(final_score, proposals)
        elif self.HEUR == "AVG": # AVG
            final_score = torch.mean(torch.stack(ref_scores), dim=0)
            final_regression = torch.mean(torch.stack(ref_bbox_preds), dim=0)
            result = self.strong_post_processor((final_score, final_regression), proposals, softmax_on=False)
        elif self.HEUR == "UNION": # UNION
            prop_list = [len(p) for p in proposals]
            ref_score_list = [rs.split(prop_list) for rs in ref_scores]
            ref_bbox_list = [rb.split(prop_list) for rb in ref_bbox_preds]
            final_score = [torch.cat((ref_score_list[0][i], ref_score_list[1][i], ref_score_list[2][i])) for i in range(len(proposals)) ]
            final_regression = [torch.cat((ref_bbox_list[0][i], ref_bbox_list[1][i], ref_bbox_list[2][i])) for i in range(len(proposals)) ]
            augmented_proposals = [cat_boxlist([p for _ in range(3)]) for p in proposals]
            result = self.strong_post_processor((cat(final_score), cat(final_regression)), augmented_proposals, softmax_on=False)
        else:
            raise ValueError
        return result


def build_roi_weak_head(cfg, in_channels):
    """
    Constructs a new weak head.
    REGRESS_ON=True  -> ROIWeakRegHead (MIST + box regression).
    REGRESS_ON=False -> ROIWeakHead (classification on proposals; use RoILoss + OICRPredictor).
    """
    if cfg.MODEL.ROI_WEAK_HEAD.REGRESS_ON:
        return ROIWeakRegHead(cfg, in_channels)

    cfg_cls = cfg.clone()
    cfg_cls.defrost()
    if cfg_cls.MODEL.ROI_WEAK_HEAD.LOSS == "RoIRegLoss":
        cfg_cls.MODEL.ROI_WEAK_HEAD.LOSS = "RoILoss"
    if cfg_cls.MODEL.ROI_WEAK_HEAD.PREDICTOR == "MISTPredictor":
        cfg_cls.MODEL.ROI_WEAK_HEAD.PREDICTOR = "OICRPredictor"
    cfg_cls.freeze()
    return ROIWeakHead(cfg_cls, in_channels)
