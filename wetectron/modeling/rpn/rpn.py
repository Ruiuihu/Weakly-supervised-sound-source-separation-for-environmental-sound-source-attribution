# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from wetectron.modeling.poolers import Pooler
from wetectron.modeling import registry
from wetectron.modeling.box_coder import BoxCoder
from wetectron.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor
from .utils import concat_box_prediction_layers
from wetectron.structures.boxlist_ops import remove_small_boxes
from wetectron.structures.boxlist_ops import boxlist_nms
from wetectron.structures.boxlist_ops import cat_boxlist
from wetectron.structures.bounding_box import BoxList
from wetectron.layers import smooth_l1_loss


def _zero_loss(tensors):
    """Finite zero that stays attached to every tensor in a list or a single tensor."""
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    total = None
    for tensor in tensors:
        term = tensor.sum() * 0.0
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        return x


@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)


        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.rpn_box_coder = rpn_box_coder
        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

    @staticmethod
    def _filter_invalid_boxlist(boxlist):
        boxes = boxlist.bbox
        finite = torch.isfinite(boxes).all(dim=1)
        non_degenerate = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        keep = finite & non_degenerate
        return boxlist[keep]

    def _make_click_pseudo_box_targets(self, targets):
        """
        Replace bbox with fixed (W,H) boxes centered on ``click`` keypoints so the RPN
        can use the standard IoU matcher + loss_objectness / loss_rpn_box_reg.
        Original ``targets`` are not modified; ROI heads still see unmodified boxes upstream.
        """
        pw = float(self.cfg.MODEL.RPN.CLICK_PSEUDO_BOX_W)
        ph = float(self.cfg.MODEL.RPN.CLICK_PSEUDO_BOX_H)
        half_w = pw * 0.5
        half_h = ph * 0.5
        out = []
        for t in targets:
            if not t.has_field("click"):
                out.append(t)
                continue
            kp = t.get_field("click").keypoints
            if kp.numel() == 0:
                out.append(t)
                continue
            clicks = kp[:, :2]
            n = clicks.shape[0]
            x1 = clicks[:, 0] - half_w
            y1 = clicks[:, 1] - half_h
            x2 = clicks[:, 0] + half_w
            y2 = clicks[:, 1] + half_h
            bbox = torch.stack([x1, y1, x2, y2], dim=1)
            bl = BoxList(bbox, t.size, mode="xyxy")
            bl.clip_to_image(remove_empty=False)
            bl = self._filter_invalid_boxlist(bl)
            if len(bl) != n:
                out.append(t)
                continue
            if t.has_field("labels"):
                lbl = t.get_field("labels")
                if lbl.shape[0] == n:
                    bl.add_field("labels", lbl)
            if t.has_field("difficult"):
                df = t.get_field("difficult")
                if df.shape[0] == n:
                    bl.add_field("difficult", df)
            bl.add_field("click", t.get_field("click"))
            out.append(bl)
        return out

    def _point_supervised_rpn_loss(self, anchors, objectness, box_regression, targets):
        anchors_per_image = [cat_boxlist(anchors_img) for anchors_img in anchors]
        flat_anchors = torch.cat([a.bbox for a in anchors_per_image], dim=0)

        labels = []
        center_targets = []
        for anchors_img, targets_img in zip(anchors_per_image, targets):
            num_anchors = anchors_img.bbox.shape[0]
            labels_img = anchors_img.bbox.new_zeros((num_anchors,), dtype=torch.float32)
            centers_img = anchors_img.bbox.new_zeros((num_anchors, 2))

            if targets_img.has_field("click"):
                clicks = targets_img.get_field("click").keypoints
                if clicks.numel() > 0:
                    clicks = clicks[:, :2]
                    clicks_tiled = torch.unsqueeze(torch.cat((clicks, clicks), dim=1), dim=1)
                    num_obj = clicks.shape[0]
                    box_repeat = torch.cat([anchors_img.bbox.unsqueeze(0) for _ in range(num_obj)], dim=0)
                    diff = clicks_tiled - box_repeat
                    matched_ids = (
                        (diff[:, :, 0] > 0)
                        * (diff[:, :, 1] > 0)
                        * (diff[:, :, 2] < 0)
                        * (diff[:, :, 3] < 0)
                    )
                    matched_idxs = matched_ids.float().argmax(0)
                    matched_idxs[matched_ids.sum(0) == 0] = -1
                    pos_inds = matched_idxs >= 0
                    labels_img[pos_inds] = 1.0
                    if pos_inds.any():
                        centers_img[pos_inds] = clicks[matched_idxs[pos_inds]]

            labels.append(labels_img)
            center_targets.append(centers_img)

        labels = torch.cat(labels, dim=0)
        center_targets = torch.cat(center_targets, dim=0)
        objectness, box_regression = concat_box_prediction_layers(objectness, box_regression)
        objectness = objectness.squeeze(1)

        objectness_loss = F.binary_cross_entropy_with_logits(objectness, labels)

        decoded_boxes = self.rpn_box_coder.decode(box_regression, flat_anchors)
        pred_centers = (decoded_boxes[:, :2] + decoded_boxes[:, 2:]) * 0.5
        pos_inds = labels > 0
        if pos_inds.any():
            center_loss = smooth_l1_loss(
                pred_centers[pos_inds],
                center_targets[pos_inds],
                beta=1.0 / 9,
                size_average=False,
            ) / pos_inds.sum().float()
        else:
            center_loss = box_regression.sum() * 0.0

        return {
            "loss_objectness_point": objectness_loss,
            "loss_rpn_center": center_loss,
        }

    def forward(self, images, features, targets=None, atten_logits=None, atten_map=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        rpn_targets = targets
        if (
            self.training
            and targets is not None
            and bool(getattr(self.cfg.MODEL.RPN, "USE_CLICK_PSEUDO_BOX_FOR_TRAIN", False))
        ):
            rpn_targets = self._make_click_pseudo_box_targets(targets)

        use_click_pseudo = bool(
            getattr(self.cfg.MODEL.RPN, "USE_CLICK_PSEUDO_BOX_FOR_TRAIN", False)
        )

        def _standard_rpn():
            objectness, rpn_box_regression = self.head(features)
            anchors = self.anchor_generator(images, features)
            if self.training:
                if self.cfg.MODEL.RPN_ONLY:
                    boxes = anchors
                else:
                    with torch.no_grad():
                        boxes = self.box_selector_train(
                            anchors, objectness, rpn_box_regression, rpn_targets
                        )

                supervise_with_boxes = bool(
                    getattr(self.cfg.MODEL.RPN, "SUPERVISE_WITH_BOXES", False)
                )
                if not supervise_with_boxes:
                    # Recording-level training: keep proposal generation, and keep
                    # box/click coordinates out of the RPN loss.
                    losses = {
                        "loss_objectness": _zero_loss(objectness),
                        "loss_rpn_box_reg": _zero_loss(rpn_box_regression),
                    }
                    return boxes, losses

                use_point_supervision = (
                    targets is not None
                    and self.cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS == "point"
                    and all(t.has_field("click") for t in targets)
                    and not use_click_pseudo
                )
                if use_point_supervision:
                    losses = self._point_supervised_rpn_loss(
                        anchors, objectness, rpn_box_regression, targets
                    )
                else:
                    loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
                        anchors, objectness, rpn_box_regression, rpn_targets
                    )
                    losses = {
                        "loss_objectness": loss_objectness,
                        "loss_rpn_box_reg": loss_rpn_box_reg,
                    }
                return boxes, losses
            return self._forward_test(anchors, objectness, rpn_box_regression)

        def _atten_rpn(atten_logits_tensor):
            min_size = int(getattr(self.cfg.MODEL.RPN.ATTN_PRIOR, "MIN_SIZE", 20))
            nms_thresh = float(getattr(self.cfg.MODEL.RPN.ATTN_PRIOR, "NMS_THRESH", 0.7))
            max_props = int(getattr(self.cfg.MODEL.RPN.ATTN_PRIOR, "MAX_PROPOSALS", 2000))

            result = []
            anchors = self.anchor_generator(images, features, atten_logits_tensor)
            for anchor_levels, atten_logit in zip(anchors, atten_logits_tensor):
                img_size = anchor_levels[0].size
                if self.cfg.MODEL.RPN.USE_FPN and len(anchor_levels) > 2:
                    # P4 (stride 16): same nominal scale as legacy C4; avoid fusing all FPN levels (too slow).
                    level_anchors = anchor_levels[2]
                else:
                    level_anchors = anchor_levels[0]
                boxes = level_anchors.bbox
                boxlist = BoxList(boxes, img_size, mode="xyxy")
                boxlist = boxlist.clip_to_image(remove_empty=False)
                boxlist = self._filter_invalid_boxlist(boxlist)
                boxlist = boxlist.clip_to_image(remove_empty=False)
                boxlist = self._filter_invalid_boxlist(boxlist)
                boxlist = remove_small_boxes(boxlist, min_size)

                if len(boxlist) == 0:
                    boxlist.add_field("objectness", torch.zeros((0,), device=boxes.device))
                    result.append(boxlist)
                    continue

                atten_chw = atten_logit
                if atten_chw.dim() == 2:
                    atten_chw = atten_chw.unsqueeze(0)
                objectness = (
                    self.pooler([atten_chw.unsqueeze(0)], [boxlist])
                    .mean(3)
                    .mean(2)
                    .squeeze()
                    .sigmoid()
                )
                boxlist.add_field("objectness", objectness)
                boxlist = boxlist_nms(
                    boxlist,
                    nms_thresh,
                    max_proposals=max_props,
                    score_field="objectness",
                )
                result.append(boxlist)
            return result

        # Always run standard RPN; optionally fuse attention-guided proposals.
        boxes_std, losses = _standard_rpn()

        use_attn = (
            hasattr(self.cfg.MODEL.RPN, "ATTN_PRIOR")
            and bool(self.cfg.MODEL.RPN.ATTN_PRIOR.ENABLED)
            and atten_logits is not None
        )
        if not use_attn:
            return boxes_std, losses

        # Sparsity insurance: skip attention if energy map too sparse.
        sparsity_thresh = float(getattr(self.cfg.MODEL.RPN.ATTN_PRIOR, "SPARSITY_THRESH", 0.0))
        if sparsity_thresh > 0:
            with torch.no_grad():
                p = atten_logits.sigmoid()
                # fraction of pixels above 0.5
                frac = (p > 0.5).float().mean().item()
            if frac < sparsity_thresh:
                return boxes_std, losses

        boxes_attn = _atten_rpn(atten_logits)

        # Fusion: concatenate + NMS + TopK, with minimum keep fallback.
        min_keep = int(getattr(self.cfg.MODEL.RPN.ATTN_PRIOR, "MIN_KEEP_PER_IMAGE", 0))
        fused = []
        for std_box, attn_box in zip(boxes_std, boxes_attn):
            if len(attn_box) == 0:
                fused_box = std_box
            else:
                fused_box = cat_boxlist((std_box, attn_box))
                fused_box = boxlist_nms(
                    fused_box,
                    float(self.cfg.MODEL.RPN.NMS_THRESH),
                    max_proposals=-1,
                    score_field="objectness",
                )

            if min_keep and len(fused_box) < min_keep:
                # fallback: ensure at least min_keep proposals using standard scores
                obj = std_box.get_field("objectness")
                k = min(min_keep, len(std_box))
                if k > 0:
                    _, inds = torch.topk(obj, k, dim=0, largest=True, sorted=True)
                    fused_box = std_box[inds]
            fused.append(fused_box)

        return fused, losses

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(                  ### inference.py - forward
                    anchors, objectness, rpn_box_regression, targets
                )
        #loss_objectness, loss_rpn_box_reg = self.loss_evaluator(  ### loss.py - __call__
        #    anchors, objectness, rpn_box_regression, targets
        #)
        #losses = {
        #    "loss_objectness": loss_objectness,
        #    "loss_rpn_box_reg": loss_rpn_box_reg,
        #}
        losses = {}
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels)
