import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasetbuild import build_spec_dataset
from wetectron.config.defaults import _C
from wetectron.modeling.cdb import ConvConcreteDB
from wetectron.modeling.detector import build_detection_model
from wetectron.solver import (
    make_cdb_optimizer,
    make_lr_cdb_scheduler,
    make_lr_scheduler,
    make_optimizer,
)
from wetectron.modeling.proposal_filters.energy_gate import filter_proposals_by_energy_inference
from wetectron.structures.image_list import to_image_list
from wetectron.structures.energy_iou import (
    IOU_MODES,
    energy_map_from_image_tensors,
    pairwise_iou_tensor,
)
from wetectron.utils.checkpoint import DetectronCheckpointer
from wetectron.utils.roi_inputs import eval_rois_argument
from wetectron.utils.detectron2_weight_remap import remap_detectron2_r50_c4_state_dict_if_needed
from wetectron.utils.imagenet_pretrained import load_imagenet_resnet50_backbone
from wetectron.utils.imports import import_file
from wetectron.utils.model_zoo import cache_url


def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_COCO_PRETRAINED_DIR = os.path.join(_REPO_ROOT, "pretrained_coco")
# Same host/path prefix as wetectron.config.paths_catalog.ModelCatalog.S3_C2_DETECTRON_URL
_DETECTRON2_FB_PREFIX = "https://dl.fbaipublicfiles.com/detectron2"


def apply_download_proxy(proxy_url: str) -> None:
    """Set env vars so urllib / torch.hub use HTTP(S) CONNECT proxy (e.g. Clash mixed port)."""
    if not proxy_url or not str(proxy_url).strip():
        return
    u = str(proxy_url).strip()
    os.environ["HTTP_PROXY"] = u
    os.environ["HTTPS_PROXY"] = u
    os.environ.setdefault("http_proxy", u)
    os.environ.setdefault("https_proxy", u)
    print(f"Download proxy: {u}")


# Public training profile: recording-level tags only. Box and click coordinates
# stay available for evaluation, and are not RPN or MIL training targets.
RECORDING_LEVEL_OPTS = [
    "MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS",
    "none",
    "MODEL.ROI_WEAK_HEAD.USE_EXTERNAL_PSEUDO",
    "False",
    "MODEL.RPN.USE_CLICK_PSEUDO_BOX_FOR_TRAIN",
    "False",
    "MODEL.RPN.SUPERVISE_WITH_BOXES",
    "False",
    "SOLVER.CONTRA",
    "True",
    "MODEL.ROI_WEAK_HEAD.OICR_P",
    "0.0",
    "MODEL.ROI_WEAK_HEAD.REGRESS_ON",
    "True",
    "MODEL.RPN.ATTN_PRIOR.ENABLED",
    "True",
    "MODEL.ROI_WEAK_HEAD.INFERENCE_ENERGY_GATE.ENABLED",
    "False",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train WSCL model with model + model_cdb.")
    parser.add_argument("--image-root", type=str, default="", help="Spectrogram image root (required for training).")
    parser.add_argument(
        "--label-csv",
        type=str,
        default="",
        help="Recording-level CSV with columns relative_path,classes. Required for training.",
    )
    parser.add_argument(
        "--annotation-root",
        type=str,
        default="",
        help="Unused. Box XML is not read; mAP waits until box annotations are available.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="__background__,rain,insect,frog,flow,dog,construction,chicken,birds",
        help="Comma-separated class names, e.g. __background__,rain,insect,...",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Unused. Training uses every image; there is no validation split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used for train/val split.",
    )
    parser.add_argument("--config-file", type=str, default="", help="Optional cfg yaml.")
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Optional KEY VALUE pairs to override cfg. Examples: "
        "energy gate: MODEL.ROI_WEAK_HEAD.INFERENCE_ENERGY_GATE.ENABLED True; "
        "ROI weak head: MODEL.ROI_WEAK_HEAD.OICR_P 0.2 MODEL.ROI_WEAK_HEAD.REGRESS_ON True "
        "MODEL.ROI_WEAK_HEAD.REGRESS_HEUR AVG",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints_wscl_260515", help="Checkpoint output directory.")
    parser.add_argument(
        "--pseudo-box-file",
        type=str,
        default="",
        help="Pickle of offline pseudo boxes (target.external_pseudo). Use with "
        "MODEL.ROI_WEAK_HEAD.USE_EXTERNAL_PSEUDO True.",
    )
    parser.add_argument(
        "--proposal-file",
        type=str,
        default="",
        help="Pickle of offline RPN proposals (bypass live RPN when MODEL.FASTER_RCNN False).",
    )
    parser.add_argument(
        "--proposal-top-k",
        type=int,
        default=2000,
        help="Max offline proposals per image after load.",
    )
    parser.add_argument(
        "--debug-cuda-sync",
        action="store_true",
        help="Enable CUDA_LAUNCH_BLOCKING=1 for accurate stack traces.",
    )
    parser.add_argument(
        "--max-rois-per-image",
        type=int,
        default=48,
        help="Max proposals per image before ROI head (0=disable). Default: env WSCL_MAX_ROIS_PER_IMAGE or 64.",
    )
    parser.add_argument(
        "--max-rois-per-batch",
        type=int,
        default=384,
        help="Hard cap on total RoIs in a batch (split evenly across images; 0=disable). "
        "Strongly recommended for DropBlock / large per-batch RoI count. Default: env WSCL_MAX_ROIS_PER_BATCH or 512.",
    )
    parser.add_argument(
        "--no-imagenet-pretrained",
        action="store_true",
        help="Do not load torchvision ImageNet-1K ResNet-50 weights into the backbone (R-50 only).",
    )
    parser.add_argument(
        "--coco-rpn-weight",
        type=str,
        default="",
        help=(
            "Optional COCO pretrain checkpoint to warm-start backbone+RPN only. "
            "catalog://..., https URL, or local path. Detectron2 zoo "
            "`faster_rcnn_R_50_C4_1x` pickles are supported (numpy tensors); "
            "run scripts/download_r50_c4_pretrained.py (curl + proxy). "
            "RPN cls/bbox may skip if anchor counts differ from this codebase."
        ),
    )
    parser.add_argument(
        "--coco-pretrained-base-url",
        type=str,
        default="",
        help="Replace the official Detectron2 CDN prefix (%s) when resolving catalog:// "
        "weights (mirror must keep the same path after /detectron2). "
        "Override with env WSCL_COCO_PRETRAINED_BASE_URL." % _DETECTRON2_FB_PREFIX,
    )
    parser.add_argument(
        "--coco-pretrained-dir",
        type=str,
        default=_DEFAULT_COCO_PRETRAINED_DIR,
        help="Local folder to store/load COCO backbone+RPN weights before hitting the network. "
        "Override with env WSCL_COCO_PRETRAINED_DIR.",
    )
    parser.add_argument(
        "--download-coco-only",
        action="store_true",
        help="Only resolve/download --coco-rpn-weight into --coco-pretrained-dir then exit.",
    )
    parser.add_argument(
        "--download-proxy",
        type=str,
        default="",
        help="HTTP(S) proxy for downloading pretrained weights (torch.hub / urllib). "
        "Empty string disables the proxy.",
    )
    parser.add_argument(
        "--allow-box-supervision",
        action="store_true",
        help="Train with box or click coordinates (point supervision, click pseudo boxes, "
        "or external pseudo boxes). The default uses recording-level tags only.",
    )
    parser.add_argument(
        "--rpn-click-pseudo-box",
        action="store_true",
        help="RPN training only, and only with --allow-box-supervision: fixed-size pseudo boxes "
        "on clicks plus standard IoU RPN loss. Ignored under the recording-level default.",
    )
    parser.add_argument(
        "--rpn-pseudo-box-w",
        type=float,
        default=61.0,
        help="Pseudo box width in original-image pixels. Used with --allow-box-supervision and --rpn-click-pseudo-box.",
    )
    parser.add_argument(
        "--rpn-pseudo-box-h",
        type=float,
        default=36.0,
        help="Pseudo box height in original-image pixels. Used with --allow-box-supervision and --rpn-click-pseudo-box.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load --ckpt and run validation mAP only. Unavailable until box annotations exist.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Checkpoint path for --eval-only (model + model_cdb state_dict).",
    )
    parser.add_argument(
        "--resume-ckpt",
        type=str,
        default="",
        help="Warm-start training from a saved WSCL checkpoint (strict=False; for REGRESS_ON finetune).",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Finetune mode: skip COCO warm-start if --resume-ckpt set; use differential LR in optimizer.",
    )
    parser.add_argument(
        "--skip-coco-pretrained",
        action="store_true",
        help="Do not load --coco-rpn-weight (e.g. when resuming a fully trained checkpoint).",
    )
    parser.add_argument(
        "--gate-ablation",
        action="store_true",
        help="With --eval-only: run mAP twice with INFERENCE_ENERGY_GATE OFF then ON.",
    )
    parser.add_argument(
        "--eval-rpn-upper-bound",
        action="store_true",
        help="With --eval-only: also report diagnostic RPN-only mAP using val image-level GT class tags.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: cfg.DATALOADER.NUM_WORKERS). On Windows try 4 if startup is slow.",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Pin memory for faster CPU->GPU copy when using CUDA (default: on).",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pin_memory.",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=True,
        help="Keep DataLoader workers alive across epochs (default: on when num_workers>0).",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent_workers.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker when num_workers>0 (default: 2).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print training loss every N batches; always prints the last batch of each epoch.",
    )
    parser.add_argument(
        "--no-cudnn-benchmark",
        action="store_true",
        help="Disable torch.backends.cudnn.benchmark (slower but more reproducible conv picks).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Run validation mAP every K epochs (default: 1). Final epoch always evaluates.",
    )
    parser.add_argument(
        "--eval-subset",
        type=int,
        default=0,
        help="If >0, use only the first N val images for mid-training eval (faster). "
        "The last epoch always uses the full val set.",
    )
    return parser.parse_args()


def collate_fn(batch):
    return list(zip(*batch))


def format_loss_dict(loss_dict: Dict[str, torch.Tensor], max_items: int = 12) -> str:
    """Compact loss breakdown for logging (scalar tensors only)."""
    parts = []
    for k in sorted(loss_dict.keys()):
        v = loss_dict[k]
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            parts.append(f"{k}={float(v.item()):.4f}")
        if len(parts) >= max_items:
            break
    return " ".join(parts)


def print_train_weak_head_cfg(cfg) -> None:
    rw = cfg.MODEL.ROI_WEAK_HEAD
    print(
        f"Train weak-head: REGRESS_ON={rw.REGRESS_ON} LOSS={rw.LOSS} PREDICTOR={rw.PREDICTOR} "
        f"OICR_P={rw.OICR_P} REGRESS_HEUR={rw.REGRESS_HEUR} PARTIAL_LABELS={rw.PARTIAL_LABELS} "
        f"ROI_LOSS_REFINE={rw.ROI_LOSS_REFINE} SOLVER.CONTRA={cfg.SOLVER.CONTRA} "
        f"DB.METHOD={cfg.DB.METHOD} lmda={cfg.lmda} loss={cfg.loss}"
    )


def build_train_dataloader_kwargs(args, cfg, device: torch.device) -> dict:
    """Shared DataLoader kwargs for train/eval loaders (collate_fn added separately)."""
    num_workers = (
        args.num_workers if args.num_workers is not None else cfg.DATALOADER.NUM_WORKERS
    )
    pin_memory = bool(getattr(args, "pin_memory", True)) and device.type == "cuda"
    kwargs = {"num_workers": num_workers}
    if pin_memory:
        kwargs["pin_memory"] = True
    if num_workers > 0:
        use_persistent = bool(getattr(args, "persistent_workers", True))
        # PyTorch 1.7.x on Windows: pin_memory thread dies on 2nd+ epoch if workers persist.
        if use_persistent and pin_memory and sys.platform == "win32":
            use_persistent = False
            print(
                "[DataLoader] Windows: disabled persistent_workers (incompatible with pin_memory "
                "on this PyTorch build; use --no-pin-memory to keep persistent workers)."
            )
        if use_persistent:
            kwargs["persistent_workers"] = True
        prefetch = getattr(args, "prefetch_factor", 2)
        if prefetch > 0:
            kwargs["prefetch_factor"] = prefetch
    return kwargs

def limit_rois_per_image(rois, max_rois_per_image: int):
    if max_rois_per_image <= 0:
        return rois
    clipped = []
    for r in rois:
        if r is None or len(r) <= max_rois_per_image:
            clipped.append(r)
        else:
            clipped.append(r[:max_rois_per_image])
    return clipped


def limit_rois_per_batch(rois: List, max_total_rois: int) -> List:
    """
    If sum(len(r)) exceeds max_total_rois, trim each non-None BoxList to at most
    max(1, max_total_rois // num_images) boxes (same cap for every image in the batch).
    """
    if max_total_rois <= 0:
        return rois
    n_img = len(rois)
    if n_img == 0:
        return rois
    total = sum(len(r) if r is not None else 0 for r in rois)
    if total <= max_total_rois:
        return rois
    per = max(1, max_total_rois // n_img)
    out = []
    for r in rois:
        if r is None:
            out.append(None)
        elif len(r) <= per:
            out.append(r)
        else:
            out.append(r[:per])
    return out


def compute_iou_tensor(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Geometric IoU (alias for energy_iou.geom_iou_tensor)."""
    return pairwise_iou_tensor(box, boxes, energy_2d=None, iou_mode="geom")


def compute_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
    return float(ap)


def print_eval_inference_cfg(cfg) -> None:
    g = cfg.MODEL.ROI_WEAK_HEAD.INFERENCE_ENERGY_GATE
    rw = cfg.MODEL.ROI_WEAK_HEAD
    print(
        "Eval cfg: live_RPN proposals (eval_rois=None when batch has no offline proposals) | "
        f"INFERENCE_ENERGY_GATE ENABLED={bool(g.ENABLED)} TOPK={g.TOPK_PER_IMAGE} "
        f"MAX={g.MAX_PROPOSALS_PER_IMAGE} MIN_MEAN_ENERGY={g.MIN_MEAN_ENERGY} | "
        f"ROI OICR_P={rw.OICR_P} REGRESS_ON={rw.REGRESS_ON} REGRESS_HEUR={rw.REGRESS_HEUR} "
        f"PARTIAL_LABELS={rw.PARTIAL_LABELS}"
    )


def _map_from_pred_gt_collections(
    preds_by_class: Dict[int, List[Tuple[int, float, torch.Tensor]]],
    gts_by_class: Dict[int, Dict[int, List[torch.Tensor]]],
    num_classes: int,
    iou_mode: str = "geom",
    energy_by_image_id: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, float]:
    if iou_mode not in IOU_MODES:
        raise ValueError(f"iou_mode must be one of {IOU_MODES}, got {iou_mode!r}")
    if iou_mode != "geom" and energy_by_image_id is None:
        raise ValueError(f"energy_by_image_id required for iou_mode={iou_mode!r}")

    def map_at_iou(iou_thresh: float) -> float:
        aps = []
        for c in range(1, num_classes):
            total_gt = sum(len(v) for v in gts_by_class[c].values())
            if total_gt == 0:
                continue

            used = {img_id: torch.zeros(len(gt_list), dtype=torch.bool) for img_id, gt_list in gts_by_class[c].items()}
            detections = sorted(preds_by_class[c], key=lambda x: x[1], reverse=True)
            if len(detections) == 0:
                aps.append(0.0)
                continue

            tp = torch.zeros(len(detections), dtype=torch.float32)
            fp = torch.zeros(len(detections), dtype=torch.float32)

            for i, (img_id, _score, p_box) in enumerate(detections):
                gt_list = gts_by_class[c].get(img_id, [])
                if len(gt_list) == 0:
                    fp[i] = 1.0
                    continue
                gt_tensor = torch.stack(gt_list, dim=0)
                e2d = energy_by_image_id.get(img_id) if energy_by_image_id else None
                ious = pairwise_iou_tensor(p_box, gt_tensor, e2d, iou_mode=iou_mode)
                max_iou, max_idx = torch.max(ious, dim=0)
                if max_iou.item() >= iou_thresh and not used[img_id][max_idx]:
                    tp[i] = 1.0
                    used[img_id][max_idx] = True
                else:
                    fp[i] = 1.0

            cum_tp = torch.cumsum(tp, dim=0)
            cum_fp = torch.cumsum(fp, dim=0)
            recalls = cum_tp / max(float(total_gt), 1e-8)
            precisions = cum_tp / (cum_tp + cum_fp).clamp(min=1e-8)
            aps.append(compute_ap(recalls, precisions))

        return float(sum(aps) / len(aps)) if aps else 0.0

    return {"map50": map_at_iou(0.5), "map75": map_at_iou(0.75)}


def per_class_ap_from_collections(
    preds_by_class: Dict[int, List[Tuple[int, float, torch.Tensor]]],
    gts_by_class: Dict[int, Dict[int, List[torch.Tensor]]],
    num_classes: int,
    iou_thresh: float = 0.5,
    iou_mode: str = "geom",
    energy_by_image_id: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[int, float]:
    """Per-class AP at one IoU threshold (for diagnosis)."""
    out: Dict[int, float] = {}

    def ap_for_class(c: int) -> float:
        total_gt = sum(len(v) for v in gts_by_class[c].values())
        if total_gt == 0:
            return float("nan")
        used = {
            img_id: torch.zeros(len(gt_list), dtype=torch.bool)
            for img_id, gt_list in gts_by_class[c].items()
        }
        detections = sorted(preds_by_class[c], key=lambda x: x[1], reverse=True)
        if len(detections) == 0:
            return 0.0
        tp = torch.zeros(len(detections), dtype=torch.float32)
        fp = torch.zeros(len(detections), dtype=torch.float32)
        for i, (img_id, _score, p_box) in enumerate(detections):
            gt_list = gts_by_class[c].get(img_id, [])
            if len(gt_list) == 0:
                fp[i] = 1.0
                continue
            gt_tensor = torch.stack(gt_list, dim=0)
            e2d = energy_by_image_id.get(img_id) if energy_by_image_id else None
            ious = pairwise_iou_tensor(p_box, gt_tensor, e2d, iou_mode=iou_mode)
            max_iou, max_idx = torch.max(ious, dim=0)
            if max_iou.item() >= iou_thresh and not used[img_id][max_idx]:
                tp[i] = 1.0
                used[img_id][max_idx] = True
            else:
                fp[i] = 1.0
        cum_tp = torch.cumsum(tp, dim=0)
        cum_fp = torch.cumsum(fp, dim=0)
        recalls = cum_tp / max(float(total_gt), 1e-8)
        precisions = cum_tp / (cum_tp + cum_fp).clamp(min=1e-8)
        return compute_ap(recalls, precisions)

    for c in range(1, num_classes):
        out[c] = ap_for_class(c)
    return out


def _accumulate_gt_boxes(
    gts_by_class: Dict[int, Dict[int, List[torch.Tensor]]],
    tgt,
    sample_id: int,
    num_classes: int,
) -> None:
    gt_boxes = tgt.bbox
    gt_labels = tgt.get_field("labels")
    for c in range(1, num_classes):
        gt_mask = gt_labels == c
        cls_gt = gt_boxes[gt_mask]
        if sample_id not in gts_by_class[c]:
            gts_by_class[c][sample_id] = []
        for box in cls_gt:
            gts_by_class[c][sample_id].append(box)


def _accumulate_preds_from_detection(
    preds_by_class: Dict[int, List[Tuple[int, float, torch.Tensor]]],
    pred,
    sample_id: int,
    num_classes: int,
) -> None:
    """Collect per-class detections for mAP (CPU tensors; same semantics as per-det loop)."""
    pred_boxes = pred.bbox.cpu()
    pred_labels = pred.get_field("labels").cpu()
    pred_scores = pred.get_field("scores").cpu()
    for c in range(1, num_classes):
        mask = pred_labels == c
        if not mask.any():
            continue
        boxes_c = pred_boxes[mask]
        scores_c = pred_scores[mask]
        preds_by_class[c].extend(
            (sample_id, float(s.item()), boxes_c[i]) for i, s in enumerate(scores_c)
        )


def collect_eval_preds_gts(
    model: torch.nn.Module,
    model_cdb: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    size_divisible: int,
    cfg=None,
    store_energy_maps: bool = False,
) -> Tuple[
    Dict[int, List[Tuple[int, float, torch.Tensor]]],
    Dict[int, Dict[int, List[torch.Tensor]]],
    Dict[int, torch.Tensor],
]:
    """One eval forward; optionally cache per-image energy maps for energy IoU modes."""
    model.eval()
    model_cdb.eval()

    preds_by_class: Dict[int, List[Tuple[int, float, torch.Tensor]]] = {
        c: [] for c in range(1, num_classes)
    }
    gts_by_class: Dict[int, Dict[int, List[torch.Tensor]]] = {c: {} for c in range(1, num_classes)}
    energy_by_image_id: Dict[int, torch.Tensor] = {}

    pin_memory = getattr(data_loader, "pin_memory", False)
    non_blocking = pin_memory and device.type == "cuda"

    with torch.no_grad():
        for images, targets, rois, indices in data_loader:
            image_tensors = [img.to(device, non_blocking=non_blocking) for img in images]
            image_list = to_image_list(image_tensors, size_divisible=size_divisible)
            if store_energy_maps and cfg is not None:
                emaps = energy_map_from_image_tensors(image_list.tensors, cfg)
            outputs = model(
                image_list,
                targets=None,
                rois=eval_rois_argument(rois),
                model_cdb=model_cdb,
            )

            for bi, (pred, tgt, sample_id) in enumerate(zip(outputs, targets, indices)):
                tgt = tgt.to("cpu")
                sample_id = int(sample_id)
                if store_energy_maps and cfg is not None:
                    energy_by_image_id[sample_id] = emaps[bi].detach().cpu()
                _accumulate_gt_boxes(gts_by_class, tgt, sample_id, num_classes)
                _accumulate_preds_from_detection(
                    preds_by_class, pred, sample_id, num_classes
                )

    return preds_by_class, gts_by_class, energy_by_image_id


def evaluate_map(
    model: torch.nn.Module,
    model_cdb: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    size_divisible: int,
    iou_mode: str = "geom",
    cfg=None,
    preds_gts_energy: Optional[Tuple] = None,
) -> Dict[str, float]:
    need_energy = iou_mode != "geom"
    if preds_gts_energy is not None:
        preds_by_class, gts_by_class, energy_by_image_id = preds_gts_energy
    else:
        preds_by_class, gts_by_class, energy_by_image_id = collect_eval_preds_gts(
            model,
            model_cdb,
            data_loader,
            device,
            num_classes,
            size_divisible,
            cfg=cfg,
            store_energy_maps=need_energy,
        )
    e_map = energy_by_image_id if need_energy else None
    return _map_from_pred_gt_collections(
        preds_by_class, gts_by_class, num_classes, iou_mode=iou_mode, energy_by_image_id=e_map
    )


def evaluate_map_all_modes(
    model: torch.nn.Module,
    model_cdb: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    size_divisible: int,
    cfg,
    modes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Single forward; mAP@50/75 for geom + energy IoU modes."""
    if modes is None:
        modes = list(IOU_MODES)
    bundle = collect_eval_preds_gts(
        model,
        model_cdb,
        data_loader,
        device,
        num_classes,
        size_divisible,
        cfg=cfg,
        store_energy_maps=True,
    )
    out = {}
    for mode in modes:
        out[mode] = evaluate_map(
            model,
            model_cdb,
            data_loader,
            device,
            num_classes,
            size_divisible,
            iou_mode=mode,
            cfg=cfg,
            preds_gts_energy=bundle,
        )
    return out


def evaluate_map_rpn_upper_bound(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    size_divisible: int,
    cfg,
) -> Dict[str, float]:
    """
    Diagnostic upper bound: RPN boxes + objectness, labels = val image-level GT class tags.
    Not a deployable test protocol (uses per-image GT class presence).
    """
    model.eval()

    preds_by_class: Dict[int, List[Tuple[int, float, torch.Tensor]]] = {c: [] for c in range(1, num_classes)}
    gts_by_class: Dict[int, Dict[int, List[torch.Tensor]]] = {c: {} for c in range(1, num_classes)}

    gate_on = bool(cfg.MODEL.ROI_WEAK_HEAD.INFERENCE_ENERGY_GATE.ENABLED)

    with torch.no_grad():
        for images, targets, _rois, indices in data_loader:
            image_tensors = [img.to(device) for img in images]
            image_list = to_image_list(image_tensors, size_divisible=size_divisible)
            features = model.backbone(image_list.tensors)

            use_attn = (
                hasattr(cfg.MODEL.RPN, "ATTN_PRIOR")
                and bool(cfg.MODEL.RPN.ATTN_PRIOR.ENABLED)
            )
            atten_logits = None
            if use_attn:
                out_hw = features[0].shape[-2:]
                atten_logits = model._make_energy_atten_logits(image_list, out_hw)
            proposals, _ = model.rpn(image_list, features, targets=None, atten_logits=atten_logits)

            if gate_on:
                proposals = filter_proposals_by_energy_inference(
                    proposals, image_list.tensors, cfg
                )

            for prop, tgt, sample_id in zip(proposals, targets, indices):
                prop = prop.to("cpu")
                tgt = tgt.to("cpu")
                sample_id = int(sample_id)
                _accumulate_gt_boxes(gts_by_class, tgt, sample_id, num_classes)

                if len(prop) == 0 or not prop.has_field("objectness"):
                    continue
                obj_scores = prop.get_field("objectness")
                boxes = prop.bbox
                img_classes = tgt.get_field("labels").unique()
                img_classes = img_classes[img_classes > 0]
                topk = min(int(cfg.TEST.DETECTIONS_PER_IMG), len(prop))
                if topk <= 0:
                    continue
                _, inds = torch.topk(obj_scores, topk, dim=0, largest=True, sorted=False)
                for c in img_classes:
                    c_int = int(c.item())
                    for i in inds.tolist():
                        preds_by_class[c_int].append(
                            (sample_id, float(obj_scores[i].item()), boxes[i])
                        )

    return _map_from_pred_gt_collections(preds_by_class, gts_by_class, num_classes)


def evaluate_map_rpn_oicr_seed_upper_bound(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    size_divisible: int,
    cfg,
) -> Dict[str, float]:
    """One proposal per image-level class (argmax objectness), diagnostic only."""
    model.eval()
    preds_by_class: Dict[int, List[Tuple[int, float, torch.Tensor]]] = {
        c: [] for c in range(1, num_classes)
    }
    gts_by_class: Dict[int, Dict[int, List[torch.Tensor]]] = {c: {} for c in range(1, num_classes)}
    gate_on = bool(cfg.MODEL.ROI_WEAK_HEAD.INFERENCE_ENERGY_GATE.ENABLED)

    with torch.no_grad():
        for images, targets, _rois, indices in data_loader:
            image_tensors = [img.to(device) for img in images]
            image_list = to_image_list(image_tensors, size_divisible=size_divisible)
            features = model.backbone(image_list.tensors)
            use_attn = (
                hasattr(cfg.MODEL.RPN, "ATTN_PRIOR")
                and bool(cfg.MODEL.RPN.ATTN_PRIOR.ENABLED)
            )
            atten_logits = None
            if use_attn:
                out_hw = features[0].shape[-2:]
                atten_logits = model._make_energy_atten_logits(image_list, out_hw)
            proposals, _ = model.rpn(image_list, features, targets=None, atten_logits=atten_logits)
            if gate_on:
                proposals = filter_proposals_by_energy_inference(
                    proposals, image_list.tensors, cfg
                )
            for prop, tgt, sample_id in zip(proposals, targets, indices):
                prop = prop.to("cpu")
                tgt = tgt.to("cpu")
                sample_id = int(sample_id)
                _accumulate_gt_boxes(gts_by_class, tgt, sample_id, num_classes)
                if len(prop) == 0 or not prop.has_field("objectness"):
                    continue
                obj_scores = prop.get_field("objectness")
                boxes = prop.bbox
                img_classes = tgt.get_field("labels").unique()
                img_classes = img_classes[img_classes > 0]
                best_i = int(torch.argmax(obj_scores).item())
                for c in img_classes:
                    c_int = int(c.item())
                    preds_by_class[c_int].append(
                        (sample_id, float(obj_scores[best_i].item()), boxes[best_i])
                    )
    return _map_from_pred_gt_collections(preds_by_class, gts_by_class, num_classes)


def load_checkpoint_for_eval(
    ckpt_path: str,
    model: torch.nn.Module,
    model_cdb: torch.nn.Module,
    device: torch.device,
    strict: bool = True,
) -> dict:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=strict)
    if not strict and (missing or unexpected):
        print(
            f"Resume load (strict=False): missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )
        if missing:
            print(f"  missing (first 8): {missing[:8]}")
        if unexpected:
            print(f"  unexpected (first 8): {unexpected[:8]}")
    model_cdb.load_state_dict(ck["model_cdb_state_dict"], strict=strict)
    return ck


def _set_model_energy_gate(model: torch.nn.Module, enabled: bool) -> None:
    """Forward reads model.cfg; ablation must patch the detector copy, not global cfg."""
    model.cfg.defrost()
    model.cfg.MODEL.ROI_WEAK_HEAD.INFERENCE_ENERGY_GATE.ENABLED = enabled
    model.cfg.freeze()


def run_eval_only(
    args,
    cfg,
    model: torch.nn.Module,
    model_cdb: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
) -> None:
    if not args.ckpt:
        raise ValueError("--eval-only requires --ckpt")

    ck = load_checkpoint_for_eval(args.ckpt, model, model_cdb, device)
    print(f"Loaded checkpoint: {args.ckpt}")
    if "map50" in ck:
        print(f"  recorded epoch={ck.get('epoch')} map50={ck.get('map50')} map75={ck.get('map75')}")

    num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
    size_div = cfg.DATALOADER.SIZE_DIVISIBILITY

    def _run_full(label: str, gate_enabled: bool) -> Dict[str, float]:
        _set_model_energy_gate(model, gate_enabled)
        print(f"\n--- Full pipeline ({label}) ---")
        print_eval_inference_cfg(model.cfg)
        metrics = evaluate_map(model, model_cdb, eval_loader, device, num_classes, size_div)
        print(f"mAP@50={metrics['map50']:.6f} mAP@75={metrics['map75']:.6f}")
        return metrics

    if args.gate_ablation:
        m_off = _run_full("energy_gate OFF", False)
        m_on = _run_full("energy_gate ON", True)
        print(
            f"\nGate ablation delta: mAP@50 {m_on['map50'] - m_off['map50']:+.6f} "
            f"mAP@75 {m_on['map75'] - m_off['map75']:+.6f}"
        )
    else:
        _set_model_energy_gate(
            model, bool(cfg.MODEL.ROI_WEAK_HEAD.INFERENCE_ENERGY_GATE.ENABLED)
        )
        _run_full("current cfg", bool(model.cfg.MODEL.ROI_WEAK_HEAD.INFERENCE_ENERGY_GATE.ENABLED))

    if args.eval_rpn_upper_bound:
        eval_cfg = model.cfg
        print("\n--- RPN upper bound (top-K objectness per image-level class) ---")
        print_eval_inference_cfg(eval_cfg)
        rb = evaluate_map_rpn_upper_bound(
            model, eval_loader, device, num_classes, size_div, eval_cfg
        )
        print(
            f"RPN-topK-per-class mAP@50={rb['map50']:.6f} mAP@75={rb['map75']:.6f}"
        )
        rb_seed = evaluate_map_rpn_oicr_seed_upper_bound(
            model, eval_loader, device, num_classes, size_div, eval_cfg
        )
        print(
            f"RPN-OICR-seed (1 prop/class) mAP@50={rb_seed['map50']:.6f} mAP@75={rb_seed['map75']:.6f} "
            "(diagnostic; uses val image-level GT class tags)"
        )


def save_last_checkpoint(save_dir: str, epoch: int, model: torch.nn.Module, model_cdb: torch.nn.Module) -> str:
    ckpt_path = os.path.join(save_dir, "last.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_cdb_state_dict": model_cdb.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def save_top3_checkpoint(
    save_dir: str,
    epoch: int,
    map50: float,
    map75: float,
    model: torch.nn.Module,
    model_cdb: torch.nn.Module,
    top3: List[Tuple[float, str]],
):
    ckpt_name = f"epoch_{epoch:03d}_map50_{map50:.4f}_map75_{map75:.4f}.pth"
    ckpt_path = os.path.join(save_dir, ckpt_name)

    torch.save(
        {
            "epoch": epoch,
            "map50": map50,
            "map75": map75,
            "model_state_dict": model.state_dict(),
            "model_cdb_state_dict": model_cdb.state_dict(),
        },
        ckpt_path,
    )
    top3.append((map50, ckpt_path))
    top3.sort(key=lambda x: x[0], reverse=True)

    while len(top3) > 3:
        _score, to_remove = top3.pop(-1)
        if os.path.exists(to_remove):
            os.remove(to_remove)


def load_coco_backbone_rpn_only(cfg, model: torch.nn.Module, weight_path: str) -> int:
    """
    Load only backbone/rpn parameters from a COCO checkpoint with suffix matching
    and shape checks, to avoid class-head mismatch failures.
    """
    if not weight_path:
        return 0
    ckpt_loader = DetectronCheckpointer(
        cfg, model, optimizer=None, scheduler=None, save_dir="", save_to_disk=False
    )
    loaded = ckpt_loader._load_file(weight_path)
    loaded_state = loaded.get("model", loaded)
    loaded_state = remap_detectron2_r50_c4_state_dict_if_needed(loaded_state)
    if all(k.startswith("module.") for k in loaded_state.keys()):
        loaded_state = {k.replace("module.", "", 1): v for k, v in loaded_state.items()}

    model_state = model.state_dict()
    loaded_keys = list(loaded_state.keys())
    updated = 0
    for mk in list(model_state.keys()):
        if not (mk.startswith("backbone.") or mk.startswith("rpn.")):
            continue
        # Longest suffix match to mimic detectron loader behavior.
        candidates = [lk for lk in loaded_keys if mk.endswith(lk)]
        if not candidates:
            continue
        best_lk = max(candidates, key=len)
        src = loaded_state[best_lk]
        if not isinstance(src, torch.Tensor):
            src = torch.as_tensor(src)
        if model_state[mk].shape != src.shape:
            continue
        model_state[mk] = src
        updated += 1

    model.load_state_dict(model_state, strict=False)
    return updated


def resolve_pretrained_download_url(cfg, ref: str) -> str:
    """Turn catalog://... into an https URL; pass through http(s) URLs."""
    ref = ref.strip()
    if ref.startswith("catalog://"):
        paths_catalog = import_file("wetectron.config.paths_catalog", cfg.PATHS_CATALOG, True)
        return paths_catalog.ModelCatalog.get(ref[len("catalog://") :])
    return ref


def apply_detectron2_mirror_prefix(url: str, mirror_base: str) -> str:
    """If mirror_base is set, swap https://dl.fbaipublicfiles.com/detectron2 for that origin."""
    base = (mirror_base or "").strip().rstrip("/")
    if not base:
        return url
    if url.startswith(_DETECTRON2_FB_PREFIX):
        return base + url[len(_DETECTRON2_FB_PREFIX) :]
    return url


def ensure_coco_pretrained_local(cfg, ref: str, cache_dir: str, mirror_base: str = "") -> str:
    """
    If ref is an existing file path, return it.
    Otherwise resolve URL (catalog/http) and return path under cache_dir,
    downloading via cache_url only when missing (urllib respects HTTP_PROXY/HTTPS_PROXY).
    """
    ref = (ref or "").strip()
    if not ref:
        return ""
    if os.path.isfile(ref):
        print(f"COCO pretrained: using local file {ref}")
        return ref

    os.makedirs(cache_dir, exist_ok=True)
    url = resolve_pretrained_download_url(cfg, ref)
    url = apply_detectron2_mirror_prefix(url, mirror_base)
    if mirror_base and url.startswith("http"):
        print(f"COCO pretrained: using mirror base -> {url[:120]}{'...' if len(url) > 120 else ''}")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise FileNotFoundError(
            "COCO weight is not a local file and did not resolve to an http(s) URL: "
            f"ref={ref!r} -> {url!r}"
        )

    # cache_url skips download if target already exists
    out_path = cache_url(url, model_dir=cache_dir, progress=True)
    if os.path.isfile(out_path):
        print(f"COCO pretrained: ready at {out_path}")
    return out_path


def main():
    args = parse_args()
    if args.eval_only:
        print("mAP evaluation needs box annotations, which are not available yet.")
        sys.exit(1)
    if args.debug_cuda_sync:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        print(
            "[Warn] --debug-cuda-sync forces CUDA_LAUNCH_BLOCKING=1; "
            "training and eval will be much slower. Disable for production runs."
        )

    apply_download_proxy(args.download_proxy)

    if not args.download_coco_only:
        if not args.image_root or not args.label_csv:
            print("--image-root and --label-csv are required.")
            sys.exit(1)

    cfg = _C.clone()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.allow_box_supervision:
        cfg.merge_from_list(["MODEL.RPN.SUPERVISE_WITH_BOXES", "True"])
        if args.rpn_click_pseudo_box:
            cfg.merge_from_list(
                [
                    "MODEL.RPN.USE_CLICK_PSEUDO_BOX_FOR_TRAIN",
                    "True",
                    "MODEL.RPN.CLICK_PSEUDO_BOX_W",
                    str(args.rpn_pseudo_box_w),
                    "MODEL.RPN.CLICK_PSEUDO_BOX_H",
                    str(args.rpn_pseudo_box_h),
                ]
            )
    else:
        if args.rpn_click_pseudo_box or args.pseudo_box_file.strip():
            print(
                "Recording-level default: ignoring --rpn-click-pseudo-box and "
                "--pseudo-box-file. Pass --allow-box-supervision to use them."
            )
        cfg.merge_from_list(RECORDING_LEVEL_OPTS)
        print(
            "Recording-level supervision: class tags only "
            "(PARTIAL_LABELS=none, RPN box/click loss off, SOLVER.CONTRA=True, OICR_P=0)."
        )
    cfg.freeze()
    if args.allow_box_supervision and args.rpn_click_pseudo_box:
        print(
            "RPN: click→pseudo box + standard IoU loss "
            f"(W={cfg.MODEL.RPN.CLICK_PSEUDO_BOX_W}, H={cfg.MODEL.RPN.CLICK_PSEUDO_BOX_H}); "
            "ROI weak-head supervision unchanged (PARTIAL_LABELS=point)."
        )
    if not cfg.MODEL.ROI_WEAK_HEAD.REGRESS_ON:
        print(
            "ROI: REGRESS_ON=False -> ROIWeakHead (proposal boxes kept at inference), "
            "loss=RoILoss, predictor=OICRPredictor (auto if cfg had RoIRegLoss/MISTPredictor)."
        )

    coco_pretrained_dir = os.environ.get("WSCL_COCO_PRETRAINED_DIR", args.coco_pretrained_dir)
    coco_mirror_base = os.environ.get("WSCL_COCO_PRETRAINED_BASE_URL", args.coco_pretrained_base_url)

    if args.download_coco_only:
        if not args.coco_rpn_weight:
            print("--download-coco-only requires --coco-rpn-weight")
            sys.exit(1)
        path = ensure_coco_pretrained_local(
            cfg, args.coco_rpn_weight, coco_pretrained_dir, mirror_base=coco_mirror_base
        )
        print(f"Done. Local weight path: {path}")
        sys.exit(0)

    device = torch.device(cfg.MODEL.DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not args.no_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.save_dir, exist_ok=True)
    dl_common = build_train_dataloader_kwargs(args, cfg, device)
    non_blocking = bool(dl_common.get("pin_memory", False))

    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    pseudo_box_file = args.pseudo_box_file.strip() or None
    if not args.allow_box_supervision:
        pseudo_box_file = None
    proposal_file = args.proposal_file.strip() or None
    train_ds = build_spec_dataset(
        image_root=args.image_root,
        class_names=class_names,
        cfg=cfg,
        is_train=True,
        pseudo_box_file=pseudo_box_file,
        proposal_file=proposal_file,
        proposal_top_k=args.proposal_top_k,
        label_csv=args.label_csv,
    )
    print(
        "Training on all %d images from %s. Per-epoch mAP is skipped until box annotations exist."
        % (len(train_ds), args.label_csv)
    )

    model = build_detection_model(cfg).to(device)

    data_loader = DataLoader(
        train_ds,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        **dl_common,
    )

    print(
        f"DataLoader: num_workers={dl_common.get('num_workers')} "
        f"pin_memory={dl_common.get('pin_memory', False)} "
        f"persistent_workers={dl_common.get('persistent_workers', False)} "
        f"prefetch_factor={dl_common.get('prefetch_factor', 'n/a')} "
        f"log_interval={args.log_interval}"
    )

    if args.finetune or args.resume_ckpt:
        cfg.defrost()
        steps_total = args.epochs * max(len(data_loader), 1)
        mid_step = max(1, steps_total // 2)
        cfg.SOLVER.STEPS = (mid_step,)
        cfg.SOLVER.WARMUP_ITERS = min(int(cfg.SOLVER.WARMUP_ITERS), max(1, len(data_loader) // 4))
        cfg.freeze()
        print(
            f"[Finetune] LR schedule: STEPS=({mid_step},) GAMMA={cfg.SOLVER.GAMMA} "
            f"WARMUP_ITERS={cfg.SOLVER.WARMUP_ITERS} BASE_LR={cfg.SOLVER.BASE_LR}"
        )

    if not args.no_imagenet_pretrained and "R-50" in str(cfg.MODEL.BACKBONE.CONV_BODY):
        n_loaded, n_left = load_imagenet_resnet50_backbone(model)
        print(
            f"ImageNet pretrained R-50: applied {n_loaded} weight tensors to backbone; "
            f"{n_left} body params not in torchvision dict (left as after build)."
        )
    elif not args.no_imagenet_pretrained:
        print(
            f"Skip ImageNet R-50 preload (backbone is {cfg.MODEL.BACKBONE.CONV_BODY}, not R-50)."
        )

    if cfg.MODEL.WEIGHT:
        _ckpt = DetectronCheckpointer(
            cfg, model, optimizer=None, scheduler=None, save_dir="", save_to_disk=False
        )
        _ckpt.load(cfg.MODEL.WEIGHT)
        print(f"Loaded extra checkpoint weights from MODEL.WEIGHT={cfg.MODEL.WEIGHT!r}")

    skip_coco = bool(args.skip_coco_pretrained or args.resume_ckpt or args.finetune)
    if args.coco_rpn_weight and not skip_coco:
        try:
            local_ckpt = ensure_coco_pretrained_local(
                cfg, args.coco_rpn_weight, coco_pretrained_dir, mirror_base=coco_mirror_base
            )
            n = load_coco_backbone_rpn_only(cfg, model, local_ckpt)
            print(
                f"COCO backbone+RPN warm-start: loaded {n} tensors from {local_ckpt} "
                f"(ref {args.coco_rpn_weight!r})"
            )
        except Exception as e:
            print(
                f"[Warn] Failed to load COCO backbone+RPN warm-start from {args.coco_rpn_weight!r}: {e}"
            )
            print("[Warn] Continue training with current initialized weights.")
    elif args.coco_rpn_weight and skip_coco:
        print(
            "[Train] Skipped COCO warm-start "
            "(resume_ckpt/finetune/skip-coco-pretrained)."
        )

    model_cdb = ConvConcreteDB(cfg, model.backbone.out_channels).to(device)

    if args.resume_ckpt:
        ck_resume = load_checkpoint_for_eval(
            args.resume_ckpt, model, model_cdb, device, strict=False
        )
        print(
            f"Resumed training weights from {args.resume_ckpt} "
            f"(epoch={ck_resume.get('epoch')} map50={ck_resume.get('map50')})"
        )

    print_train_weak_head_cfg(cfg)
    optimizer = make_optimizer(cfg, model, finetune=bool(args.finetune or args.resume_ckpt))
    scheduler = make_lr_scheduler(cfg, optimizer)
    optimizer_cdb = make_cdb_optimizer(cfg, model_cdb)
    scheduler_cdb = make_lr_cdb_scheduler(cfg, optimizer_cdb)
    use_cdb_step = str(cfg.DB.METHOD).lower() == "concrete"

    # Numeric-stability guards (override by environment variables when needed).
    max_abs_main_loss = float(os.environ.get("WSCL_MAX_ABS_MAIN_LOSS", "1e6"))
    max_abs_cdb_loss = float(os.environ.get("WSCL_MAX_ABS_CDB_LOSS", "1e4"))
    max_grad_norm_main = float(os.environ.get("WSCL_MAX_GRAD_NORM_MAIN", "50.0"))
    max_grad_norm_cdb = float(os.environ.get("WSCL_MAX_GRAD_NORM_CDB", "10.0"))
    max_rois_per_image = (
        args.max_rois_per_image
        if args.max_rois_per_image is not None
        else int(os.environ.get("WSCL_MAX_ROIS_PER_IMAGE", "64"))
    )
    max_rois_per_batch = (
        args.max_rois_per_batch
        if args.max_rois_per_batch is not None
        else int(os.environ.get("WSCL_MAX_ROIS_PER_BATCH", "512"))
    )
    print(
        f"RoI caps (reduce if CUDA OOM): max_per_image={max_rois_per_image} "
        f"max_per_batch={max_rois_per_batch} ims_per_batch={cfg.SOLVER.IMS_PER_BATCH}"
    )

    _logged_train_cfg = False

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training epochs"):
        model.train()
        model_cdb.train()

        for batch_idx, (images, targets, rois, _indices) in enumerate(data_loader, start=1):
            image_tensors = [img.to(device, non_blocking=non_blocking) for img in images]
            targets = [t.to(device) for t in targets]
            rois = [r.to(device) if r is not None else None for r in rois]
            rois = limit_rois_per_image(rois, max_rois_per_image)
            rois = limit_rois_per_batch(rois, max_rois_per_batch)
            image_list = to_image_list(image_tensors, size_divisible=cfg.DATALOADER.SIZE_DIVISIBILITY)

            # Step-1: update main model (minimize task loss)
            optimizer.zero_grad(set_to_none=True)
            optimizer_cdb.zero_grad(set_to_none=True)
            loss_dict, _metrics = model(image_list, targets, rois, model_cdb=model_cdb, iteration={"iter": batch_idx})
            if not all(torch.isfinite(v).all() for v in loss_dict.values()):
                print("[MainLossTrace] non-finite loss_dict at batch {}".format(batch_idx))
                for k, v in loss_dict.items():
                    finite_mask = torch.isfinite(v)
                    finite_ratio = finite_mask.float().mean().item()
                    if finite_mask.any():
                        finite_vals = v[finite_mask]
                        vmin = finite_vals.min().item()
                        vmax = finite_vals.max().item()
                    else:
                        vmin = float("nan")
                        vmax = float("nan")
                    print(
                        "[MainLossTrace] {} shape={} finite_ratio={:.6f} finite_min={:.6e} finite_max={:.6e}".format(
                            k, tuple(v.shape), finite_ratio, vmin, vmax
                        )
                    )
                raise RuntimeError("non-finite component found in main loss_dict")
            loss = sum(v for v in loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite main loss at batch {batch_idx}: {loss}")
            if abs(loss.item()) > max_abs_main_loss:
                raise RuntimeError(
                    f"main loss exploded at batch {batch_idx}: {loss.item():.6e} > {max_abs_main_loss:.6e}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm_main)
            optimizer.step()
            scheduler.step()

            # Step-2: update CDB model (maximize task loss with a negative sign)
            loss_cdb = torch.tensor(0.0, device=device)
            if use_cdb_step:
                optimizer.zero_grad(set_to_none=True)  # keep model grads clean
                optimizer_cdb.zero_grad(set_to_none=True)
                set_requires_grad(model, False)
                try:
                    loss_dict_cdb, _metrics_cdb = model(
                        image_list, targets, rois, model_cdb=model_cdb, iteration={"iter": batch_idx}
                    )
                    loss_cdb = -float(cfg.DB.WEIGHT) * sum(v for v in loss_dict_cdb.values())
                    if not torch.isfinite(loss_cdb):
                        print(
                            f"[CDBGuard] skip CDB step at batch {batch_idx}: non-finite loss_cdb={loss_cdb}"
                        )
                        optimizer_cdb.zero_grad(set_to_none=True)
                    elif abs(loss_cdb.item()) > max_abs_cdb_loss:
                        print(
                            f"[CDBGuard] skip CDB step at batch {batch_idx}: "
                            f"|loss_cdb|={abs(loss_cdb.item()):.6e} > {max_abs_cdb_loss:.6e}"
                        )
                        optimizer_cdb.zero_grad(set_to_none=True)
                    elif not loss_cdb.requires_grad:
                        print(
                            f"[CDBGuard] skip CDB step at batch {batch_idx}: "
                            "loss_cdb has no grad path to model_cdb"
                        )
                    else:
                        loss_cdb.backward()
                        torch.nn.utils.clip_grad_norm_(model_cdb.parameters(), max_grad_norm_cdb)
                        optimizer_cdb.step()
                        scheduler_cdb.step()
                finally:
                    set_requires_grad(model, True)

            if batch_idx % args.log_interval == 0 or batch_idx == len(data_loader):
                loss_detail = format_loss_dict(loss_dict)
                print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"Batch {batch_idx}/{len(data_loader)} "
                    f"loss={loss.item():.6f} loss_cdb={loss_cdb.item():.6f} | {loss_detail}"
                )
            if not _logged_train_cfg and batch_idx == 1 and epoch == 1:
                _logged_train_cfg = True
                if "loss_sim" in loss_dict:
                    print(
                        f"[Train] loss_sim present (CONTRA active): "
                        f"{float(loss_dict['loss_sim'].item()):.6f}"
                    )
                else:
                    print("[Train] loss_sim absent (SOLVER.CONTRA=False or not in loss_dict)")
            del loss_dict, _metrics, loss, image_list, image_tensors, targets, rois
            if use_cdb_step and "loss_dict_cdb" in locals() and "_metrics_cdb" in locals():
                del loss_dict_cdb, _metrics_cdb
            if device.type == "cuda" and int(os.environ.get("WSCL_CUDA_EMPTY_CACHE_EACH_BATCH", "0")) == 1:
                torch.cuda.empty_cache()

        ckpt_path = save_last_checkpoint(args.save_dir, epoch, model, model_cdb)
        print(f"[Epoch {epoch}/{args.epochs}] saved {ckpt_path}")

    print("Training finished.")
    print(f"Last checkpoint: {os.path.join(args.save_dir, 'last.pth')}")


if __name__ == "__main__":
    main()
