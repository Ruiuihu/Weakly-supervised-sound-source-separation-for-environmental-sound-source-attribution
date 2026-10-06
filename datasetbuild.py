# -*- coding: utf-8 -*-
"""
Dataset for spec images + Pascal VOC-style XML annotations.
Image root and annotation root have the same subdirectory structure;
each image pairs with an XML file of the same base name in the corresponding subdir.
Compatible with OD-WSCL DataLoader (BoxList, get_groundtruth, get_img_info).
"""

import csv
import os
import pickle

import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET
from wetectron.config.defaults import _C as cfg

# Use project's BoxList if available (for integration with wetectron)
try:
    from wetectron.structures.bounding_box import BoxList
    from wetectron.structures.keypoint import Click
except ImportError:
    BoxList = None
    Click = None

# Common image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class RecordingLabels(object):
    """Image-level class ids that survive box indexing, resize, and flip.

    MIL reads ``target.get_field("labels").unique()``. The tensor length does
    not match the empty box list, so this object is not sliced with boxes.
    """

    def __init__(self, labels):
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(list(labels), dtype=torch.int64)
        self.labels = labels.to(dtype=torch.int64).view(-1)

    def unique(self):
        return torch.unique(self.labels)

    def __getitem__(self, item):
        return self

    def resize(self, size, *args, **kwargs):
        return self

    def transpose(self, method):
        return self

    def crop(self, box):
        return self

    def to(self, device, **kwargs):
        return RecordingLabels(self.labels.to(device, **kwargs))


def load_recording_csv(label_csv, class_to_ind):
    """Load rows of (relative_path, foreground class-id tensor)."""
    rows = []
    with open(label_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "relative_path" not in reader.fieldnames or "classes" not in reader.fieldnames:
            raise ValueError(
                "Label CSV %r must have columns relative_path,classes." % label_csv
            )
        for line_no, row in enumerate(reader, start=2):
            rel = (row.get("relative_path") or "").strip().replace("\\", "/")
            raw_classes = (row.get("classes") or "").strip()
            if not rel or not raw_classes:
                raise ValueError("Empty path or classes at %s:%d" % (label_csv, line_no))
            names = [part.strip().lower() for part in raw_classes.split(";") if part.strip()]
            unknown = [name for name in names if name not in class_to_ind]
            if not names or unknown:
                raise ValueError(
                    "Unknown classes %s at %s:%d" % (unknown, label_csv, line_no)
                )
            class_ids = torch.tensor([class_to_ind[name] for name in names], dtype=torch.int64)
            rows.append((rel, class_ids))
    if not rows:
        raise ValueError("Label CSV %r has no rows." % label_csv)
    return rows


def _collect_pairs(image_root, annotation_root):
    """
    Collect (image_path, annotation_path) pairs.
    Same relative path under both roots, and annotation stem matches image stem.
    """
    pairs = []
    image_root = os.path.normpath(image_root)
    annotation_root = os.path.normpath(annotation_root)
    for dirpath, _, filenames in os.walk(image_root):
        rel_dir = os.path.relpath(dirpath, image_root)
        if rel_dir == ".":
            rel_dir = ""
        ann_dir = os.path.join(annotation_root, rel_dir) if rel_dir else annotation_root
        if not os.path.isdir(ann_dir):
            continue
        for fname in filenames:
            base, ext = os.path.splitext(fname)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            img_path = os.path.join(dirpath, fname)
            ann_path = os.path.join(ann_dir, base + ".xml")
            if os.path.isfile(ann_path):
                pairs.append((img_path, ann_path))
    return pairs


def _parse_voc_xml(xml_path, class_to_ind, keep_difficult=True):
    """
    Parse Pascal VOC-style XML. Returns dict with keys:
    boxes (tensor), labels (tensor), difficult (tensor), im_info (height, width).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    difficult_list = []

    size = root.find("size")
    if size is not None:
        h = int(size.find("height").text)
        w = int(size.find("width").text)
        im_info = (h, w)
    else:
        im_info = None

    for obj in root.iter("object"):
        name_el = obj.find("name")
        if name_el is None:
            continue
        name = name_el.text
        if name is not None:
            name = name.lower().strip()
        if name not in class_to_ind:
            continue
        difficult = 0
        diff_el = obj.find("difficult")
        if diff_el is not None and diff_el.text:
            difficult = int(diff_el.text)
        if not keep_difficult and difficult == 1:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        # 0-based pixel indexes (Pascal VOC convention)
        xmin = int(bb.find("xmin").text) - 1
        ymin = int(bb.find("ymin").text) - 1
        xmax = int(bb.find("xmax").text) - 1
        ymax = int(bb.find("ymax").text) - 1
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_ind[name])
        difficult_list.append(difficult)

    res = {
        "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
        "difficult": torch.tensor(difficult_list, dtype=torch.uint8) if difficult_list else torch.zeros(0, dtype=torch.uint8),
        "im_info": im_info,
    }
    return res


def load_pseudo_box_pickle(pseudo_box_file):
    """Load offline pseudo boxes dict: ids (str stems), boxes, scores, labels."""
    with open(pseudo_box_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    id_to_idx = {str(sid): i for i, sid in enumerate(data["ids"])}
    return data, id_to_idx


def _build_click_from_boxes(boxes):
    """
    Build per-instance click points from box centers.
    Args:
        boxes (Tensor): shape [N, 4] in xyxy format.
    Returns:
        Tensor: shape [N, 2], each row is [center_x, center_y].
    """
    if boxes.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    center_x = torch.floor((boxes[:, 0] + boxes[:, 2]) * 0.5)
    center_y = torch.floor((boxes[:, 1] + boxes[:, 3]) * 0.5)
    return torch.stack([center_x, center_y], dim=1).to(dtype=torch.float32)


def _load_local_wetectron_transforms_module():
    """
    Load transform ops from wetectron/data/transforms/transforms.py via a real,
    importable module (datasetbuild_local_transforms.py) so DataLoader workers
    on Windows (spawn) can unpickle the dataset.
    """
    import datasetbuild_local_transforms as module

    return module


def _build_transforms_from_cfg(cfg, is_train=True):
    """
    Build transforms equivalent to wetectron.data.transforms.build.build_transforms
    without importing wetectron.data package.
    """
    T = _load_local_wetectron_transforms_module()
    imagenet_pca_eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    imagenet_pca_eigvec = torch.Tensor(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
    )

    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = 0.5  # match original build.py
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD,
        to_bgr255=cfg.INPUT.TO_BGR255,
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    if cfg.INPUT.PCA:
        return T.Compose(
            [
                color_jitter,
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomVerticalFlip(flip_vertical_prob),
                T.ToTensor(),
                T.Lighting(0.1, imagenet_pca_eigval, imagenet_pca_eigvec),
                normalize_transform,
            ]
        )
    return T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )


class SpecImgDataset(torch.utils.data.Dataset):
    """
    Map-style Dataset for spec images + XML object detection annotations.
    - image_root: root of image tree (e.g. spec_img), subdirs contain images.
    - annotation_root: root of XML tree (e.g. mixed_annotation3), same subdir structure;
      each XML filename (without .xml) matches an image filename (without extension).
    - class_names: list/tuple of class names.
      OD-WSCL-compatible rule in this dataset:
      label 0 is background, foreground labels are 1..N.
      If "__background__" is not provided, it is prepended automatically.
    - use_difficult: include objects with difficult=1 in annotations.
    - transforms: callable(image, target, rois) -> (image, target, rois).
    """

    def __init__(
        self,
        image_root,
        annotation_root,
        class_names=None,
        use_difficult=False,
        transforms=None,
        min_size=None,
        proposal_file=None,
        pseudo_box_file=None,
        proposal_top_k=2000,
        label_csv=None,
    ):
        if BoxList is None:
            raise ImportError("wetectron.structures.bounding_box.BoxList is required for this dataset.")
        if Click is None:
            raise ImportError("wetectron.structures.keypoint.Click is required for point-based weak supervision.")
        self.image_root = os.path.normpath(image_root)
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.min_size = min_size
        self.label_csv = label_csv

        # OD-WSCL mapping rule: background id=0, foreground ids=1..N.
        if class_names is None:
            class_names = ["__background__"]
        normalized = [str(x).lower().strip() for x in class_names if str(x).strip()]
        if not normalized:
            normalized = ["__background__"]
        if normalized[0] != "__background__":
            normalized = ["__background__"] + normalized

        self.class_names = tuple(normalized)
        self.class_to_ind = dict(zip(self.class_names, range(len(self.class_names))))

        expected_num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        if len(self.class_names) != expected_num_classes:
            raise ValueError(
                "Class count mismatch with OD-WSCL config: "
                "len(class_names_with_background)=%d, but MODEL.ROI_BOX_HEAD.NUM_CLASSES=%d. "
                "Expected mapping is background=0, foreground=1..N."
                % (len(self.class_names), expected_num_classes)
            )

        if label_csv:
            self.annotation_root = None
            loaded = load_recording_csv(label_csv, self.class_to_ind)
            self.samples = []
            for rel, class_ids in loaded:
                img_path = os.path.normpath(os.path.join(self.image_root, rel.replace("/", os.sep)))
                if not os.path.isfile(img_path):
                    raise FileNotFoundError("Image listed in %s is missing: %s" % (label_csv, img_path))
                self.samples.append((img_path, class_ids))
        else:
            if not annotation_root:
                raise ValueError("annotation_root is required when label_csv is not set.")
            self.annotation_root = os.path.normpath(annotation_root)
            self.samples = _collect_pairs(self.image_root, self.annotation_root)
            if not self.samples:
                raise FileNotFoundError(
                    "No (image, annotation) pairs found under image_root=%r and annotation_root=%r. "
                    "Check that subdirs match and each image has a same-named .xml in the corresponding annotation subdir."
                    % (self.image_root, self.annotation_root)
                )
        self.id_to_img_map = {i: img_path for i, (img_path, _) in enumerate(self.samples)}

        self.proposals = None
        self.proposal_id_to_idx = None
        self.proposal_top_k = int(proposal_top_k)
        if proposal_file:
            print("Loading proposals from: {}".format(proposal_file))
            self.proposals, self.proposal_id_to_idx = load_pseudo_box_pickle(proposal_file)

        self.pseudo_boxes = None
        self.pseudo_id_to_idx = None
        if pseudo_box_file:
            print("Loading external pseudo boxes from: {}".format(pseudo_box_file))
            self.pseudo_boxes, self.pseudo_id_to_idx = load_pseudo_box_pickle(pseudo_box_file)

    def __len__(self):
        return len(self.samples)

    def get_origin_id(self, index):
        """Return a stable id for the sample (e.g. path or stem)."""
        img_path, _ = self.samples[index]
        return os.path.splitext(os.path.basename(img_path))[0]

    def _image_size(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return img, img.size

    def _recording_target(self, class_ids, width, height):
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        target = BoxList(boxes, (width, height), mode="xyxy")
        target.add_field("labels", RecordingLabels(class_ids))
        return target

    def get_img_info(self, index):
        """Return dict with height, width, file_name (for aspect ratio grouping)."""
        img_path, payload = self.samples[index]
        rel = os.path.relpath(img_path, self.image_root)
        if isinstance(payload, str):
            try:
                anno = _parse_voc_xml(payload, self.class_to_ind, self.keep_difficult)
                if anno["im_info"] is not None:
                    h, w = anno["im_info"]
                    return {"height": h, "width": w, "file_name": rel}
            except Exception:
                pass
        img = Image.open(img_path).convert("RGB")
        return {"height": img.size[1], "width": img.size[0], "file_name": rel}

    def get_groundtruth(self, index):
        """Return BoxList for the given index (for batch sampler / evaluation)."""
        img_path, payload = self.samples[index]
        if not isinstance(payload, str):
            _img, (w, h) = self._image_size(img_path)
            return self._recording_target(payload, w, h)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        anno = _parse_voc_xml(payload, self.class_to_ind, self.keep_difficult)
        if anno["im_info"] is not None:
            gh, gw = anno["im_info"]
            w, h = gw, gh
        boxes = anno["boxes"]
        labels = anno["labels"]
        difficult = anno["difficult"]
        click = _build_click_from_boxes(boxes)
        target = BoxList(boxes, (w, h), mode="xyxy")
        target.add_field("labels", labels)
        target.add_field("difficult", difficult)
        target.add_field("click", Click(click, (w, h)))
        return target

    def __getitem__(self, index):
        img_path, payload = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if not isinstance(payload, str):
            target = self._recording_target(payload, w, h)
            target = target.clip_to_image(remove_empty=True)
        else:
            anno = _parse_voc_xml(payload, self.class_to_ind, self.keep_difficult)
            if anno["im_info"] is not None:
                gh, gw = anno["im_info"]
                w, h = gw, gh
            boxes = anno["boxes"]
            labels = anno["labels"]
            difficult = anno["difficult"]
            click = _build_click_from_boxes(boxes)

            target = BoxList(boxes, (w, h), mode="xyxy")
            target.add_field("labels", labels)
            target.add_field("difficult", difficult)
            target.add_field("click", Click(click, (w, h)))
            target = target.clip_to_image(remove_empty=True)

        stem = self.get_origin_id(index)
        if self.pseudo_boxes is not None and stem in self.pseudo_id_to_idx:
            pidx = self.pseudo_id_to_idx[stem]
            pb = self.pseudo_boxes["boxes"][pidx]
            pl = self.pseudo_boxes["labels"][pidx]
            ps = self.pseudo_boxes["scores"][pidx]
            if len(pb) > 0:
                ext = BoxList(torch.tensor(pb, dtype=torch.float32), (w, h), mode="xyxy")
                ext.add_field("labels", torch.tensor(pl, dtype=torch.int64))
                ext.add_field("scores", torch.tensor(ps, dtype=torch.float32))
                target.add_field("external_pseudo", ext)

        rois = None
        if self.proposals is not None and stem in self.proposal_id_to_idx:
            ridx = self.proposal_id_to_idx[stem]
            rb = self.proposals["boxes"][ridx]
            if len(rb) > 0:
                rois = BoxList(torch.tensor(rb, dtype=torch.float32), (w, h), mode="xyxy")
                rois = rois.clip_to_image(remove_empty=True)
                if self.proposal_top_k > 0 and len(rois) > self.proposal_top_k:
                    rois = rois[: self.proposal_top_k]

        if self.transforms is not None:
            img, target, rois = self.transforms(img, target, rois)

        if target.has_field("external_pseudo"):
            ext = target.get_field("external_pseudo")
            target.add_field("external_pseudo", ext.resize(target.size))

        return img, target, rois, index


def build_spec_dataset(
    image_root,
    annotation_root=None,
    class_names=None,
    use_difficult=False,
    transforms=None,
    cfg=None,
    is_train=True,
    proposal_file=None,
    pseudo_box_file=None,
    proposal_top_k=2000,
    label_csv=None,
):
    """
    Build SpecImgDataset for use with DataLoader.
    Example:
        from datasetbuild import build_spec_dataset
        from wetectron.data.collate_batch import BatchCollator

        dataset = build_spec_dataset(
            image_root="path/to/spectrograms",
            label_csv="path/to/recording_labels.csv",
            class_names=('__background__', 'class_a', 'class_b'),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=BatchCollator(size_divisible=0),
        )
    """
    if transforms is None:
        # Reuse wetectron's default augmentation/normalization pipeline.
        # This pipeline keeps image and BoxList targets synchronized.
        if cfg is None:
            from wetectron.config.defaults import _C as default_cfg
            cfg = default_cfg
        transforms = _build_transforms_from_cfg(cfg, is_train=is_train)

    return SpecImgDataset(
        image_root=image_root,
        annotation_root=annotation_root,
        class_names=class_names,
        use_difficult=use_difficult,
        transforms=transforms,
        proposal_file=proposal_file,
        pseudo_box_file=pseudo_box_file,
        proposal_top_k=proposal_top_k,
        label_csv=label_csv,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python datasetbuild.py <image_root> <annotation_root>")
        sys.exit(1)
    image_root = sys.argv[1]
    annotation_root = sys.argv[2]
    try:
        ds = build_spec_dataset(image_root, annotation_root, class_names=["rain", "insect", 'frog', 'flow', 'dog', 'construction', 'chicken', 'birds'])
        print("Dataset size:", len(ds))
        if len(ds) > 0:
            img, target, rois, idx = ds[0]
            if torch.is_tensor(img):
                # Tensor shape is [C, H, W]
                if img.dim() >= 3:
                    img_size = (int(img.shape[-1]), int(img.shape[-2]))  # (W, H)
                else:
                    img_size = tuple(int(v) for v in img.shape)
            elif hasattr(img, "size"):
                # PIL Image.size is already a (W, H) tuple
                img_size = img.size
            else:
                img_size = "unknown"
            print("Sample 0: image size", img_size, "boxes", target.bbox.shape)
    except Exception as e:
        print("Error:", e)
        raise
    data_loader = torch.utils.data.DataLoader(
        ds,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: list(zip(*batch)),
        worker_init_fn=None
    )

    # Validate batch extraction: print basic batch/sample stats.
    max_batches_to_print = 3
    for batch_id, batch in enumerate(data_loader):
        images, targets, rois_list, indices = batch
        print("\nBatch", batch_id, "batch_size=", len(images))
        print("indices:", list(indices))

        for i, (img_i, target_i) in enumerate(zip(images, targets)):
            if torch.is_tensor(img_i) and img_i.dim() >= 3:
                sample_size = (int(img_i.shape[-1]), int(img_i.shape[-2]))  # (W, H)
            elif hasattr(img_i, "size"):
                sample_size = img_i.size
            else:
                sample_size = "unknown"
            n_boxes = int(target_i.bbox.shape[0]) if hasattr(target_i, "bbox") else -1
            print("  sample", i, "size=", sample_size, "num_boxes=", n_boxes)

        if batch_id + 1 >= max_batches_to_print:
            break
