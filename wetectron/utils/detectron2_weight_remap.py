# Map Detectron2 COCO Faster R-CNN R-50-C4 zoo checkpoints onto wetectron key names.
# Backbone maps cleanly; RPN cls/bbox head shapes may differ if anchor counts differ.


def remap_detectron2_r50_c4_key(k: str) -> str:
    if k.startswith("proposal_generator."):
        if k.startswith("proposal_generator.anchor_generator"):
            return k.replace("proposal_generator.", "rpn.", 1)
        k = k.replace("proposal_generator.rpn_head.conv.", "rpn.head.conv.")
        k = k.replace("proposal_generator.rpn_head.objectness_logits.", "rpn.head.cls_logits.")
        k = k.replace("proposal_generator.rpn_head.anchor_deltas.", "rpn.head.bbox_pred.")
        return k
    if not k.startswith("backbone."):
        return k
    if k.startswith("backbone.stem."):
        k = k.replace("backbone.stem.conv1.norm.", "backbone.body.stem.bn1.")
        k = k.replace("backbone.stem.", "backbone.body.stem.")
        return k
    k = k.replace("backbone.res2.", "backbone.body.layer1.")
    k = k.replace("backbone.res3.", "backbone.body.layer2.")
    k = k.replace("backbone.res4.", "backbone.body.layer3.")
    k = k.replace("backbone.res5.", "backbone.body.layer4.")
    k = k.replace(".conv1.norm.", ".bn1.")
    k = k.replace(".conv2.norm.", ".bn2.")
    k = k.replace(".conv3.norm.", ".bn3.")
    k = k.replace(".shortcut.norm.", ".downsample.1.")
    k = k.replace(".shortcut.weight", ".downsample.0.weight")
    return k


def remap_detectron2_r50_c4_state_dict_if_needed(state):
    """If ``state`` looks like a Detectron2 ResNet-C4 detector, remap backbone (+proposal) keys."""
    if state is None:
        return state
    keys = list(state.keys())
    if not keys or not any(k.startswith("backbone.res2") for k in keys):
        return state
    out = {}
    for k, v in state.items():
        if k.startswith("backbone.") or k.startswith("proposal_generator."):
            out[remap_detectron2_r50_c4_key(k)] = v
    return out
