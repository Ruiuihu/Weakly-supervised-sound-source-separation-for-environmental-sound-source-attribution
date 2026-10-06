# -*- coding: utf-8 -*-
"""Helpers for precomputed vs live-RPN proposal inputs."""
from typing import List, Optional


def uses_precomputed_proposals(rois) -> bool:
    """True when the batch carries real BoxList proposals (not placeholders)."""
    return rois is not None and len(rois) > 0 and rois[0] is not None


def eval_rois_argument(rois_batch) -> Optional[List]:
    """
    For evaluation / inference: return None when no offline proposals exist so
    GeneralizedRCNN runs live RPN + optional energy gate. Otherwise return the list.
    """
    if rois_batch is None:
        return None
    if all(r is None for r in rois_batch):
        return None
    return list(rois_batch)
