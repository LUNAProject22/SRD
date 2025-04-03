# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch
import numpy as np
from maskrcnn_benchmark.utils.boxes import box_filter


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def filter_none_overlapping_prop(proposals):
    boxes = [proposal.bbox.clone().cpu().numpy() for proposal in proposals]
    o1o2_total = []
    for i in range(len(boxes)):
        o1o2_i = box_filter(boxes[i], must_overlap=True)
        o1o2_i = torch.from_numpy(np.array(o1o2_i, dtype=int)).to(device=proposals[0].bbox.device)
        o1o2_total.append(o1o2_i)
    return o1o2_total