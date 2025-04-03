# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset
from .visual_genome import VGDataset
from .action_genome import JsonDatasetRel
from .vg_new_relation import NewRelationData
from .gqa import GQADataset
from .gqa_relation_deduction import GQARelabeling

__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "VGDataset", "JsonDatasetRel",
           "NewRelationData", "GQADataset", "GQARelabeling"]
