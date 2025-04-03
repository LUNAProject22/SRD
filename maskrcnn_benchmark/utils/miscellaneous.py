# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import json
import logging
import os
from .comm import is_main_process
import numpy as np
import pickle
import torch
from collections import defaultdict, Counter

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_labels(dataset_list, output_dir):
    if is_main_process():
        logger = logging.getLogger(__name__)

        ids_to_labels = {}
        for dataset in dataset_list:
            if hasattr(dataset, 'categories'):
                ids_to_labels.update(dataset.categories)
            else:
                logger.warning("Dataset [{}] has no categories attribute, labels.json file won't be created".format(
                    dataset.__class__.__name__))

        if ids_to_labels:
            labels_file = os.path.join(output_dir, 'labels.json')
            logger.info("Saving labels mapping into {}".format(labels_file))
            with open(labels_file, 'w') as f:
                json.dump(ids_to_labels, f, indent=2)


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())


def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))

def bbox_overlaps(boxes1, boxes2):
    """
    Parameters:
        boxes1 (m, 4) [List or np.array] : bounding boxes of (x1,y1,x2,y2)
        boxes2 (n, 4) [List or np.array] : bounding boxes of (x1,y1,x2,y2)
    Return:
        iou (m, n) [np.array]
    """
    boxes1 = BoxList(boxes1, (0, 0), 'xyxy')
    boxes2 = BoxList(boxes2, (0, 0), 'xyxy')
    iou = boxlist_iou(boxes1, boxes2).cpu().numpy()
    return iou


def boxlist_inter_over_min(boxlist1, boxlist2):
    import torch
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    min_of_two = torch.zeros((N, M))
    for i in range(N):
        a1 = area1[i].repeat(M)
        compare = torch.cat((a1[:, None], area2[:, None]), dim=-1)
        min_of_two[i] = torch.min(compare, dim=-1)[0]
    assert min_of_two.shape == inter.shape
    iom = inter / min_of_two
    return iom


def bbox_iom(boxes1, boxes2):
    """
    Parameters:
        boxes1 (m, 4) [List or np.array] : bounding boxes of (x1,y1,x2,y2)
        boxes2 (n, 4) [List or np.array] : bounding boxes of (x1,y1,x2,y2)
    Return:
        iom (m, n) [np.array]
    """
    boxes1 = BoxList(boxes1, (0, 0), 'xyxy')
    boxes2 = BoxList(boxes2, (0, 0), 'xyxy')
    
    iom = boxlist_inter_over_min(boxes1, boxes2).cpu().numpy()
    return iom


def filter_duplicate_rel(relation):
    all_rel_sets = defaultdict(list)
    for (o0, o1, r) in relation:
        all_rel_sets[(o0, o1)].append(r)
    relation = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
    relation = np.array(relation, dtype=np.int32)
    return relation


def filter_new_triples(gt_triple, new_triple):
    if new_triple is None:
        return gt_triple, None
    res_1 = {}
    for (o0, o1, r) in gt_triple:
        res_1[(o0, o1)] = (r)
    res_2 = {}
    for (o0, o1, r) in new_triple:
        if (o0, o1) not in res_1 and o0 != o1:
            res_2[(o0, o1)] = (r)
    res_1 = [(k[0], k[1], v) for k,v in res_1.items()]
    res_1 = np.array(res_1, dtype=np.int32)
    if res_2 != {}:
        res_2 = [(k[0], k[1], v) for k,v in res_2.items()]
        res_2 = np.array(res_2, dtype=np.int32)
    else:
        res_2 = None
    return res_1, res_2


def loadf(filename):
    if filename.endswith('.json'):
        with open(filename, 'rb') as f:
            data = json.load(f)
    elif filename.endswith('.pkl') or filename.endswith('.pk'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    elif filename.endswith('.pth'):
        data = torch.load(filename, map_location=torch.device("cpu"))
    else:
        raise NameError('This function does not support loading file format {}'.format(filename))
    return data

def count_triples(triple):
    count = 0
    for i in range(len(triple)):
        if triple[i] is not None and triple[i] != torch.Tensor([]) and len(triple[i].shape)>1:
            count += triple[i].shape[0]
    return count

def load_recall20(result_pth, save_name):
    recall_file = os.path.join("datasets/rwt_weights", save_name)
    if os.path.exists(recall_file):
        return torch.load(recall_file)
    os.makedirs("datasets/rwt_weights", exist_ok=True)
    result_dict = torch.load(result_pth)
    r20_list = result_dict["predcls_mean_recall_list"][20]
    r20_list_51 = [0] + r20_list
    r20 = torch.from_numpy(np.array(r20_list_51))
    torch.save(r20, recall_file)
    return r20

def _get_reweighting_dic(relations, num_predicates):
    """
    weights for each predicate
    weight is the inverse frequency normalized by the median
    Returns:
        {1: f1, 2: f2, ... 100: f100}
    """
    rels = [x[:, 2] for x in relations if x is not None and x!=torch.Tensor([]) and len(x.shape)>1]
    rels = [int(y) for x in rels for y in x]
    rels = Counter(rels)
    rels = dict(rels)
    if len(rels.keys()) < num_predicates - 1:
        for i in range(1, num_predicates): # labels are in range 1, 100
            if i not in rels.keys():
                rels[i] = -1
    rels = [rels[i] for i in sorted(rels.keys())]
    vals = sorted(rels)
    rels_t = torch.tensor([-1.]+rels)
    rels_t = (1./rels_t) * np.median(vals)
    # freq_rels = [31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43, 40, 49, 41, 23,
    #         38, 6, 7, 33, 11, 46, 16, 25, 47, 19, 5, 9, 35, 24, 10, 4, 14, 
    #         13, 12, 36, 44, 42, 32, 2, 28, 26, 45, 3, 17, 18, 34, 27, 37, 39, 15]
    # for e in freq_rels:
    #     print(round(rels_t[e].item(), 4))
    return rels_t

def _get_combine_reweighting_dic(all_rels, r20_file, num_predicates):
    statis_weights = _get_reweighting_dic(all_rels, num_predicates)
    recall = torch.load(r20_file).float()
    comb_weights = 0.5 * statis_weights + recall
    return comb_weights


if __name__=="__main__":
    result_pth = "checkpoints/transformer/transformer-precls/inference/VG_stanford_filtered_with_attribute_test/result_dict.pytorch"
    save_name = "recall20_transformer_baseline.pth"
    load_recall20(result_pth, save_name)
