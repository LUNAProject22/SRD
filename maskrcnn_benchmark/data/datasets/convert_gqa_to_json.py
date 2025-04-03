import os
import json
import random
import numpy as np
from tqdm import tqdm
from yacs.config import CfgNode as CN

from maskrcnn_benchmark.data.datasets.gqa import GQADataset
from maskrcnn_benchmark.utils.miscellaneous import bbox_overlaps


random.seed(0)

# raw GQA/train scene graph annotation
raw_graph_file = "/data/dataset/GQA/train_sceneGraphs.json"
raw_graph = json.load(open(raw_graph_file)) # 74942

# configs
img_dir = "datasets/gqa/images"
train_file = "datasets/gqa/GQA_200_Train.json"
test_file = "datasets/gqa/GQA_200_Test.json"
dict_file = "datasets/gqa/GQA_200_ID_Info.json"

relation_cfg = CN()
relation_cfg.ENABLE = True
relation_cfg.NEW_TRIPLE_FILE = "output/GQA/relabel-transformer/N100/relation_cache_transformer_pretrain_idf_N100_CD.npy"
suffix = "_transformer_N100"

# output
out_dir = "output/GQA/"
new_graph_file = f"gqa200{suffix}.json"
fuse_graph_file = f"train_sceneGraphs_fuse{suffix}.json"


def construct_relation_map(num_box, relation):
    relation_map = np.zeros((num_box, num_box), dtype=int)
    for k in range(relation.shape[0]):
        if relation_map[int(relation[k, 0]), int(relation[k, 1])] > 0:
            if (random.random() > 0.5):
                relation_map[int(relation[k, 0]), int(relation[k, 1])] = int(relation[k, 2])
        else:
            relation_map[int(relation[k, 0]), int(relation[k, 1])] = int(relation[k, 2])
    return relation_map


def merge_gqa200_old_new(train_data, out_file):
    result_dict = {}

    ind_to_classes = train_data.ind_to_classes
    ind_to_predicates = train_data.ind_to_predicates

    for i in tqdm(range(len(train_data.filenames))):
        filename = train_data.filenames[i]
        image_id = filename.split(".")[0]
        infos = train_data.img_info[i]
        gt_boxes = train_data.gt_boxes[i].copy()
        gt_classes = train_data.gt_classes[i].copy()
        relation = train_data.relationships[i].copy()

        if relation_cfg.ENABLE:
            new_rel = train_data.new_triple[i]
            if new_rel is not None and new_rel.shape[0] > 0:
                relation = np.vstack((relation, new_rel.copy())) # (num_rel, 3)

        # construct 2D relation matrix
        relation_map = construct_relation_map(gt_boxes.shape[0], relation.copy())

        objects = {}
        # iterate through objects
        for j, obj_id in enumerate(gt_classes):
            obj = {}
            obj["name"] = ind_to_classes[obj_id]
            obj["x"] = int(gt_boxes[j][0])
            obj["y"] = int(gt_boxes[j][1])
            obj["w"] = int(gt_boxes[j][2] - gt_boxes[j][0])
            obj["h"] = int(gt_boxes[j][3] - gt_boxes[j][1])
            obj["attributes"] = []
            obj["relations"] = []
            nonzero = np.nonzero(relation_map[j])[0]
            for idx in nonzero:
                rel = relation_map[j][idx]
                rel_dict = {"name": ind_to_predicates[rel], "object": str(idx)}
                obj["relations"].append(rel_dict)
            objects[str(j)] = obj

        result_dict[image_id] = {"width": infos["width"], 
                                "height": infos["height"], 
                                "objects": objects}

    with open(os.path.join(out_dir, out_file), "w") as f:
        json.dump(result_dict, f)
    print("Saved result file at", os.path.join(out_dir, out_file))


def gen_gqa200_new_graph(train_data, out_file):
    result_dict = {}

    ind_to_classes = train_data.ind_to_classes
    ind_to_predicates = train_data.ind_to_predicates

    for i in tqdm(range(len(train_data.filenames))):
        filename = train_data.filenames[i]
        image_id = filename.split(".")[0]
        infos = train_data.img_info[i]
        gt_boxes = train_data.gt_boxes[i].copy()
        gt_classes = train_data.gt_classes[i].copy()

        new_rel = train_data.new_triple[i]
        if new_rel is not None and new_rel.shape[0] > 0:
            relation = new_rel.copy() # (num_rel, 3)

            # construct 2D relation matrix
            relation_map = construct_relation_map(gt_boxes.shape[0], relation.copy())

            objects = {}
            # iterate through objects
            for j, obj_id in enumerate(gt_classes):
                obj = {}
                obj["name"] = ind_to_classes[obj_id]
                obj["x"] = int(gt_boxes[j][0])
                obj["y"] = int(gt_boxes[j][1])
                obj["w"] = int(gt_boxes[j][2] - gt_boxes[j][0])
                obj["h"] = int(gt_boxes[j][3] - gt_boxes[j][1])
                obj["attributes"] = []
                obj["relations"] = []
                nonzero = np.nonzero(relation_map[j])[0]
                for idx in nonzero:
                    rel = relation_map[j][idx]
                    rel_dict = {"name": ind_to_predicates[rel], "object": str(idx)}
                    obj["relations"].append(rel_dict)
                objects[str(j)] = obj

            result_dict[image_id] = {"width": infos["width"], 
                                    "height": infos["height"], 
                                    "objects": objects}
    save_file = os.path.join(out_dir, out_file)
    with open(save_file, "w") as f:
        json.dump(result_dict, f)
    print("Saved result file at", save_file)
    return result_dict


def get_object_info_lists(objects):
    ids = []
    bboxes = []
    names = []
    for id, value in objects.items():
        ids.append(id)
        names.append(value["name"])
        bboxes.append([value["x"], value["y"], value["x"]+value["w"], value["y"]+value["h"]])
    return ids, bboxes, names


def obj_id_map(new_gr_ids, matched_obj_ids):
    m = {}
    for k, v in zip(new_gr_ids, matched_obj_ids):
        m[k] = v
    return m


def merge_new_and_raw_graph(raw_graph, new_graph):
    for image_id, new_gr in new_graph.items():
        raw_gr = raw_graph[image_id]
        new_gr_objs = new_gr["objects"]
        new_gr_ids, new_gr_bboxes, new_gr_names = get_object_info_lists(new_gr_objs)
        raw_gr_ids, raw_gr_bboxes, raw_gr_names = get_object_info_lists(raw_gr["objects"])
        ious = bbox_overlaps(new_gr_bboxes, raw_gr_bboxes)
        max_iou = np.max(ious, axis=1)
        max_idx = np.argmax(ious, axis=1)
        matched_obj_ids = [raw_gr_ids[i] for i in max_idx]
        # matched_obj_names = [raw_gr_names[i] for i in max_idx]
        # assert new_gr_names == matched_obj_names
        m = obj_id_map(new_gr_ids, matched_obj_ids)
        # correct obj ids in new graphs
        for idx in new_gr_ids:
            mapped_id = m[idx]
            obj = new_gr_objs[idx]
            new_relations = obj["relations"]
            if len(new_relations) > 0:
                mapped_rels = []
                for rel in new_relations:
                    old_obj_id = rel["object"]
                    rel["object"] = m[old_obj_id]
                    mapped_rels.append(rel)
                # append new relation to raw relations
                raw_gr["objects"][mapped_id]["relations"].extend(mapped_rels)
    return raw_graph

def count_relation(graph):
    cnt = 0
    for k, v in graph.items():
        objects = v["objects"]
        for id, obj in objects.items():
            rel = obj["relations"]
            cnt += len(rel)
    print(f"Total relations:", cnt)



save_file = os.path.join(out_dir, new_graph_file)
if os.path.exists(save_file):
    new_graph = json.load(open(save_file))
else:
    # initialize dataset
    train_data = GQADataset(split='train', img_dir=img_dir, train_file=train_file,
                            dict_file=dict_file, test_file=test_file, num_val_im=5000,
                            filter_duplicate_rels=False, relation_cfg=relation_cfg)
    new_graph = gen_gqa200_new_graph(train_data, out_file=new_graph_file)

count_relation(raw_graph)
count_relation(new_graph)

fuse_graph = merge_new_and_raw_graph(raw_graph, new_graph)
count_relation(fuse_graph)

fuse_graph_file = os.path.join(out_dir, fuse_graph_file)
with open(fuse_graph_file, "w") as f:
    json.dump(fuse_graph, f)
print("Saved result file at", fuse_graph_file)
