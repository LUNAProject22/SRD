import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import copy
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.miscellaneous import (loadf, count_triples, _get_reweighting_dic, \
                                                    _get_combine_reweighting_dic)

BOX_SCALE = 1024  # Scale at which we have the boxes

import matplotlib.pyplot as plt # display image
import matplotlib.image as mpimg # read image


class GQADataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, dict_file, train_file, test_file, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=False, flip_aug=False, custom_eval=False, custom_path='',
                relation_cfg=None, logits_cfg=None, wsupervise_cfg=None):
        """
        Torch dataset for GQA200
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all gqa images
            dict_file: JSON Contains mapping of classes/relationships to words
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        # num_im = 10000
        # num_val_im = 4

        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.train_file = train_file
        self.test_file = test_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        print('\nwe change the gqa get ground-truth!\n')

        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file) # contiguous 151, 51 containing __background__
        self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        if self.split == 'train':
            self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
                self.train_file, self.split)
        else:
            self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
                self.test_file, self.split)

        self.rwt = False
        if self.split=="train":
            all_rels = copy.deepcopy(self.relationships)
            self.relation_cfg = relation_cfg
            # Load new relation (non-existing pairs)
            if self.relation_cfg is not None and self.relation_cfg.ENABLE:
                if os.path.exists(self.relation_cfg.NEW_TRIPLE_FILE):
                    print("Loading relation cache from file:", self.relation_cfg.NEW_TRIPLE_FILE)
                    self.new_triple = np.load(self.relation_cfg.NEW_TRIPLE_FILE, allow_pickle=True)[()]
                    all_rels.extend(self.new_triple)
                    print(f"new triple: {count_triples(self.new_triple)}")

            if wsupervise_cfg is not None and wsupervise_cfg.RWT:
                self.rwt = True
                # construct a reweighting dic
                if wsupervise_cfg.FIX_WEIGHTS:
                    data_to_compute_weights = copy.deepcopy(self.relationships)
                else:
                    data_to_compute_weights = all_rels
                self.reweighting_dic = _get_reweighting_dic(data_to_compute_weights, num_predicates=len(self.ind_to_predicates))
                # overwrite prev rwt dic
                if wsupervise_cfg.RECALL_FILE != "":
                    self.reweighting_dic = _get_combine_reweighting_dic(data_to_compute_weights, wsupervise_cfg.RECALL_FILE,
                                                                        num_predicates=len(self.ind_to_predicates))
            print(f"original triple: {count_triples(self.relationships)}")

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.filenames[index])).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']), ' ', str(self.img_info[index]['height']), ' ', '='*20)

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')
        
        target = self.get_groundtruth(index, flip_img=flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_img_info(self, index):
        return self.img_info[index]

    def get_statistics(self):
        fg_matrix, bg_matrix = get_GQA_statistics(img_dir=self.img_dir, train_file=self.train_file,
                                                  dict_file=self.dict_file,
                                                  must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)
        freq_dist = fg_matrix / (fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'freq_dist': torch.from_numpy(freq_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': []
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width':int(img.width), 'height':int(img.height)})

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index].copy()
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

        if flip_img:
            new_xmin = w - box[:,2]
            new_xmax = w - box[:,0]
            box[:,0] = new_xmin
            box[:,2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy') # xyxy

        tgt_labels = torch.from_numpy(self.gt_classes[index])
        target.add_field("labels", tgt_labels.long())

        relation = self.relationships[index].copy()  # (num_rel, 3)

        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add new relation for training
        if hasattr(self, "new_triple"):
            new_rel = self.new_triple[index]
            if new_rel is not None and new_rel.shape[0] > 0:
                relation = np.vstack((relation, new_rel.copy())) # (num_rel, 3)

        # reweighting
        if self.rwt:
            relation_tuple = relation
            pairs = relation_tuple[:, :2]
            rel_lbs = relation_tuple[:, 2]
            relation_labels = torch.zeros((rel_lbs.shape[0], len(self.ind_to_predicates)))
            # relation_labels: [0, 0, 0, 1, ..., 0]
            relation_labels[torch.arange(0, relation_labels.size(0)), rel_lbs] = 1.

            assert ~(rel_lbs == 0).any(), rel_lbs
            weights = self.reweighting_dic[rel_lbs]
            # put the weight at the predicate 0, which is background
            # the loss function will extract this
            # relation_labels: [weight, 0, 0, 1, ..., 0]
            relation_labels[:, 0] = -weights
            # relation pair indexes
            target.add_field("relation_pair_idxs", torch.from_numpy(pairs).long())
            target.add_field("relation_labels", relation_labels)

        # add relation to target
        num_box = box.shape[0]
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
        else:
            target = target.clip_to_image(remove_empty=False) # True
        
        target.add_field("image_path", os.path.join(self.img_dir, self.filenames[index]))
        # # use the following code only when run "tools/relation_relabel.py"
        # if self.split=="train":
        #     curr_d = {
        #         'image_id': int(self.filenames[index].split(".")[0]),
        #         'width': self.img_info[index]['width'],
        #         'height': self.img_info[index]['height'],
        #         'img_path': os.path.join(self.img_dir, self.filenames[index]),
        #         'boxes': self.gt_boxes[index],
        #         'labels': self.gt_classes[index],
        #         'triples': self.relationships[index],
        #         'relation_map': relation_map
        #     }
        #     target.add_field("train_data", curr_d)
        return target

    def __len__(self):
        return len(self.filenames)
    
    def save_sup_data(self, specified_data_file):
        import pickle
        assert self.split=='train'
        data = []
        for filename, gt_class, gt_box, img_info, triples in \
            zip(self.filenames, self.gt_classes, self.gt_boxes, self.img_info, self.relationships):
            curr_dict = {}
            curr_dict['image_id'] = int(filename.split(".")[0])
            curr_dict['width'] = img_info['width']
            curr_dict['height'] = img_info['height']
            curr_dict['img_path'] = os.path.join(self.img_dir, filename)
            curr_dict['boxes'] = gt_box
            curr_dict['labels'] = gt_class
            curr_dict['triples'] = triples
            data.append(curr_dict)
        pickle.dump(data, open(specified_data_file, "wb"))
        return data


def get_GQA_statistics(img_dir, train_file, dict_file, must_overlap=True):
    train_data = GQADataset(split='train', img_dir=img_dir, train_file=train_file,
                           dict_file=dict_file, test_file=None, num_val_im=5000,
                           filter_duplicate_rels=False)
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(float), boxes.astype(float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, :2], boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:], boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:
        json.dump(data, outfile)

def load_info(dict_file):
    info = json.load(open(dict_file, 'r'))
    ind_to_classes = info['ind_to_classes']
    ind_to_predicates = info['ind_to_predicates']
    return ind_to_classes, ind_to_predicates


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return: 
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(data_json_file, split):
    data_info_all = json.load(open(data_json_file, 'r'))
    filenames = data_info_all['filenames_all']
    img_info = data_info_all['img_info_all']
    gt_boxes = data_info_all['gt_boxes_all']
    gt_classes = data_info_all['gt_classes_all']
    relationships = data_info_all['relationships_all']

    output_filenames = []
    output_img_info = []
    output_boxes = []
    output_classes = []
    output_relationships = []

    items = 0
    for filename, imginfo, gt_b, gt_c, gt_r in zip(filenames, img_info, gt_boxes, gt_classes, relationships):
        len_obj = len(gt_b)
        items += 1

        if split == 'val' or split == 'test':
            if items == 5580:
                continue

        if len(gt_r) > 0 and len_obj > 0:
            output_filenames.append(filename)
            output_img_info.append(imginfo)
            output_boxes.append(np.array(gt_b))
            output_classes.append(np.array(gt_c))
            output_relationships.append(np.array(gt_r))


    if split == 'val':
        output_filenames = output_filenames[:5000]
        output_img_info = output_img_info[:5000]
        output_boxes = output_boxes[:5000]
        output_classes = output_classes[:5000]
        output_relationships = output_relationships[:5000]

    return output_filenames, output_img_info, output_boxes, output_classes, output_relationships

