
import pickle as pkl
import numpy as np
import pandas as pd
from pprint import pprint
import json
from tqdm import tqdm
import h5py
import time
from torch.nn.functional import normalize
import torch


BOX_SCALE = 1024


class LPContextBasedPredicatePredictor:
    def __init__(self, prior_file='data/kg_inference/logic_programming/lp_context_info.pkl',
                 index_file='data/VisualGnome/preprocessed_codebase/VG-SGG-dicts.json', pred_size=51, default=0.0):
        priors = pkl.load(open(prior_file, 'rb'))
        self.ht2contexts = priors['ht2contexts']
        self.context2htr2count = priors['context2htr2count']
        self.pred_size = pred_size
        self.default = default
        self.label_to_idx, self.idx_to_label, self.predicate_to_idx, self.idx_to_predicate = self.load_mappings(index_file)
        self.max_class_idx = len(self.label_to_idx) - 1
        self.raw_matrix = None
        self.normalized_dim = -1
        self.norm_matrix = None
        self.current_IC = None

    def prepare_pht_matrix(self, input_context, matrix_normalize_p=1.0, matrix_normalize_dim=1, verbose=False):
        '''
        :param input_context:
        :param verbose:
        :param use_best_rel_only:
        :param matrix_normalize_p:  1.0: devided by sum
        :param matrix_normalize_dim: 1: normalized across tails; 2: normalize across predicates
        :return:
        '''
        # create all pairs for the given input context (IC)
        all_pairs = self.create_all_pairs(input_context)
        # get all context matched for these pairs
        matched_contexts = self.get_matched_context(all_pairs)
        if verbose:
            print("#matched_contexts: {}".format(len(matched_contexts)))
            # pprint(matched_contexts)
        # *compute context similarity score
        start = time.time()
        ct2sim_score = self.compute_contexts_similarity(input_context, matched_contexts, verbose=verbose)
        if verbose:
            print("Time: {}".format(time.time() - start))
        # new for Logic programming context based
        # Aggregate count for (head, tail, relation) based on context-sim
        agg_htr2counnt = {}  # aggregated head tail relation to count
        for ctx in matched_contexts:
            sim = ct2sim_score[ctx]
            htr2count = self.context2htr2count.get(ctx, None)
            if htr2count is None:
                print("Cannot find any information for context {}".format(ctx))
                continue
            for htr, c in htr2count.items():
                ht_tmp = (htr[0], htr[1])
                if ht_tmp in all_pairs:
                    agg_htr2counnt[htr] = agg_htr2counnt.get(htr, 0) + c * sim  # weighted count
                    if verbose:
                        print("{} += {}".format(htr, sim*c))
        # create and normalize the matrix
        # matrix [head, tail, predicate]
        self.norm_matrix, self.raw_matrix = self.create_matrix(agg_htr2counnt=agg_htr2counnt, p=matrix_normalize_p, dim=matrix_normalize_dim)
        self.current_IC = input_context
        self.normalized_dim = matrix_normalize_dim

    def predict(self, head, tail, input_context=None, matrix_normalize_p=1.0, matrix_normalize_dim=2, verbose=False):
        '''
        :param head: 1d-tensor head indices
        :param tail:
        :param input_context:
        :param matrix_normalize_p:
        :param matrix_normalize_dim:
        :param verbose:
        :return:
        '''
        if self.norm_matrix is None:
            if input_context is None:
                print("Input context is required for the first time inference (to create the PHT matrix)")
                return None
            else:
                self.prepare_pht_matrix(input_context=input_context, matrix_normalize_p=matrix_normalize_p, matrix_normalize_dim=matrix_normalize_dim, verbose=verbose)
        if isinstance(head, str):
            head, tail = self.get_indices(head, tail)
        # vec = self.norm_matrix[:, head, tail]
        return self.norm_matrix[head, tail, :]

    def get_index_for(self, value, is_predicate=False):
        if isinstance(value, str):
            value = value.strip()
            if is_predicate:
                value = self.predicate_to_idx.get(value, 0)
            else:
                value = self.label_to_idx.get(value, 0)
        return self.verify_idx(value)

    def get_indices(self, head, tail):
        if isinstance(head, str) and isinstance(tail, str):
            # head and tail is index
            head = head.strip()
            tail = tail.strip()
            head = self.label_to_idx.get(head, 0)
            tail = self.label_to_idx.get(tail, 0)
        head = self.verify_idx(head)
        tail = self.verify_idx(tail)
        return head, tail


    def verify_idx(self, idx, is_predicate=False):
        if idx < 0:
            idx = 0
        else:
            if is_predicate:
                if idx > len(self.predicate_to_idx)-1:
                    idx = 0
            else:
                if idx > self.max_class_idx:
                    idx = 0
        return idx

    def create_matrix(self, agg_htr2counnt, p=1.0, dim=1):
        '''
        :param p:
        :param dim: 1: normalized across tails; 2: normalize across predicates
        :return:
        matrix: [head, tail, predicate]
        '''
        num_class = len(self.idx_to_label) + 1  # +1 for background (or not found classes)
        num_pred = len(self.idx_to_predicate) + 1
        matrix = np.zeros((num_pred, num_class, num_class))  # pred x head x tail
        for htr, c in agg_htr2counnt.items():
            # TODO current here
            h, t, r = htr
            h = self.get_index_for(h)
            t = self.get_index_for(t)
            r = self.get_index_for(r, is_predicate=True)
            matrix[r][h][t] = c
        matrix = torch.tensor(matrix)
        matrix = matrix.permute(1, 2, 0) # [head, tail, predicate]
        norm_matrix = normalize(matrix, p=p, dim=dim)
        return norm_matrix, matrix

    def get_matched_context(self, pairs):
        tmp = []
        for p in pairs:
            ct = self.ht2contexts.get(p, None)
            if ct is not None:
                tmp += ct
        return list(set(tmp))

    def load_mappings(self, index_file):
        data = json.load(open(index_file))
        label_to_idx = data['label_to_idx']
        idx_to_label = data['idx_to_label']
        predicate_to_idx = data['predicate_to_idx']
        idx_to_predicate = data['idx_to_predicate']
        return self.convert_to_dict(label_to_idx, value_as_int=True), self.convert_to_dict(idx_to_label, key_as_int=True), \
               self.convert_to_dict(predicate_to_idx, value_as_int=True), self.convert_to_dict(idx_to_predicate, key_as_int=True)

    def convert_to_dict(self, idict, key_as_int=False, value_as_int=False):
        tmp = {}
        for k, v in idict.items():
            if key_as_int:
                k = int(k)
            if value_as_int:
                v = int(v)
            tmp[k] = v
        return tmp

    def jaccard_similarity(self, a, b):
        '''
        :param a: set
        :param b: set
        :return:
        '''
        nom = a.intersection(b)
        den = a.union(b)
        sim = len(nom)/len(den)
        return sim

    def compute_contexts_similarity(self, input_context, matched_contexts, verbose=False):
        ct2score = {}
        sip = set(input_context)
        for ct in matched_contexts:
            ct2score[ct] = self.jaccard_similarity(set(ct), sip)
            if verbose:
                print("[{:.3f}]: {}".format(ct2score[ct], ct))
        return ct2score

    def create_all_pairs(self, ilist):
        all_pairs = []
        for i in range(len(ilist)):
            for j in range(len(ilist)):
                all_pairs.append((ilist[i], ilist[j]))
        return list(set(all_pairs))

    def compute_probability_for_dict(self, idict):
        total = np.sum(list(idict.values()))
        output = {k: v/total for k, v in idict.items()}
        return output


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start : i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start : i_rel_end + 1]
            obj_idx = _relations[i_rel_start : i_rel_end + 1] - i_obj_start # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        # if filter_non_overlap:
        #     assert split == 'train'
        #     # construct BoxList object to apply boxlist_iou method
        #     # give a useless (height=0, width=0)
        #     boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
        #     inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
        #     rel_overs = inters[rels[:, 0], rels[:, 1]]
        #     inc = np.where(rel_overs > 0.0)[0]
        #
        #     if inc.size > 0:
        #         rels = rels[inc]
        #     else:
        #         split_mask[image_index[i]] = 0
        #         continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships, image_index


def load_img2triples(roidb_file='data/VisualGnome/preprocessed_codebase/VG-SGG-with-attri.h5',
                     mapping_file='data/VisualGnome/preprocessed_codebase/VG-SGG-dicts.json', setname='test'):
    split_mask, boxes, gt_classes, gt_attributes, relationships, image_index = load_graphs(roidb_file=roidb_file,
                                                                                           split=setname, num_im=-1, num_val_im=5000, filter_empty_rels=True)
    mappings = json.load(open(mapping_file))
    idx2label = mappings['idx_to_label']
    idx2pred = mappings['idx_to_predicate']
    num_imgs = len(relationships)

    img2info = {}
    print("Loading info from", setname)
    for idx in tqdm(range(num_imgs), total=num_imgs):
        img = image_index[idx]
        img_gt_classes = gt_classes[idx]
        img_rels = relationships[idx]
        context_objs = list(set([idx2label[str(a)] for a in img_gt_classes]))
        triples = []
        for rel in img_rels:
            s, o, r = rel  # s and o are the index of the objects
            s = str(img_gt_classes[s])  # get the actual object class
            o = str(img_gt_classes[o])  # get the actual object class
            stxt = idx2label[s]
            otxt = idx2label[o]
            rtxt = idx2pred[str(r)]
            tripl = (stxt, otxt, rtxt)
            triples.append(tripl)
        img2info[img] = [context_objs, triples]
    return img2info


if __name__ == "__main__":
    prior_file = 'datasets/kg/lp_context_info.pkl'
    index_file = 'datasets/vg/VG-SGG-dicts.json'
    # initialize the predictor
    predictor = LPContextBasedPredicatePredictor(prior_file=prior_file, index_file=index_file)

    # example, input should be a list of unique objects (not index)
    icontext = ['sheep', 'sheep', 'wave', 'sheep', 'sheep']
    normalize_mapping = {'by_tail': 1, 'by_predicate': 2}
    # prepare the normalized matrix (only need to run once for each input context)
    predictor.prepare_pht_matrix(icontext, matrix_normalize_dim=normalize_mapping['by_tail'], verbose=False)  # {(head, tail): vector}
    # for each pair
    prediction = predictor.predict('wave', 'wave')
    print("Predicate-Vector prediction: {}".format(prediction))

    # more details (not needed)
    # h, t = predictor.get_indices('wave', 'wave')
    # vec = predictor.raw_matrix[h, t, :]
    # vec2 = predictor.norm_matrix[h, t, :]
    # print(vec)
    # print(vec2)
    # print(prediction)
    # print("The current input context (IC): {}".format(predictor.current_IC))
    # print("The current matrix (raw context weighted frequency): {}".format(predictor.norm_matrix))
    # print("The current normalized matrix (normalized by dimension: {}): {}".format(predictor.norm_matrix, predictor.normalized_dim))
