
import pickle as pkl
import numpy as np
from pprint import pprint
import json

import h5py
import time
from torch.nn.functional import normalize


BOX_SCALE = 1024


class FreqContext_LPNoContextTail:
    def __init__(self, prior_file='data/kg_inference/prior_knowledge/freq_context-lp_no_context_by_tail.pkl',
                 index_file='data/VisualGnome/preprocessed_codebase/VG-SGG-dicts.json',
                 # dim_head=0, dim_tail=1, dim_predicate=2
                 ):
        priors = pkl.load(open(prior_file, 'rb'))
        self.lp_no_context_tail_matrix = priors['lp_no_context_matrix_norm_by_tail'].permute(1, 2, 0)  # (head, tail, predicate)
        # self.dim_head = dim_head
        # self.dim_tail = dim_tail
        # self.dim_predicate = dim_predicate
        # self.ht2ct_keys: (head, tail) --> list of contexts
        # self.ht2ct_values: (head, tail) --> frequency of that pair in the corresponding context
        self.ht2ct_keys, self.ht2ct_values = self.prepare_ht2context_count(priors['head_tail_to_context_count'])
        self.label_to_idx, self.idx_to_label, self.predicate_to_idx, self.idx_to_predicate = self.load_mappings(index_file)

        self.ic_based_matrix = None
        self.ic_based_pair_prob = None

    def predict(self, head, tail, input_context=None, with_normalization=True, verbose=False):
        '''
        :param head: 1d-tensor head indices
        :param tail:
        :param input_context:
        :param verbose:
        :return:
        '''
        if self.ic_based_matrix is None:
            if input_context is None:
                print("Input context is required for the first time inference (to create the matrix)")
                return None
            else:
                self.prepare_matrix(input_context=input_context, verbose=verbose, with_normalization=with_normalization)
        if isinstance(head, str):
            head = self.get_index_for(head)
        if isinstance(tail, str):
            tail = self.get_index_for(tail)
        # vec = self.norm_matrix[:, head, tail]
        return self.ic_based_matrix[head, tail, :]

    def prepare_matrix(self, input_context, verbose=False, with_normalization=False, p=1.0, dim=2):
        '''
        Compute sim x freq(h, t) --> normalize across all pairs of (h,t)
        :param input_context:
        :param verbose:
        :return:
        '''
        # create all pairs
        all_pairs = self.create_all_pairs(input_context)
        # get all contexts matched for these pairs
        matched_contexts = self.get_matched_context(all_pairs)
        if verbose:
            print("#matched_contexts: {}".format(len(matched_contexts)))
        # compute the sim scores
        start = time.time()
        ct2score = self.compute_contexts_similarity(input_context, matched_contexts)
        if verbose:
            print("Time: {}".format(time.time() - start))
        # get pair2scores
        pair2scores = self.get_context_sim_array(all_pairs, ct2score)

        # compute accumulated weighted-count for each pair
        pair2count = self.compute_accumulated_weighted_count(pair2scores, verbose=verbose)

        # compute prob and sort pairs
        pair2prob = self.compute_probability_for_dict(pair2count)  # p(h, t | IC)
        if verbose:
            print("Pair 2 probability")
            pprint(pair2prob)
        self.ic_based_pair_prob = pair2prob
        # prepare weighted matrix = p(h, t | IC) * p(r|h,t)
        self.compute_weighted_matrix(self.ic_based_pair_prob, with_normalization=with_normalization, p=p, dim=dim, verbose=verbose)

    def get_pred_vector(self, head, tail, matrix):
        h = self.get_index_for(head)
        t = self.get_index_for(tail)
        return matrix[h, t, :]

    def get_index_for(self, value, is_predicate=False):
        if isinstance(value, str):
            # should come in here
            value = value.strip()
            if is_predicate:
                value = self.predicate_to_idx.get(value, 0)
            else:
                value = self.label_to_idx.get(value, 0)
        return value

    def compute_weighted_matrix(self, ht2prob, with_normalization=False, p=1.0, dim=2, verbose=False):
        self.ic_based_matrix = self.lp_no_context_tail_matrix.clone()  # (head, tail, relation)
        for ht, p in ht2prob.items():
            h = self.get_index_for(ht[0])
            t = self.get_index_for(ht[1])
            if verbose:
                print("ht: {}".format(ht))
                print("h: {}".format(h))
                print("t: {}".format(t))
                print("Matrix (before): {}".format(self.ic_based_matrix[h, t, :]))
                print("p: {}".format(p))
            self.ic_based_matrix[h, t, :] = self.ic_based_matrix[h, t, :] * p
            if verbose:
                print("Matrix (after): {}".format(self.ic_based_matrix[h, t, :]))
                # input()
        # normalize
        if with_normalization:
            if verbose:
                print("Before normalization: {}".format(self.get_pred_vector('logo', 'shirt', self.ic_based_matrix)))
                print("Matrix size: {}. Dim: {}".format(self.ic_based_matrix.shape, dim))
            self.ic_based_matrix = normalize(self.ic_based_matrix, p=p, dim=dim)
            if verbose:
                print("After normalization: {}".format(self.get_pred_vector('logo', 'shirt', self.ic_based_matrix)))

    def prepare_ht2context_count(self, ht2context_count):
        ht2ct_keys = {}
        ht2ct_values = {}
        for ht, ct2c in ht2context_count.items():
            ht2ct_keys[ht] = list(ct2c.keys())
            ht2ct_values[ht] = np.asarray(list(ct2c.values()))
        return ht2ct_keys, ht2ct_values

    def get_matched_context(self, pairs):
        tmp = []
        for p in pairs:
            ct = self.ht2ct_keys.get(p, None)
            if ct is not None:
                tmp += ct
        return list(set(tmp))

    def get_context_sim_array(self, pairs, ct2score):
        pair2scores = {}  # pair -> context similarity scores, placed in the same order as in the self.ht2ct_keys
        for p in pairs:
            cts = self.ht2ct_keys.get(p, None)
            if cts is not None:
                tmp = [ct2score.get(ct, 0) for ct in cts]
                pair2scores[p] = np.asarray(tmp)
        return pair2scores

    def compute_accumulated_weighted_count(self, pair2scores, verbose=False):
        p2acc = {}
        for p, scores in pair2scores.items():
            acc = scores * self.ht2ct_values[p]
            p2acc[p] = np.sum(acc)
            if verbose:
                print("Pairs: {}".format(p))
                print("Scores: {}".format(scores))
                print("Prior-Counts: {}".format(self.ht2ct_values[p]))
                print("Accumulated arr: {}".format(acc))
                print("Accumulated value: {}".format(p2acc[p]))
                # input()
        return p2acc

    def get_triples_using_best_predicate(self, pair2prob, verbose=False):
        # for each pair using only 1 best predicate (with constraint)
        triple_accumulate = {}
        for ht, prob in pair2prob.items():
            if prob == 0:
                break
            best_rel = self.head_tail_best_rel[ht]
            rel = best_rel[0]
            rel_prob = best_rel[2]
            tp = ht + (rel, )
            triple_accumulate[tp] = prob * rel_prob
        triple_prob = self.compute_probability_for_dict(triple_accumulate)
        sorted_triples = sorted(triple_prob.items(), key=lambda x:x[1], reverse=True)
        if verbose:
            print("Sorted triples")
            pprint(sorted_triples)
        return sorted_triples

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

    def compute_contexts_similarity(self, input_context, matched_contexts):
        ct2score = {}
        sip = set(input_context)
        for ct in matched_contexts:
            ct2score[ct] = self.jaccard_similarity(set(ct), sip)
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



if __name__ == "__main__":
    prior_file = 'datasets/kg/freq_context-lp_no_context_by_tail.pkl'
    index_file = 'datasets/vg/VG-SGG-dicts.json'
    # initialize the predictor
    predictor = FreqContext_LPNoContextTail(prior_file=prior_file, index_file=index_file)

    icontext = ['arm', 'boy', 'food', 'logo', 'shirt']

    # prepare ic-based matrix
    predictor.prepare_matrix(icontext, verbose=False)
    # obtain prediction
    prediction = predictor.predict('logo', 'shirt')
    print("[ic_based_matrix] (logo, shirt): {}".format(predictor.get_pred_vector('logo', 'shirt', predictor.ic_based_matrix)))
    print("[lp_no_context_tail_matrix] (logo, shirt): {}".format(predictor.get_pred_vector('logo', 'shirt', predictor.lp_no_context_tail_matrix)))
    print("Predicate-Vector prediction: {}".format(prediction))
    print("pair prob: {}".format(predictor.ic_based_pair_prob[('logo', 'shirt')]))
