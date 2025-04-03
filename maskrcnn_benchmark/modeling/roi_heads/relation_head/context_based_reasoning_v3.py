
import pickle as pkl
import numpy as np
import pandas as pd
from pprint import pprint
import json
from tqdm import tqdm
import h5py
import time

BOX_SCALE = 1024


class ContextBasedPredicatePredictor:
    def __init__(self, prior_file='data/KG/preprocessed/alias_filtered_codebase_h5/train_triples_context_essential_info.pkl',
                 index_file='data/VisualGnome/preprocessed_codebase/VG-SGG-dicts.json', pred_size=51, default=0.0):
        priors = pkl.load(open(prior_file, 'rb'))
        self.context_head_tail_count = priors['context_head_tail_count']
        self.head_tail_to_rel_to_count = priors['head_tail_to_rel_to_count']
        self.head_tail_best_rel = priors['head_tail_best_rel']
        self.rel_prob_given_head_tail = priors['rel_prob_given_head_tail']
        self.ht2ct_keys, self.ht2ct_values = self.prepare_ht2context_count(priors['head_tail_to_context_count'])

        self.pred_size = pred_size
        self.default = default
        self.contexts_list = pd.Series((a for a in self.context_head_tail_count.keys()))
        self.contexts_set = [set(a) for a in self.contexts_list]
        self.label_to_idx, self.idx_to_label, self.predicate_to_idx, self.idx_to_predicate = self.load_mappings(index_file)
        self.ht2vector = self.prepare_pred_prior()
        self.pred_default_vector = np.full(self.pred_size, self.default)

    def prepare_ht2context_count(self, ht2context_count):
        ht2ct_keys = {}
        ht2ct_values = {}
        for ht, ct2c in ht2context_count.items():
            ht2ct_keys[ht] = list(ct2c.keys())
            ht2ct_values[ht] = np.asarray(list(ct2c.values()))
        return ht2ct_keys, ht2ct_values

    def prepare_pred_prior(self):
        ht2vector = {}  # {head, tail: vector}
        for ht, r2p in self.rel_prob_given_head_tail.items():
            v = np.full(self.pred_size, self.default)
            for rel, prob in r2p.items():
                v[self.predicate_to_idx.get(rel, 0)] = prob  # 0 in case the verb is not found, should not have any
            ht2vector[ht] = v
        return ht2vector

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
                input()
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

    def predict(self, input_context, verbose=False, use_best_rel_only=False):
        # create all pairs
        all_pairs = self.create_all_pairs(input_context)
        # new: get all context matched for these pairs
        matched_contexts = self.get_matched_context(all_pairs)
        if verbose:
            print("#matched_contexts: {}".format(len(matched_contexts)))
        # compute the scores
        start = time.time()
        ct2score = self.compute_contexts_similarity(input_context, matched_contexts)
        if verbose:
            print("Time: {}".format(time.time() - start))
        # get pair2scores
        pair2scores = self.get_context_sim_array(all_pairs, ct2score)

        # compute accumulated weighted-count for each pair
        pair2count = self.compute_accumulated_weighted_count(pair2scores, verbose=verbose)

        # compute prob and sort pairs
        pair2prob = self.compute_probability_for_dict(pair2count)
        if verbose:
            sorted_pairs = sorted(pair2prob.items(), key=lambda x: x[1], reverse=True)
            print("Sorted pairs")
            pprint(sorted_pairs)
        if use_best_rel_only:
            return self.get_triples_using_best_predicate(pair2prob, verbose=verbose)
        else:
            ht2prediction = {}
            for ht, prob in pair2prob.items():
                v = self.ht2vector.get(ht, self.pred_default_vector)
                v2 = prob * v
                ht2prediction[ht] = v2
        return ht2prediction

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


def evaluate_image(predictor, info, ks):
    context = info[0]
    gt_triples = info[1]
    pred_triples = predictor.predict(context, use_best_rel_only=True)
    # evaluate
    pt = [a[0] for a in pred_triples]
    num_gt = len(gt_triples)
    k2scores = {}
    for k in ks:
        if k > len(pt):
            break
        tmp = pt[:k]
        correct = len(set(tmp).intersection(gt_triples))
        prec = correct/len(tmp)
        recall = correct/num_gt
        if prec + recall > 0:
            f1 = 2 * (prec * recall) / (prec + recall)
        else:
            f1 = 0.0
        k2scores[k] = (prec, recall, f1)
    return k2scores


def evaluate_test():
    prior_file = 'data/KG/preprocessed/alias_filtered_codebase_h5/train_triples_context_essential_info.pkl'
    index_file = 'data/VisualGnome/preprocessed_codebase/VG-SGG-dicts.json'
    # initialize the predictor
    predictor = ContextBasedPredicatePredictor(prior_file=prior_file, index_file=index_file)
    img2info = load_img2triples(setname='test')
    print("Evaluating")
    ks = [1, 5, 10, 20]
    imgs = []
    precs = {k: [] for k in ks}
    recalls = {k: [] for k in ks}
    f1s = {k: [] for k in ks}
    count = 0
    for img, info in tqdm(img2info.items()):
        count += 1
        scores = evaluate_image(predictor, info, ks)
        imgs.append(img)
        for k in ks:
            if k not in scores:
                tmp = ['-', '-', '-']
            else:
                tmp = scores[k]
            precs[k].append(tmp[0])
            recalls[k].append(tmp[1])
            f1s[k].append(tmp[2])
        if count % 10 == 0:
            tmp = {"image": imgs}
            for k in ks:
                tmp["Prec@{}".format(k)] = precs[k]
            for k in ks:
                tmp["Recall@{}".format(k)] = recalls[k]
            for k in ks:
                tmp["F1@{}".format(k)] = f1s[k]

            df = pd.DataFrame(tmp)
            df.to_csv("data/KG/preprocessed/alias_filtered_codebase_h5/eval_v3.csv")

if __name__ == "__main__":
    prior_file = 'datasets/kg/train_triples_context_essential_info.pkl'
    index_file = 'datasets/vg/VG-SGG-dicts.json'
    # initialize the predictor
    predictor = ContextBasedPredicatePredictor(prior_file=prior_file, index_file=index_file)

    # example, input should be a list of unique objects (not index)
    icontext = ['hair', 'plate', 'arm', 'shirt', 'man', 'roof', 'sign', 'food', 'boy', 'logo', 'pole', 'handle', 'hand']
    # get the predicate distributions prediction for all the pairs (allow same-object pair, e.g., (car, car))
    ht2vec = predictor.predict(icontext)  # {(head, tail): vector}
    # to get the vector of a pair (head, tail), access ht2vec[(head, tail)]
    vec = ht2vec[('man', 'hair')]
    print(vec)

    # printing vectors for all pairs
    # for a in icontext:
    #     for b in icontext:
    #         if (a,b) in ht2vec:
    #             print("{}\t{}\t{}".format(a, b, ht2vec[(a,b)]))

    sorted_triples = predictor.predict(icontext, use_best_rel_only=True)
    pprint(sorted_triples)

    # test for real images
    # evaluate_test()