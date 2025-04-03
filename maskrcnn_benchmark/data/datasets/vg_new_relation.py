import os
import torch
import time
import pickle
from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.boxes import box_filter
from maskrcnn_benchmark.utils.boxes_rel import filter_new_rel_nn, filter_new_rel_overlapping
from visualization.vis_utils import *
from maskrcnn_benchmark.utils.miscellaneous import bbox_iom, filter_duplicate_rel, filter_new_triples, \
                                                loadf, bbox_overlaps, count_triples
from maskrcnn_benchmark.data.datasets.context_based_KB import ContextKB
from maskrcnn_benchmark.data.datasets.visual_genome import load_info, load_image_filenames, load_graphs


BOX_SCALE = 1024  # Scale at which we have the boxes
ImageFile.LOAD_TRUNCATED_IMAGES = True

class NewRelationData(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000, 
                filter_duplicate_rels=True, filter_non_overlap=False, flip_aug=False, 
                custom_eval=False, custom_path='', relation_cfg=None, logits_cfg=None):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """

        assert split in {'train'}
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.custom_eval = custom_eval

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(dict_file) # 151, 51 containing __background__
        self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}
        # filter_out relations in extra
        freq_rels = [0, 31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43, 40, 49, 41, 23,
                     38, 6, 7, 33, 11, 46, 16, 25, 47, 19, 5, 9, 35, 24, 10, 4, 14, 
                     13, 12, 36, 44, 42, 32, 2, 28, 26, 45, 3, 17, 18, 34, 27, 37, 39, 15]
        # freq_pred_name = [self.ind_to_predicates[l] for l in freq_rels]
        self.freq_rels = np.array(freq_rels, dtype=np.int64)

        # Step 1: load / save GT data
        self.logits_cfg = logits_cfg
        if logits_cfg is not None and os.path.exists(logits_cfg.SPECIFIED_DATA_FILE):
            self.data = loadf(logits_cfg.SPECIFIED_DATA_FILE)
            print('Loaded supervision data from ', logits_cfg.SPECIFIED_DATA_FILE)
            self.img_info = [{"width":x["width"], "height": x["height"], "image_id": x["image_id"]} for x in self.data]
            self.filenames = [x["img_path"] for x in self.data]
            self.gt_boxes = [x["boxes"] for x in self.data]
            self.gt_classes = [x["labels"] for x in self.data]
            self.relationships = [x["triples"] for x in self.data]
        else:
            self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
                self.roidb_file, self.split, num_im, num_val_im=num_val_im,
                filter_empty_rels=filter_empty_rels,
                filter_non_overlap=self.filter_non_overlap,
            )
            self.filenames, self.img_info = load_image_filenames(img_dir, image_file) # length equals to split_mask
            self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
            self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]

            if self.filter_duplicate_rels:
                assert self.split == 'train'
                relation_filtered = []
                for i in range(len(self.filenames)):
                    relation_filtered.append(filter_duplicate_rel(self.relationships[i].copy()))
                self.relationships = relation_filtered
            if self.split == 'train':
                self.data = self.save_sup_data(logits_cfg.SPECIFIED_DATA_FILE)
        print(f"original triple: {count_triples(self.relationships)}")
        self.relation_cfg = relation_cfg
        # Step 2: # run "relation_relabel.py" script to get inference results of pretrained models.
        if self.logits_cfg.STAGE == "get-logits":
            return
        if self.relation_cfg.N_SPO_RULE:
            statistics_dic = self.get_statistics()
            self.freq_dist = statistics_dic["freq_dist"]
        # Load relation label of Internal Transfer
        if self.relation_cfg.COMB_INTERNAL_TRANS:
            self.relationships = self.load_internal_trans_label()
            print("Loaded internal transfer labels.")
        # Step 3: combine pretrained model's results by prior prob.
        os.makedirs(self.relation_cfg.OUTPUT_DIR, exist_ok=True)
        if self.logits_cfg.ENABLE:
            if os.path.exists(self.logits_cfg.LOGITS_COMB_PRIOR):
                start = time.time()
                self.data = loadf(self.logits_cfg.LOGITS_COMB_PRIOR)
                print('Time taken to load file', self.logits_cfg.LOGITS_COMB_PRIOR, time.time()-start)
            else:
                start = time.time()
                if self.logits_cfg.COMB_METHOD == 'mult':
                    self.multiply_prediction_by_prior()
                else: # 'sum'
                    self.sum_prediction_and_prior()
                print('Time taken to generate file', self.logits_cfg.LOGITS_COMB_PRIOR, time.time()-start)
        # Step 4: deduct new relations with / without logit-prob.
        if (self.relation_cfg is not None) and self.relation_cfg.ENABLE:
            top_k_freq = self.relation_cfg.TOP_K_FREQ
            self.freq_rels = self.freq_rels[:top_k_freq + 1]
            print(f"top_k_freq: {top_k_freq}")

            REL_CACHE_PREFIX = self.relation_cfg.REL_CACHE_PREFIX
            if self.relation_cfg.USE_KG:
                print("USE_CONTEXT: {}".format(self.relation_cfg.USE_CONTEXT))
                print("USE_DISTILLATION: {}".format(self.relation_cfg.USE_DISTILLATION))
                if self.relation_cfg.USE_CONTEXT:
                    REL_CACHE_PREFIX = REL_CACHE_PREFIX + "_C"
                else:
                    REL_CACHE_PREFIX = REL_CACHE_PREFIX + "_nC"
                if self.relation_cfg.USE_DISTILLATION:
                    REL_CACHE_PREFIX = REL_CACHE_PREFIX + "D"
                else:
                    REL_CACHE_PREFIX = REL_CACHE_PREFIX + "nD"
            relation_cache_file = os.path.join(self.relation_cfg.OUTPUT_DIR,
                                               f'relation_cache_{REL_CACHE_PREFIX}.npy')

            self.cache_prefix = REL_CACHE_PREFIX
            if os.path.exists(relation_cache_file):
                print("Loading relation cache from file:", relation_cache_file)
                self.new_triple = np.load(relation_cache_file, allow_pickle=True)[()]
            else:
                self.new_triple = self.load_new_relation()
                np.save(relation_cache_file, self.new_triple, allow_pickle=True)
                print("Saved relation cache file:", relation_cache_file)
            print("Total number of new triples:", self.total_num_new_triple)
            # save gt_rel and new_rel into dict
            # self.save_rel_dict(REL_CACHE_PREFIX)


    def __getitem__(self, index):
        if self.custom_eval:
            img = Image.open(self.custom_files[index]).convert("RGB")
            target = torch.LongTensor([-1])
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, index

        try:
            img = Image.open(self.filenames[index]).convert("RGB")
        except:
            raise RuntimeError("Can't open image", self.filenames[index])
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']), ' ', str(self.img_info[index]['height']), ' ', '='*20)

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')

        target = self.get_groundtruth(index, flip_img=flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index


    def get_statistics(self):
        fg_matrix, bg_matrix = self.get_VG_statistics()
        eps = 1e-5
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / (fg_matrix.sum(2)[:, :, None] + eps) + eps)
        freq_dist = fg_matrix / (fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'freq_dist': torch.from_numpy(freq_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width':int(img.width), 'height':int(img.height)})

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]


    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index].copy() / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:,2]
            new_xmax = w - box[:,0]
            box[:,0] = new_xmin
            box[:,2] = new_xmax
        assert len(box.shape) == 2
        target = BoxList(box, (w, h), 'xyxy') # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        # target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy() # (num_rel, 3)

        # add new relation for training
        if self.split!='test' and (self.relation_cfg is not None) and self.relation_cfg.ENABLE:
            new_rel = self.new_triple[index]
            if new_rel is not None:
                relation = np.vstack((relation, new_rel)) # (num_rel, 3)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i,0]), int(relation[i,1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
            else:
                relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation)) # for evaluation
        else:
            target = target.clip_to_image(remove_empty=False)
        # target.add_field("image_path", self.filenames[index])
        assert np.amax(relation[:, :2]) < num_box, "Pair index out of bound"
        assert np.amax(relation[:, :2]) < target.get_field("labels").shape[0], "Pair index out of bound2"
        curr_d = self.data[index]
        curr_d["relation_map"] = relation_map
        target.add_field("train_data", curr_d)
        return target

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.filenames)
    

    def prepare_new_rel(self, index, labels, bboxes):
        """
        Options: ovlp/nn/none
        none:
            Add new relation to every instance of pairs. eg:
            new triple: car on street
            object list: car, car, street
            return: [[car1_index, pred_id, street_index], [car2_index, pred_id, street_index]]
        nn:
            For a subject, add new relation to the nearest object instance, 
            if there exists multiple instances of the same object class.
        ovlp:
            Add new relation to pairs whose object overlaps with subject.
        """
        image_id = self.img_info[index]['image_id']
        if str(image_id) not in self.new_rel_dict.keys():
            return None
        new_rel = self.new_rel_dict[str(image_id)]

        triple = []
        for tmp in new_rel:
            (o0, r, o1) = tmp[0]
            s = self.label_to_idx[o0]
            r0 = self.predicate_to_idx[r]
            o = self.label_to_idx[o1]
            s_idxs = (labels == s).nonzero(as_tuple=True)[0]
            o_idxs = (labels == o).nonzero(as_tuple=True)[0]
            if len(s_idxs) > 0 and len(o_idxs) > 0:
                for s_i in s_idxs:
                    if self.relation_cfg.NEW_REL_FILTER in ["", "none"]:
                        for o_i in o_idxs:
                            if s_i != o_i:
                                triple.append([int(s_i), int(o_i), int(r0)])
                    elif self.relation_cfg.NEW_REL_FILTER == "nn":
                        s_bbox = bboxes[s_i]
                        o_bboxes = bboxes[o_idxs]
                        o_ii = filter_new_rel_nn(s_bbox, o_bboxes)
                        o_i = o_idxs[o_ii]
                        if s_i != o_i:
                            triple.append([int(s_i), int(o_i), int(r0)])
                    elif self.relation_cfg.NEW_REL_FILTER == "ovlp":
                        s_bbox = bboxes[s_i].reshape(-1, 4)
                        o_bboxes = bboxes[o_idxs].reshape(-1, 4)
                        o_ii = filter_new_rel_overlapping(s_bbox, o_bboxes)
                        o_i_keep = o_idxs[o_ii]
                        for o_i in o_i_keep:
                            if s_i != o_i:
                                triple.append([int(s_i), int(o_i), int(r0)])
                    else:
                        print("<visual_genome.py> Not supported config ({}).".format(self.relation_cfg.NEW_REL_FILTER))
        if len(triple) > 0:
            return np.array(triple, dtype=np.int64)
        else:
            return None


    def prepare_new_rel_iou(self, index, labels, bboxes, text):
        if self.relation_cfg.USE_CONTEXT:
            image_id = self.img_info[index]['image_id']
            cluster = self.context_prior.img2cluster[image_id]
            if self.relation_cfg.SAMPLE_REL_FROM_PROB:
                self.pair_score = self.context_prior.cluster2pair_prob.get(cluster, None)
                self.top_rel_of_pair = self.context_prior.cluster2pred_prob.get(cluster, None)
            else:
                self.pair_score = self.context_prior.cluster2pair_prob.get(str(cluster), None)
                self.top_rel_of_pair = self.context_prior.cluster2pred_prob.get(str(cluster), None)
            if self.pair_score is None or self.top_rel_of_pair is None:
                return None
        triple = []
        # a "v, o" pair should owned by only one subject
        vo_s_dict = defaultdict(list)
        for sbj_idx, lb in enumerate(labels):
            sbj_str = self.ind_to_classes[lb]
            # retrieve object scores and compute iou scores
            if sbj_str not in self.pair_score.keys():
                continue
            if isinstance(self.top_rel_of_pair, dict) and sbj_str not in self.top_rel_of_pair.keys():
                continue
            obj_dict = self.pair_score[sbj_str]
            rank_score = []
            for obj, obj_score in obj_dict.items():
                # find the indexes of obj instances
                obj_id = self.label_to_idx[obj]
                obj_idxs = (labels == obj_id).nonzero(as_tuple=True)[0]
                # exclude self
                if sbj_idx in obj_idxs:
                    obj_idxs = obj_idxs[obj_idxs != sbj_idx]
                if obj_idxs.shape[0] == 0:
                    continue
                # compute iom
                o_bboxes = bboxes[obj_idxs].reshape(-1, 4)
                iom = bbox_iom(bboxes[sbj_idx].reshape(-1, 4), o_bboxes)
                scores = iom[0] * obj_score
                for o_idx, s, iou_s in zip(obj_idxs, scores, iom[0]):
                    if s > 0: # overlapping filtering
                        rank_score.append((s, o_idx))
            if rank_score == []:
                continue
            obj_max_score = max(rank_score)
            id = labels[obj_max_score[1]]
            if self.relation_cfg.SAMPLE_REL_FROM_PROB:
                p_vec = self.top_rel_of_pair[:, labels[sbj_idx], id].numpy()
                p_vec = np.asarray(p_vec).astype('float64')
                if sum(p_vec) == 0:
                    continue
                p_vec /= sum(p_vec)
                try:
                    vb = np.random.choice(len(self.ind_to_predicates), 1, p=p_vec)[0]
                except:
                    print(p_vec)
                o = int(obj_max_score[1])
                sr = obj_max_score[0]
                vo_s_dict[(vb, o)].append((sr, sbj_idx))
            else:
                obj_str = self.ind_to_classes[id]
                if obj_str in self.top_rel_of_pair[sbj_str]:
                    rel = self.top_rel_of_pair[sbj_str][obj_str]
                    # triple.append([sbj_idx, int(obj_max_score[1]), int(self.predicate_to_idx[rel])])
                    vb = int(self.predicate_to_idx[rel])
                    o = int(obj_max_score[1])
                    sr = obj_max_score[0]
                    vo_s_dict[(vb, o)].append((sr, sbj_idx))

        # if len(triple) > 0:
        if len(vo_s_dict.keys()) > 0:
            for (vb, o), v in vo_s_dict.items():
                if len(v) > 1:
                    sbj_max_score = max(v)
                    triple.append([sbj_max_score[1], o, vb])
                else:
                    triple.append([v[0][1], o, vb])
            return np.array(triple, dtype=np.int64)
        else:
            return None


    def filter_out_freq_rels(self, relations):
        rels = relations[:, 2]
        in_mask = (rels.reshape(-1, 1) == self.freq_rels.reshape(-1, 1).T).any(-1)
        return relations[~in_mask]

    def filter_out_nonexisting_triples(self, relations, labels):
        if relations is None or relations.shape[0] == 0:
            return None
        sbj_ids = labels[relations[:, 0]]
        obj_ids = labels[relations[:, 1]]
        rels = relations[:, 2]
        freqs = self.freq_dist[sbj_ids, obj_ids, rels]
        # print(np.round(freqs.numpy(), 3))
        keep = freqs > 0
        if relations.shape[0] == 1:
            if keep:
                return relations
            else:
                return None
        return relations[keep]

    def prepare_new_rel_pretrain(self, index):
        image_id = self.img_info[index]['image_id']
        curr_dict = self.data[image_id].copy()
        pair_idx = curr_dict["rel_pair_idxs"] # [num_pairs, 2]
        num_gt_rel = curr_dict["triples"].shape[0]
        bg_pair_idx = pair_idx[num_gt_rel:]
        if self.relation_cfg.USE_KG:
            try:
                bg_post_prob = curr_dict["rel_prob_mult_prior"][num_gt_rel:]
            except:
                bg_post_prob = curr_dict["rel_post_prob"][num_gt_rel:]
        else:
            bg_post_prob = curr_dict["rel_prob"][num_gt_rel:]
        assert bg_pair_idx.shape[0] == bg_post_prob.shape[0]

        # filter out pairs with all zero prob
        bg_nonzero_idx = np.sum(bg_post_prob, axis=-1) > 0
        bg_pair_idx_nonzero = bg_pair_idx[bg_nonzero_idx]
        bg_post_prob_nonzero = bg_post_prob[bg_nonzero_idx]
        top_rel_score = np.max(bg_post_prob_nonzero[:, 1:], axis=-1)
        top_rel = np.argmax(bg_post_prob_nonzero[:, 1:], axis=-1) + 1
        # filter non-overlapping pairs
        boxes = curr_dict["boxes"]
        sbj_boxes = boxes[bg_pair_idx_nonzero[:, 0]]
        obj_boxes = boxes[bg_pair_idx_nonzero[:, 1]]
        iou = bbox_overlaps(sbj_boxes, obj_boxes).diagonal()
        keep_idx = iou > 0
        # keep_idx = (iou > 0) * (iou < 0.9)
        keep_pair_idx = bg_pair_idx_nonzero[keep_idx]
        keep_rel_id = top_rel[keep_idx]
        if not self.relation_cfg.RANK_PAIRS or keep_rel_id.shape[0]==1:
            triple = np.column_stack((keep_pair_idx, keep_rel_id))
        else:
            # sort pairs base on max score
            keep_top_rel_score = top_rel_score[keep_idx]
            _, sorting_idx = torch.sort(torch.from_numpy(keep_top_rel_score).view(-1), dim=0, descending=True)
            keep_percent = 0.5
            keep_num = int(sorting_idx.shape[0] * keep_percent)
            rel_pair_idx = keep_pair_idx[sorting_idx][:keep_num]
            rel_class_score = keep_top_rel_score[sorting_idx][:keep_num]
            rel_labels = keep_rel_id[sorting_idx][:keep_num]
            triple = np.column_stack((rel_pair_idx, rel_labels))
        return triple
    
    def transfer_gt_rel_pretrain(self, index):
        image_id = self.img_info[index]['image_id']
        curr_dict = self.data[image_id]
        pair_idx = curr_dict["rel_pair_idxs"] # [num_pairs, 2]
        num_gt_rel = curr_dict["triples"].shape[0]
        fg_pair_idx = pair_idx[:num_gt_rel]
        if self.relation_cfg.USE_KG:
            try:
                fg_post_prob = curr_dict["rel_prob_mult_prior"][:num_gt_rel]
            except:
                fg_post_prob = curr_dict["rel_post_prob"][:num_gt_rel]
        else:
            fg_post_prob = curr_dict["rel_prob"][:num_gt_rel]
        assert fg_pair_idx.shape[0] == fg_post_prob.shape[0]

        top_rel_score = np.max(fg_post_prob[:, 1:], axis=-1)
        top_rel = np.argmax(fg_post_prob[:, 1:], axis=-1) + 1

        triple = np.column_stack((fg_pair_idx, top_rel))
        return triple


    def init_kg_prior(self):
        if self.relation_cfg.USE_CONTEXT:
            if self.relation_cfg.USE_DISTILLATION:
                c2k_file = os.path.join(self.relation_cfg.KG_DATA_ROOT, "context-based", "distillation",
                                        self.relation_cfg.CLUSTER_TO_KNOWLEDGE)
            else:
                c2k_file = os.path.join(self.relation_cfg.KG_DATA_ROOT, "context-based", "no-distillation",
                                        self.relation_cfg.CLUSTER_TO_KNOWLEDGE)
            self.context_prior = ContextKB(kmeans_model_path=self.relation_cfg.KMEANS_MODEL, 
                                            kmeans_pred_file=self.relation_cfg.KMEANS_PRED_FILE, 
                                            cluster_to_knowledge_file=c2k_file)
        else:
            if self.relation_cfg.USE_DISTILLATION:
                pair_score_file = os.path.join(self.relation_cfg.KG_DATA_ROOT, "no-context", "distillation",
                                               self.relation_cfg.PAIR_SCORE)
                top_rel_of_pair_file = os.path.join(self.relation_cfg.KG_DATA_ROOT, "no-context", "distillation",
                                               self.relation_cfg.TOP_REL_OF_PAIR)
                triple_prob_file = os.path.join(self.relation_cfg.KG_DATA_ROOT, "no-context", "distillation",
                                               self.relation_cfg.NO_CONTEXT_TRIPLE_PROB)
            else:
                pair_score_file = os.path.join(self.relation_cfg.KG_DATA_ROOT, "no-context", "no-distillation",
                                               self.relation_cfg.PAIR_SCORE)
                top_rel_of_pair_file = os.path.join(self.relation_cfg.KG_DATA_ROOT, "no-context", "no-distillation",
                                               self.relation_cfg.TOP_REL_OF_PAIR)
                triple_prob_file = os.path.join(self.relation_cfg.KG_DATA_ROOT, "no-context", "no-distillation",
                                               self.relation_cfg.NO_CONTEXT_TRIPLE_PROB)
            if self.relation_cfg.NEW_REL_FILTER in ["pretrain", "pretrain_internal"]:
                self.triple_prob = loadf(triple_prob_file)
            else:
                self.new_rel_dict = loadf(self.relation_cfg.NEW_REL_FILE)
                self.pair_score = loadf(pair_score_file)
                self.top_rel_of_pair = loadf(top_rel_of_pair_file)


    def load_new_relation(self):
        """Compute new relation triples from KG prior
        """
        assert len(self.filenames) == len(self.gt_classes)
        if not (hasattr(self, "context_prior") or hasattr(self, "pair_score")) \
            and self.relation_cfg.USE_KG:
            self.init_kg_prior()

        info = loadf(self.dict_file)
        self.label_to_idx = info['label_to_idx'] # without adding __background__
        self.predicate_to_idx = info['predicate_to_idx'] # without adding __background__
        print("Start computing new relation triples...")

        # iterate through the whole split
        new_triple = []
        self.total_num_new_triple = 0
        self.empty_new_triple_img_cnt = 0
        for index in tqdm(range(len(self.filenames))):
            filename = self.filenames[index]
            labels = torch.from_numpy(self.gt_classes[index])
            relation = self.relationships[index].copy()

            image_id = self.img_info[index]['image_id']
            names = [self.ind_to_classes[ii] for ii in self.gt_classes[index]]

            img_info = self.get_img_info(index)
            w, h = img_info['width'], img_info['height']
            # important: recover original box from BOX_SCALE
            box = self.gt_boxes[index].copy() / BOX_SCALE * max(w, h)
            box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

            if self.relation_cfg.NEW_REL_FILTER == "iou":
                new_rel = self.prepare_new_rel_iou(index, labels, box, names)
            elif self.relation_cfg.NEW_REL_FILTER == "pretrain":
                new_rel = self.prepare_new_rel_pretrain(index)
            elif self.relation_cfg.NEW_REL_FILTER == "pretrain_internal":
                new_rel = self.transfer_gt_rel_pretrain(index)
            else:
                new_rel = self.prepare_new_rel(index, labels, box)
            # filter out frequent relations
            if self.relation_cfg.FILTER_FREQ_REL:
                new_rel = self.filter_out_freq_rels(new_rel)
            # number of specific triple should > 0 in trainset
            if self.relation_cfg.N_SPO_RULE:
                new_rel = self.filter_out_nonexisting_triples(new_rel, labels)
            # filter duplicate relations already exist
            if self.relation_cfg.NEW_REL_FILTER != "pretrain_internal":
                relation, new_rel = filter_new_triples(relation, new_rel)
            if new_rel is None:
                self.empty_new_triple_img_cnt += 1
                new_triple.append(None)
            else:
                new_triple.append(new_rel)
                self.total_num_new_triple += new_rel.shape[0]
                # visualization
                vis_out_dir = self.relation_cfg.VIS_OUT_DIR
                if vis_out_dir != "" and index < 100:
                    if not os.path.exists(vis_out_dir):
                        os.makedirs(vis_out_dir)
                    pic = draw_object(filename, box, names)
                    pic = draw_relation(pic, box, self.ind_to_predicates, relation, names, (255,255,100))# yellow
                    pic = draw_relation(pic, box, self.ind_to_predicates, new_rel, names, (255,100,255)) # purple
                    pic.save(os.path.join(vis_out_dir, str(self.img_info[index]['image_id'])+'.png'))
                    pic.close()
        print("{} images have no new relation created.".format(self.empty_new_triple_img_cnt))
        print("There are {} new triples on average are added for an image.".format(
              self.total_num_new_triple / (len(self.filenames) - self.empty_new_triple_img_cnt)))
        return new_triple


    def multiply_prediction_by_prior(self,):
        assert os.path.exists(self.logits_cfg.DATA_WITH_LOGITS), "File not found."
        self.data = loadf(self.logits_cfg.DATA_WITH_LOGITS)
        if not self.relation_cfg.USE_KG:
            print("NOT using kg prior.")
            return
        self.init_kg_prior()
        dic = {}
        for k, v in self.data.items():
            labels = v["labels"]
            pair_idx = v["rel_pair_idxs"] # [num_pairs, 2]
            pair_ids = labels[pair_idx]
            rel_prob = v["rel_prob"] # [num_pairs, 51]
            if self.relation_cfg.USE_CONTEXT:
                cluster = self.context_prior.img2cluster.get(k, -1)
                assert cluster >= 0, "Image is not assigned to any cluster!"
                prior_dist = self.context_prior.cluster2info[cluster].clone() #torch[51, 151, 151]
            else:
                prior_dist = self.triple_prob.clone()
            pair_prior = prior_dist[:, pair_ids[:, 0], pair_ids[:, 1]].transpose(1, 0)
            pair_prior = torch.nn.functional.normalize(pair_prior, p=1, dim=-1) # for ranking
            # top_k = pair_prior.topk(k=5, dim=1)
            # print(np.round(top_k.values.numpy(), 3))
            v["pair_prior"] = pair_prior.numpy()
            post_prob = (torch.from_numpy(rel_prob) * pair_prior * 10)
            v["rel_prob_mult_prior"] = post_prob.numpy()
            dic[k] = v
        save_path = self.logits_cfg.LOGITS_COMB_PRIOR
        # pickle.dump(dic, open(save_path, "wb"))
        print("Saved", save_path)


    def sum_prediction_and_prior(self,):
        assert os.path.exists(self.logits_cfg.DATA_WITH_LOGITS), "File not found."
        self.data = loadf(self.logits_cfg.DATA_WITH_LOGITS)
        if not self.relation_cfg.USE_KG:
            print("NOT using kg prior.")
            return
        self.init_kg_prior()
        dic = {}
        for k, v in self.data.items():
            labels = v["labels"]
            pair_idx = v["rel_pair_idxs"] # [num_pairs, 2]
            pair_ids = labels[pair_idx]
            rel_logits = v["logits"] # [num_pairs, 51]
            if self.relation_cfg.USE_CONTEXT:
                cluster = self.context_prior.img2cluster.get(k, -1)
                assert cluster >= 0, "Image is not assigned to any cluster!"
                prior_dist = self.context_prior.cluster2info[cluster] #torch[51, 151, 151]
            else:
                prior_dist = self.triple_prob
            pair_prior = prior_dist[:, pair_ids[:, 0], pair_ids[:, 1]].transpose(1, 0)
            v["pair_prior"] = pair_prior.numpy()
            pair_prior = torch.nn.functional.normalize(pair_prior, p=1, dim=-1) # for ranking
            # top_k = pair_prior.topk(k=5, dim=1)
            # print(np.round(top_k.values.numpy(), 3))
            eps = 1e-5
            pair_prior = torch.log(pair_prior / (pair_prior + eps) + eps)
            post_prob = torch.from_numpy(rel_logits) + pair_prior
            post_prob = post_prob.softmax(dim=-1)
            v["rel_post_prob"] = post_prob.numpy()
            dic[k] = v
        save_path = self.logits_cfg.LOGITS_COMB_PRIOR
        pickle.dump(dic, open(save_path, "wb"))
        print("Saved", save_path)

    def load_internal_trans_label(self):
        assert os.path.exists(self.relation_cfg.INTERNAL_FILE), "Internal transfer file not found."
        intrans = loadf(self.relation_cfg.INTERNAL_FILE)
        intrans_list = []
        for filename in self.filenames:
            rel = intrans[filename]
            intrans_list.append(rel)
        return intrans_list


    def save_rel_dict(self, REL_CACHE_PREFIX):
        rel_to_save = {}
        for index in range(len(self.filenames)):
            relation = self.relationships[index]
            new_rel = self.new_triple[index]
            image_id = self.img_info[index]['image_id']
            assert len(self.filenames) == len(self.gt_classes)
            names = [self.ind_to_classes[ii] for ii in self.gt_classes[index]]
            
            rel_to_save[image_id] = {}
            rel_to_save[image_id]["gt_rel"] = rel_idx_to_name(relation, names, self.ind_to_predicates)
            if new_rel is None:
                rel_to_save[image_id]["new_rel"] = None
            else:
                rel_to_save[image_id]["new_rel"] = rel_idx_to_name(new_rel, names, self.ind_to_predicates)
        save_name = "output/relation_{}_{}.npy".format(REL_CACHE_PREFIX, self.split)
        np.save(save_name, rel_to_save, allow_pickle=True)
        print("Saved relation (gt and new) data into {}".format(save_name))


    def save_sup_data(self, specified_data_file):
        assert self.split=='train'
        data = []
        for filename, gt_class, gt_box, img_info, triples, attri in \
            zip(self.filenames, self.gt_classes, self.gt_boxes, self.img_info, self.relationships, self.gt_attributes):
            curr_dict = {}
            curr_dict['image_id'] = img_info['image_id']
            curr_dict['width'] = img_info['width']
            curr_dict['height'] = img_info['height']
            curr_dict['img_path'] = filename
            curr_dict['boxes'] = gt_box
            curr_dict['labels'] = gt_class
            curr_dict['triples'] = triples
            curr_dict['attri'] = attri
            data.append(curr_dict)
        pickle.dump(data, open(specified_data_file, "wb"))
        return data


    def get_VG_statistics(self):
        assert self.split=="train", "Statistics must be generated from the TRAIN split!"
        num_obj_classes = len(self.ind_to_classes)
        num_rel_classes = len(self.ind_to_predicates)
        fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
        bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)
        print("Computing VG statistics...")
        for ex_ind in tqdm(range(len(self.filenames))):
            gt_classes = self.gt_classes[ex_ind].copy()
            gt_relations = self.relationships[ex_ind].copy()
            gt_boxes = self.gt_boxes[ex_ind].copy()

            # For the foreground, we'll just look at everything
            o1o2 = gt_classes[gt_relations[:, :2]]
            for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
                fg_matrix[o1, o2, gtr] += 1
            # For the background, get all of the things that overlap.
            o1o2_total = gt_classes[np.array(
                box_filter(gt_boxes, must_overlap=True), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1

        return fg_matrix, bg_matrix


    def get_pred_frequency(self):
        assert self.split=="train", "Statistics must be generated from the TRAIN split!"
        pred_vec = np.zeros((len(self.ind_to_predicates)))
        for i in range(len(self.filenames)):
            gt_relations = self.relationships[i].copy()
            for gtr in gt_relations[:,2]:
                pred_vec[gtr] += 1
        freq_map = [(pred_vec[i], i) for i in range(len(pred_vec))]
        freq_map = sorted(freq_map)
        freq_pred = [freq_map[i][1] for i in range(len(freq_map))]
        sorted_freq_pred = [0] + freq_pred[1:][::-1]
        return sorted_freq_pred



if __name__=='__main__':
    import argparse
    from yacs.config import CfgNode as CN
    from maskrcnn_benchmark.utils.miscellaneous import save_config

    parser = argparse.ArgumentParser(description="Relation Deduction")
    parser.add_argument(
        "--config-file",
        default="configs/relation_deduction/relabel_new_relation_pretrain.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    relation_cfg = cfg.NEW_RELATION
    logits_cfg = cfg.USE_LOGITS
    data = NewRelationData(split='train', img_dir=cfg.IMG_DIR, roidb_file=cfg.ROIDB_FILE,
                        dict_file=cfg.DICT_FILE, image_file=cfg.IMAGE_FILE, num_val_im=5000,
                        filter_non_overlap = False,
                        relation_cfg=relation_cfg, logits_cfg=logits_cfg)
    if hasattr(data, "cache_prefix"):
        save_cfg_name = f"config_{data.cache_prefix}.yml"
    else:
        save_cfg_name = "config.yml"
    save_config(cfg, os.path.join(cfg.NEW_RELATION.OUTPUT_DIR, save_cfg_name))
    print(f"Saved results in {cfg.NEW_RELATION.OUTPUT_DIR}")