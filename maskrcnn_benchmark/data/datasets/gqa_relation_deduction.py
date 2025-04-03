import os
import pickle
import torch
import time
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import copy
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.miscellaneous import (loadf, count_triples, bbox_iom, filter_new_triples)
from maskrcnn_benchmark.utils.miscellaneous import bbox_overlaps as bbox_overlaps_misc

from visualization.vis_utils import *
from maskrcnn_benchmark.data.datasets.context_based_KB import ContextKB

BOX_SCALE = 1024  # Scale at which we have the boxes



class GQARelabeling(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, dict_file, train_file, test_file, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=False, flip_aug=False,
                relation_cfg=None, logits_cfg=None):
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

        assert split in {'train'}
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

        # Step 1: load / save GT data
        if self.split == 'train':
            self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
                self.train_file, self.split)
        else:
            self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
                self.test_file, self.split)
        print(f"original triple: {count_triples(self.relationships)}")

        # filter_out relations in extra
        self.freq_rels = np.arange(len(self.ind_to_predicates))
        
        self.relation_cfg = relation_cfg
        self.logits_cfg = logits_cfg
        # Step 2: # run "relation_relabel.py" script to get inference results of pretrained models.
        if self.logits_cfg.STAGE == "get-logits":
            return
        if self.relation_cfg.N_SPO_RULE:
            statistics_dic = self.get_statistics()
            self.freq_dist = statistics_dic["freq_dist"]

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
                    raise ValueError("COMB_METHOD should be 'mult'.")
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
        # use the following code only when run "tools/relation_relabel.py"
        if self.split=="train":
            curr_d = {
                'image_id': int(self.filenames[index].split(".")[0]),
                'width': self.img_info[index]['width'],
                'height': self.img_info[index]['height'],
                'img_path': os.path.join(self.img_dir, self.filenames[index]),
                'boxes': self.gt_boxes[index],
                'labels': self.gt_classes[index],
                'triples': self.relationships[index],
                'relation_map': relation_map
            }
            target.add_field("train_data", curr_d)
        return target

    def __len__(self):
        return len(self.filenames)
    

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
        image_id = int(self.filenames[index].split("/")[-1].split(".")[0])
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
        bg_post_prob_nonzero = bg_post_prob[bg_nonzero_idx].astype('float64')
        bg_post_prob_nonzero = bg_post_prob_nonzero / np.sum(bg_post_prob_nonzero, axis=-1).reshape(-1, 1)

        top_rel_score = np.max(bg_post_prob_nonzero[:, 1:], axis=-1)
        top_rel = np.argmax(bg_post_prob_nonzero[:, 1:], axis=-1) + 1
        
        # sample rel based on post-prob score
        # top_rel = np.zeros(bg_post_prob_nonzero.shape[0], dtype=int)
        # for i in range(bg_post_prob_nonzero.shape[0]):
        #     sample = np.random.choice(len(self.ind_to_predicates), 1, p=bg_post_prob_nonzero[i])[0]
        #     top_rel[i] = sample
        # filter non-overlapping pairs
        boxes = curr_dict["boxes"]
        sbj_boxes = boxes[bg_pair_idx_nonzero[:, 0]]
        obj_boxes = boxes[bg_pair_idx_nonzero[:, 1]]
        iou = bbox_overlaps_misc(sbj_boxes, obj_boxes).diagonal()
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
        self.label_to_idx = info['obj_name_to_id']
        self.predicate_to_idx = info['rel_name_to_id']
        print("Start computing new relation triples...")

        # iterate through the whole split
        new_triple = []
        self.total_num_new_triple = 0
        self.empty_new_triple_img_cnt = 0
        for index in tqdm(range(len(self.filenames))):
            filename = os.path.join(self.img_dir, self.filenames[index])
            labels = torch.from_numpy(self.gt_classes[index])
            relation = self.relationships[index].copy()

            image_id = int(filename.split("/")[-1].split(".")[0])
            names = [self.ind_to_classes[ii] for ii in self.gt_classes[index]]

            img_info = self.get_img_info(index)
            w, h = img_info['width'], img_info['height']
            # important: recover original box from BOX_SCALE
            box = self.gt_boxes[index].copy()
            box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

            if self.relation_cfg.NEW_REL_FILTER == "iou":
                new_rel = self.prepare_new_rel_iou(index, labels, box, names)
            elif self.relation_cfg.NEW_REL_FILTER == "pretrain":
                new_rel = self.prepare_new_rel_pretrain(index)
            elif self.relation_cfg.NEW_REL_FILTER == "pretrain_internal":
                new_rel = self.transfer_gt_rel_pretrain(index)
            else:
                raise ValueError("Support only ['iou', 'pretrain']. Don't support other deduction methods.")
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
                    pic.save(os.path.join(vis_out_dir, str(image_id)+'.png'))
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


def get_GQA_statistics(img_dir, train_file, dict_file, must_overlap=True):
    from yacs.config import CfgNode as CN
    logits_cfg = CN(new_allowed=True)
    logits_cfg.STAGE = "get-logits"

    train_data = GQARelabeling(split='train', img_dir=img_dir, train_file=train_file,
                           dict_file=dict_file, test_file=None, num_val_im=5000,
                           filter_duplicate_rels=False, logits_cfg=logits_cfg)
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



if __name__=='__main__':
    import argparse
    from yacs.config import CfgNode as CN
    from maskrcnn_benchmark.utils.miscellaneous import save_config

    parser = argparse.ArgumentParser(description="Relation Deduction")
    parser.add_argument(
        "--config-file",
        default="configs/relation_deduction/relabel_new_relation_pretrain_gqa.yaml",
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

    data = GQARelabeling(split='train', img_dir=cfg.IMG_DIR, dict_file=cfg.DICT_FILE, 
                        train_file=cfg.TRAIN_FILE, test_file=cfg.TEST_FILE,
                        filter_non_overlap = False,
                        relation_cfg=cfg.NEW_RELATION, logits_cfg=cfg.USE_LOGITS)
    if hasattr(data, "cache_prefix"):
        save_cfg_name = f"config_{data.cache_prefix}.yml"
    else:
        save_cfg_name = "config.yml"
    save_config(cfg, os.path.join(cfg.NEW_RELATION.OUTPUT_DIR, save_cfg_name))
    print(f"Saved results in {cfg.NEW_RELATION.OUTPUT_DIR}")
