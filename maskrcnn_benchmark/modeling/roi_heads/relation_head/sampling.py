# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

class RelationSampling(object):
    def __init__(
        self,
        fg_thres,
        require_overlap,
        num_sample_per_gt_rel,
        batch_size_per_image,
        positive_fraction,
        use_gt_box,
        test_overlap,
        test_overlap_thres
    ):
        self.fg_thres = fg_thres
        self.require_overlap = require_overlap
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.use_gt_box = use_gt_box
        self.test_overlap = test_overlap
        self.test_overlap_thres = test_overlap_thres

    def prepare_test_pairs(self, device, proposals, ds_name='ag'):
        # prepare object pairs for relation prediction
        rel_pair_idxs = []
        for p in proposals:
            n = len(p)
            if 'ag' in ds_name:
                if n <= 1:
                    rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
                else:
                    obj_idx = torch.from_numpy(np.arange(1, n)).to(device)
                    sbj_idx = torch.zeros((n-1), dtype=torch.int64, device=device)
                    pair_idx = torch.column_stack((sbj_idx, obj_idx))
                    rel_pair_idxs.append(pair_idx)
            else:
                cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
                # mode==sgdet and require_overlap
                # if (not self.use_gt_box) and self.test_overlap:
                if self.test_overlap:
                    cand_matrix = cand_matrix.byte() & boxlist_iou(p, p).gt(self.test_overlap_thres).byte()
                idxs = torch.nonzero(cand_matrix).view(-1,2)
                if len(idxs) > 0:
                    rel_pair_idxs.append(idxs)
                else:
                    # if there is no candidate pairs, give a placeholder of [[0, 0]]
                    rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
        return rel_pair_idxs


    def gtbox_relsample(self, proposals, targets, ds_name='ag'):
        assert self.use_gt_box
        num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            num_prp = proposal.bbox.shape[0]
            assert proposal.bbox.shape[0] == target.bbox.shape[0]
            tgt_rel_matrix = target.get_field("relation") # [tgt, tgt]
            assert tgt_rel_matrix.shape[0] == num_prp, 'tgt_rel_matrix dim wrong'
            if 'ag' in ds_name:
                tgt_classes = target.get_field("labels")
                tgt_head_id = tgt_rel_matrix[:, 0].contiguous().view(-1)
                tgt_tail_id = tgt_rel_matrix[:, 1].contiguous().view(-1)
                tgt_rel_labs = tgt_rel_matrix[:, 2].contiguous().view(-1)

                tgt_head_idxs = []
                tgt_tail_idxs = []
                for head, tail in zip(tgt_head_id, tgt_tail_id):
                    if (head not in tgt_classes) or (tail not in tgt_classes):
                        continue
                    idx_h = int((tgt_classes == head).nonzero(as_tuple=False)[0])
                    idx_t = int((tgt_classes == tail).nonzero(as_tuple=False)[0])
                    tgt_head_idxs.append(idx_h)
                    tgt_tail_idxs.append(idx_t)
                tgt_head_idxs = torch.from_numpy(np.array(tgt_head_idxs)).to(device)
                tgt_tail_idxs = torch.from_numpy(np.array(tgt_tail_idxs)).to(device)
                tgt_pair_idxs = torch.column_stack((tgt_head_idxs, tgt_tail_idxs))
            else:
                tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
                if tgt_pair_idxs.shape[0] == 0:
                    tgt_pair_idxs = torch.zeros((1, 2), dtype=torch.int64, device=device)
                tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
                tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
                tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)
            assert tgt_pair_idxs.shape[1] == 2

            # sym_binary_rels
            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            rel_sym_binarys.append(binary_rel)

            # generate fg bg rel_pairs
            if tgt_pair_idxs.shape[0] > num_pos_per_img:
                perm = torch.randperm(tgt_pair_idxs.shape[0], device=device)[:num_pos_per_img]
                tgt_pair_idxs = tgt_pair_idxs[perm]
                tgt_rel_labs = tgt_rel_labs[perm]

            if 'ag' not in ds_name or 'AG' not in ds_name:
                rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp, device=device).long()
                rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0
                tgt_bg_idxs = torch.nonzero(rel_possibility > 0)
                num_fg = min(tgt_pair_idxs.shape[0], num_pos_per_img)

                num_bg = self.batch_size_per_image - num_fg
                perm = torch.randperm(tgt_bg_idxs.shape[0], device=device)[:num_bg]
                tgt_bg_idxs = tgt_bg_idxs[perm]

                img_rel_idxs = torch.cat((tgt_pair_idxs, tgt_bg_idxs), dim=0)
                img_rel_labels = torch.cat((tgt_rel_labs.long(), torch.zeros(tgt_bg_idxs.shape[0], device=device).long()), dim=0).contiguous().view(-1)
            else:
                img_rel_idxs = tgt_pair_idxs
                img_rel_labels = tgt_rel_labs.long().contiguous().view(-1)

            rel_idx_pairs.append(img_rel_idxs)
            rel_labels.append(img_rel_labels)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys


    def detect_relsample(self, proposals, targets):
        # corresponding to rel_assignments function in neural-motifs
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])  contain fields: labels, predict_logits
            targets (list[BoxList]) contain fields: labels
        """
        self.num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            prp_box = proposal.bbox
            prp_lab = proposal.get_field("labels").long()
            tgt_box = target.bbox
            tgt_lab = target.get_field("labels").long()
            tgt_rel_matrix = target.get_field("relation") # [tgt, tgt]
            # IoU matching
            ious = boxlist_iou(target, proposal)  # [tgt, prp]
            is_match = (tgt_lab[:,None] == prp_lab[None]) & (ious > self.fg_thres) # [tgt, prp]
            # Proposal self IoU to filter non-overlap
            prp_self_iou = boxlist_iou(proposal, proposal)  # [prp, prp]
            if self.require_overlap and (not self.use_gt_box):
                rel_possibility = (prp_self_iou > 0) & (prp_self_iou < 1)  # not self & intersect
            else:
                num_prp = prp_box.shape[0]
                rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp, device=device).long()
            # only select relations between fg proposals
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, binary_rel = self.motif_rel_fg_bg_sampling(device, tgt_rel_matrix, ious, is_match, rel_possibility)
            rel_idx_pairs.append(img_rel_triplets[:, :2]) # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2]) # (num_rel, )
            rel_sym_binarys.append(binary_rel)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys


    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, ious, is_match, rel_possibility):
        """
        prepare to sample fg relation triplet and bg relation triplet
        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]
        """
        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs] # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[tgt_tail_idxs] # num_tgt_rel, num_prp (matched prp head)
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                # binary rel only consider related or not, so its symmetric
                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])
            # find matching pair in proposals (might be more than one)
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs
            prp_head_idxs = prp_head_idxs.view(-1,1).expand(num_match_head,num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1,-1).expand(num_match_head,num_match_tail).contiguous().view(-1)
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_labels = torch.tensor([tgt_rel_lab]*prp_tail_idxs.shape[0], dtype=torch.int64, device=device).view(-1,1)
            fg_rel_i = cat((prp_head_idxs.view(-1,1), prp_tail_idxs.view(-1,1), fg_labels), dim=-1).to(torch.int64)
            # select if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(-1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.shape[0] > 0:
                fg_rel_triplets.append(fg_rel_i)

        # select fg relations
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
        else:
            fg_rel_triplets = cat(fg_rel_triplets, dim=0).to(torch.int64)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img:
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]

        # select bg relations
        bg_rel_inds = torch.nonzero(rel_possibility>0).view(-1,2)
        bg_rel_labs = torch.zeros(bg_rel_inds.shape[0], dtype=torch.int64, device=device)
        bg_rel_triplets = cat((bg_rel_inds, bg_rel_labs.view(-1,1)), dim=-1).to(torch.int64)

        num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
        if bg_rel_triplets.shape[0] > 0:
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)

        # if both fg and bg is none
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            bg_rel_triplets = torch.zeros((1, 3), dtype=torch.int64, device=device)

        return cat((fg_rel_triplets, bg_rel_triplets), dim=0), binary_rel


class WRelationSampling(object):
    def __init__(
            self,
            fg_thres,
            require_overlap,
            num_sample_per_gt_rel,
            batch_size_per_image,
            positive_fraction,
            use_gt_box,
            test_overlap,
            num_rel_classes,
            test_overlap_thres
    ):
        self.fg_thres = fg_thres
        self.require_overlap = require_overlap
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.use_gt_box = use_gt_box
        self.test_overlap = test_overlap
        self.num_rel_classes = num_rel_classes
        self.test_overlap_thres = test_overlap_thres

    def prepare_test_pairs(self, device, proposals):
        # prepare object pairs for relation prediction
        rel_pair_idxs = []
        for p in proposals:
            n = len(p)
            cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
            # mode==sgdet and require_overlap
            # if (not self.use_gt_box) and self.test_overlap:
            if self.test_overlap:
                cand_matrix = cand_matrix.byte() & boxlist_iou(p, p).gt(self.test_overlap_thres).byte()
            idxs = torch.nonzero(cand_matrix).view(-1, 2)
            if len(idxs) > 0:
                rel_pair_idxs.append(idxs)
            else:
                # if there is no candidate pairs, give a placeholder of [[0, 0]]
                rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
        return rel_pair_idxs

    def gtbox_relsample(self, proposals, targets):
        # print("batch_size_per_image", self.batch_size_per_image)
        assert self.use_gt_box
        num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        # rel_soft_labels = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            num_prp = proposal.bbox.shape[0]

            assert proposal.bbox.shape[0] == target.bbox.shape[0]
            # tgt_rel_matrix = target.get_field("relation")  # [tgt, tgt]
            # tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            tgt_pair_idxs = target.get_field("relation_pair_idxs").long()
            # tgt_soft_labels = None
            # if target.has_field("relation_soft_labels"):
            #     tgt_soft_labels = target.get_field("relation_soft_labels")
            # print(tgt_pair_idxs.shape)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = target.get_field("relation_labels")

            # sym_binary_rels
            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            # if num_prp <= tgt_head_idxs.max() or num_prp <= tgt_tail_idxs.max():
            #     tgt_tail_idxs[0] = 0
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            rel_sym_binarys.append(binary_rel)

            rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp,
                                                                                               device=device).long()
            rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0
            tmp=rel_possibility > 0
            tgt_bg_idxs = tmp.nonzero()

            # generate fg bg rel_pairs
            if tgt_pair_idxs.shape[0] > num_pos_per_img:
                perm = torch.randperm(tgt_pair_idxs.shape[0], device=device)[:num_pos_per_img]
                tgt_pair_idxs = tgt_pair_idxs[perm]
                tgt_rel_labs = tgt_rel_labs[perm]
                # if tgt_soft_labels is not None:
                #     tgt_soft_labels = tgt_soft_labels[perm]
            num_fg = min(tgt_pair_idxs.shape[0], num_pos_per_img)

            # TODO original code is to sample as much as possible, change to sample num_fg
            num_bg = self.batch_size_per_image - num_fg
            # num_bg = num_fg
            perm = torch.randperm(tgt_bg_idxs.shape[0], device=device)[:num_bg]
            tgt_bg_idxs = tgt_bg_idxs[perm]

            img_rel_idxs = torch.cat((tgt_pair_idxs, tgt_bg_idxs), dim=0)
            bg_labels = torch.zeros((tgt_bg_idxs.shape[0], self.num_rel_classes), device=device, dtype=torch.float32)
            # TODO  bg pairs
            bg_labels[:, 0] = 1.
            img_rel_labels = torch.cat((tgt_rel_labs, bg_labels), dim=0)
            # if tgt_soft_labels is not None:
            #     bg_soft_labels = torch.ones((tgt_bg_idxs.shape[0], self.num_rel_classes), device=device,
            #                             dtype=torch.float32)
            #     rel_soft_labels.append(torch.cat((tgt_soft_labels, bg_soft_labels), dim=0))

            rel_idx_pairs.append(img_rel_idxs)
            rel_labels.append(img_rel_labels)
        # if len(rel_soft_labels) > 1:
        #     rel_sym_binarys = rel_soft_labels
        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    def detect_relsample(self, proposals, targets):
        # corresponding to rel_assignments function in neural-motifs
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])  contain fields: labels, predict_logits
            targets (list[BoxList]) contain fields: labels
        """
        # import pickle
        # device = proposals[0].bbox.device
        # pickle.dump({'proposals':proposals, "targets":targets}, open("tmp/debug_detect/"+str(device)+"-"+str(self.index)+'.pk', "wb"))
        # self.index += 1

        self.num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        rel_soft_labels = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            prp_box = proposal.bbox
            prp_lab = proposal.get_field("labels").long()
            tgt_box = target.bbox
            tgt_lab = target.get_field("labels").long()
            tgt_pair_idxs = target.get_field("relation_pair_idxs")  # [tgt, tgt]
            # IoU matching
            ious = boxlist_iou(target, proposal)  # [tgt, prp]
            is_match = (tgt_lab[:, None] == prp_lab[None]) & (ious > self.fg_thres)  # [tgt, prp]
            # Proposal self IoU to filter non-overlap
            prp_self_iou = boxlist_iou(proposal, proposal)  # [prp, prp]
            if self.require_overlap and (not self.use_gt_box):
                rel_possibility = (prp_self_iou > 0) & (prp_self_iou < 1)  # not self & intersect
            else:
                num_prp = prp_box.shape[0]
                rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp,
                                                                                                   device=device).long()
            # only select relations between fg proposals
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, img_pair_idxs, binary_rel = self.motif_rel_fg_bg_sampling(device, tgt_pair_idxs, ious, is_match,
                                                                         rel_possibility, target)
            rel_idx_pairs.append(img_pair_idxs)  # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets)  # (num_rel, )
            rel_sym_binarys.append(binary_rel)
        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    def motif_rel_fg_bg_sampling(self, device, tgt_pair_idxs, ious, is_match, rel_possibility, target):
        """
        prepare to sample fg relation triplet and bg relation triplet
        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]
        """
        # tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = target.get_field("relation_labels")
        # print(tgt_rel_labs)
        # tgt_soft_labs = None
        # if target.has_field("relation_soft_labels"):
        #     tgt_soft_labs = target.get_field("relation_soft_labels")

        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[tgt_tail_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
        fg_pair_idxs = []
        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                # binary rel only consider related or not, so its symmetric
                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = tgt_rel_labs[i]
            # find matching pair in proposals (might be more than one)
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs
            prp_head_idxs = prp_head_idxs.view(-1, 1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1, -1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
            # construct corresponding proposal triplets corresponding to i_th gt relation
            # fg_labels = torch.tensor([tgt_rel_lab] * prp_tail_idxs.shape[0], dtype=torch.int64, device=device).view(-1, 1)
            fg_labels = tgt_rel_lab.repeat(prp_tail_idxs.shape[0], 1)
            # fg_rel_i = cat((prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1), fg_labels), dim=-1).to(torch.float32)
            fg_rel_i = fg_labels
            fg_pair_i = cat((prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1)), dim=-1).to(torch.int64)
            # select if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(
                    -1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_i = fg_rel_i[perm]
                fg_pair_i = fg_pair_i[perm]
            if fg_rel_i.shape[0] > 0:
                fg_pair_idxs.append(fg_pair_i)
                fg_rel_triplets.append(fg_rel_i)

        # select fg relations
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, self.num_rel_classes), dtype=torch.float32, device=device)
            fg_pair_idxs = torch.zeros((0, 2), dtype=torch.int64, device=device)
        else:
            fg_rel_triplets = cat(fg_rel_triplets, dim=0)
            fg_pair_idxs = cat(fg_pair_idxs, dim=0)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img:
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]
                fg_pair_idxs = fg_pair_idxs[perm]

        # select bg relations
        bg_rel_inds = torch.nonzero(rel_possibility > 0).view(-1, 2)
        bg_rel_labs = torch.zeros((bg_rel_inds.shape[0], self.num_rel_classes), dtype=torch.float32, device=device)
        bg_rel_labs[:, 0] = 1.
        # print(bg_rel_inds.size(), bg_rel_labs.size())
        # bg_rel_triplets = cat((bg_rel_inds, bg_rel_labs), dim=-1).to(torch.float32)
        bg_rel_triplets = bg_rel_labs
        bg_pair_idxs = bg_rel_inds

        num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
        if bg_rel_triplets.shape[0] > 0:
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
            bg_pair_idxs = bg_pair_idxs[perm]
        else:
            bg_rel_triplets = torch.zeros((0, self.num_rel_classes), dtype=torch.float32, device=device)
            bg_rel_triplets[:, 0] = 1.
            bg_pair_idxs = torch.zeros((0, 2), dtype=torch.int64, device=device)

        # if both fg and bg is none
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            bg_rel_triplets = torch.zeros((1, self.num_rel_classes), dtype=torch.float32, device=device)
            bg_rel_triplets[:, 0] = 1.
            bg_pair_idxs = torch.zeros((1, 2), dtype=torch.int32, device=device)
        return cat((fg_rel_triplets, bg_rel_triplets), dim=0), cat((fg_pair_idxs, bg_pair_idxs), dim=0), binary_rel



def make_weaksup_roi_relation_sample_processor(cfg):
    samp_processor = WRelationSampling(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP,
        cfg.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL,
        cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION,
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
        cfg.TEST.RELATION.REQUIRE_OVERLAP,
        num_rel_classes=cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES,
        test_overlap_thres=cfg.TEST.RELATION.TEST_OVERLAP_THRES
    )
    return samp_processor


def make_roi_relation_samp_processor(cfg):
    samp_processor = RelationSampling(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP,
        cfg.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL,
        cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION,
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
        cfg.TEST.RELATION.REQUIRE_OVERLAP,
        cfg.TEST.RELATION.TEST_OVERLAP_THRES
    )

    return samp_processor
