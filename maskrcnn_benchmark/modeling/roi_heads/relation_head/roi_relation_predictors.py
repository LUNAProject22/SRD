# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import json
import numpy as np
import pandas as pd
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat, filter_none_overlapping_prop
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .model_baseline import WordContext, PosEmbed
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_motifs import to_onehot
from .no_context_freq_obj_sim import RelPredictor_NoContext
from .lp_context import LPContextBasedPredicatePredictor
from .freq_context_and_lp_no_context_tail import FreqContext_LPNoContextTail
from .freq_context_based_reasoning_v5 import ContextBasedPredicatePredictor
from .clip_vg.clip_model import openclip_relation


@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        if statistics['att_classes'] is not None:
            obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
            assert self.num_att_cls==len(att_classes)
        else:
            obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)

        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # context kg (!when change kg_mode here, should change mode in KGContextPredictor class too)
        self.kg_mode = config.MODEL.KG.MODE #context_freq/lp_no_context/lp_context/fc_lp_no_context_tail/none
        if self.kg_mode != 'none':
            with torch.no_grad():
                self.kg_predictor = KGContextPredictor(self.cfg, in_channels)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))


    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        kg_reasoning = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append( head_rep[pair_idx[:,0]] - tail_rep[pair_idx[:,1]] )
            else:
                ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
            if self.kg_mode == 'context_freq':
                unique_obj_pred = torch.unique(obj_pred)
                # input should be a list of unique objects (not index)
                icontext = [self.kg_predictor.idx_to_label[str(int(i.item()))] for i in unique_obj_pred]
                # get the predicate distributions prediction for all the pairs (allow same-object pair, e.g., (car, car))
                ht2vec = self.kg_predictor.predictor.predict(icontext)  # {(head, tail): vector}
                rel_pair_id = torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1)
                dists = self.kg_predictor.get_kg_rel_dists(rel_pair_id, ht2vec)
                dists = dists.to(obj_pred.device)
                kg_reasoning.append(dists)
            elif self.kg_mode == 'fc_lp_no_context_tail':
                unique_obj_pred = torch.unique(obj_pred)
                icontext = [self.kg_predictor.idx_to_label[str(i.item())] for i in unique_obj_pred]
                self.kg_predictor.predictor.prepare_matrix(icontext, verbose=False)
                prior = self.kg_predictor.predictor.predict(obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]).to(pair_idx.device)
                kg_reasoning.append(prior)
            elif self.kg_mode == 'lp_no_context':
                prior = self.kg_predictor.predictor.predict(obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]).to(pair_idx.device)
                kg_reasoning.append(prior)
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.kg_mode != 'none':
            kg_reasoning = cat(kg_reasoning, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list, kg_reasoning



    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list, kg_reasoning = \
            self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _, _ = \
                    self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats

        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False, kg_reasoning=kg_reasoning)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                if self.kg_mode == 'none':
                    add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)
                else:
                    add_losses['auxiliary_frq'] = F.cross_entropy(kg_reasoning, rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                if self.kg_mode == 'none':
                    rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) \
                            - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
                else:
                    # rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False, kg_reasoning=kg_reasoning) \
                    #             - self.calculate_logits(union_features, avg_ctx_rep, pair_pred, use_label_dist=False, \
                    #                 kg_reasoning=torch.zeros_like(kg_reasoning, device=kg_reasoning.device)) #2
                    rel_dists = F.softmax(self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False) \
                            - self.calculate_logits(union_features, avg_ctx_rep, pair_pred, use_label_dist=False), dim=1) \
                                + F.normalize(kg_reasoning.float(), p=1, dim=1) #3
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) \
                            - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) \
                            - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False, kg_reasoning=None):
        if self.kg_mode == 'none':
            if use_label_dist:
                frq_dists = self.freq_bias.index_with_probability(frq_rep)
            else:
                frq_dists = self.freq_bias.index_with_labels(frq_rep.long())
        else:
            frq_dists = None

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest

        elif self.fusion_type == 'sum':
            if self.kg_mode == 'none':
                union_dists = vis_dists + ctx_dists + frq_dists
            else:
                union_dists = vis_dists + ctx_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


@registry.ROI_RELATION_PREDICTOR.register("CLIPPredictor")
class CLIPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CLIPPredictor, self).__init__()
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.use_gt_box = config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
        self.use_gt_class = config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL

        # module construct
        model_name = 'ViT-B-16'
        model_path = config.MODEL.CLIP_CKPT
        vg_dict_json = './datasets/vg/VG-SGG-dicts-with-attri.json'
        image_folder = 'datasets/vg/'
        self.model = openclip_relation(model_name, model_path, vg_dict_json, image_folder)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Input:
            proposals (list[BoxList])
            roi_features (batch_num_roi, 4096)
            union_features (batch_num_rel_pair, 4096)
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
        """

        if self.training or self.use_gt_class:
            obj_preds = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            obj_dists = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        obj_dists = obj_dists.split(num_objs, dim=0)

        rel_dists = []
        for i, p in enumerate(proposals):
            image_name = p.get_field("image_path").split("/")[-1]
            gt_boxes = p.bbox
            gt_objects = p.get_field("labels")
            gt_relations = rel_pair_idxs[i]

            #model.predict_one_image(image_name, gt_boxes, gt_objects)
            Detected_List, Detected_List_Probs = self.model.predict_one_image(image_name, gt_boxes, gt_objects, gt_relations)
            # Detected_List, Detected_List_Probs= self.model.predict_one_image(image_name, gt_boxes, gt_objects)
            Detected_List_Probs = torch.stack(Detected_List_Probs, dim=0).squeeze(1)

            rel_dists.append(Detected_List_Probs)

        add_losses = {}
        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("KGpriorPredictor")
class KGpriorPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(KGpriorPredictor, self).__init__()
        '''
        :param knowledge_file: path to subj_obj_rel_filtered_aggregated.csv
        :param index_file: path to VG-SGG-dicts.json
        '''
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)

        # Knowledge Graph
        self.DEFAULT = 0.0 # default probability for not-found relations
        self.KG_FILE = "datasets/kg/train_triples_count_filtered_aggregated.csv" #"datasets/kg/subj_obj_rel_filtered_aggregated.csv"
        self.INDEX_FILE = "datasets/vg/VG-SGG-dicts.json"
        self.label_to_idx, self.idx_to_label, self.predicate_to_idx, self.idx_to_predicate = self.load_mappings(self.INDEX_FILE)
        self.head2tail2vector = self.load_prior_knowledge(self.KG_FILE)
        self.overlapping = True

    def load_mappings(self, index_file):
        data = json.load(open(index_file))
        label_to_idx = data['label_to_idx']
        idx_to_label = data['idx_to_label']
        predicate_to_idx = data['predicate_to_idx']
        idx_to_predicate = data['idx_to_predicate']
        return label_to_idx, idx_to_label, predicate_to_idx, idx_to_predicate

    def load_prior_knowledge(self, knowledge_file):
        dt = pd.read_csv(knowledge_file)
        head2tail2vector = {}  # {head: {tail: [vector]}}
        for head, tail, rel, prob in zip(dt['head'], dt['tail'], dt['relation'], dt['prob']):
            if head not in head2tail2vector:
                head2tail2vector[head] = {}
            if tail not in head2tail2vector[head]:
                head2tail2vector[head][tail] = np.full(self.num_rel_cls, self.DEFAULT)
            head2tail2vector[head][tail][self.predicate_to_idx.get(rel, 0)] = prob  # 0 in case the verb is not found, should not have any
        return head2tail2vector

    def is_valid_input(self, head, tail):
        if head is not None and tail is not None and head in self.head2tail2vector and tail in self.head2tail2vector[head]:
            return True
        return False

    def get_predicate_distribution(self, h, t, as_tensor=False):
        if isinstance(h, int) and isinstance(t, int):
            # head and tail is index
            head = self.idx_to_label.get(str(h), None)
            tail = self.idx_to_label.get(str(t), None)
        if self.is_valid_input(head, tail):
            vec = self.head2tail2vector[head][tail]
        else:
            vec = np.zeros(self.num_rel_cls)
        if as_tensor:
            vec = torch.from_numpy(vec)
        return vec


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Input:
            proposals (list[BoxList])
            roi_features (batch_num_roi, 4096)
            union_features (batch_num_rel_pair, 4096)
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
        """
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_preds = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_logits = to_onehot(obj_preds, self.num_obj_cls)
            obj_dists = F.softmax(obj_logits, dim=1)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_dists = F.softmax(obj_logits, dim=1)
            obj_preds = obj_dists.max(dim=1)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_dists = obj_dists.split(num_objs, dim=0)

        if self.overlapping:
            rel_pair_idxs = filter_none_overlapping_prop(proposals)

        prod_reps = []
        for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds): #image-level
            prior = []
            for i in range(pair_idx.shape[0]): #pair-level
                prior_i = self.get_predicate_distribution(obj_pred[pair_idx[i, 0]].item(), \
                                obj_pred[pair_idx[i, 1]].item(), as_tensor=True).to(pair_idx.device)
                prior.append(prior_i)
            prior = torch.stack(prior, dim=0)
            prod_reps.append(prior)

        rel_dists = prod_reps
        add_losses = {}
        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("KGContextPredictor")
class KGContextPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(KGContextPredictor, self).__init__()

        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)

        self.mode = config.MODEL.KG.MODE # context_freq/lp_no_context/lp_context/fc_lp_no_context_tail/none
        self.INDEX_FILE = "datasets/vg/VG-SGG-dicts.json"
        self.label_to_idx, self.idx_to_label, self.predicate_to_idx, self.idx_to_predicate = self.load_mappings(self.INDEX_FILE)

        if self.mode == 'context_freq':
            # context based reasoning
            self.PRIOR_FILE = 'datasets/kg/train_triples_context_essential_info.pkl'
            prior_knowledge_file = 'datasets/kg/prior_knowledge_0920.pkl'
            self.predictor = ContextBasedPredicatePredictor(prior_file=self.PRIOR_FILE, \
                            prior_knowledge_file=prior_knowledge_file, index_file=self.INDEX_FILE)
        elif self.mode == 'lp_no_context':
            # lp reasoning no context
            # self.PRIOR_FILE = 'datasets/kg/matrix_normalized_by_tails.pkl'
            # self.PRIOR_FILE = 'datasets/kg/prior_knowledge.pkl'
            # self.predictor = LPNoContextPredPredictor(prior_file=self.PRIOR_FILE, index_file=self.INDEX_FILE)
            self.PRIOR_FILE = 'datasets/kg/prior_knowledge_0920.pkl'
            self.predictor = RelPredictor_NoContext(prior_file=self.PRIOR_FILE, index_file=self.INDEX_FILE,\
                            knowledge_key='matrix_rht_sim_paraphrase-mpnet-base-v2')
        elif self.mode == 'lp_context':
            self.PRIOR_FILE = 'datasets/kg/lp_context_info.pkl'
            self.predictor = LPContextBasedPredicatePredictor(prior_file=self.PRIOR_FILE, index_file=self.INDEX_FILE)
        elif self.mode == 'fc_lp_no_context_tail':
            self.PRIOR_FILE = 'datasets/kg/freq_context-lp_no_context_by_tail.pkl'
            self.predictor = FreqContext_LPNoContextTail(prior_file=self.PRIOR_FILE, index_file=self.INDEX_FILE)


    def load_mappings(self, index_file):
        data = json.load(open(index_file))
        label_to_idx = data['label_to_idx']
        idx_to_label = data['idx_to_label']
        predicate_to_idx = data['predicate_to_idx']
        idx_to_predicate = data['idx_to_predicate']
        return label_to_idx, idx_to_label, predicate_to_idx, idx_to_predicate

    def get_kg_rel_pair_id(self, ht2vec):
        idxs = []
        for (h, t), v in ht2vec.items():
            idxs.append([int(self.label_to_idx[h]), int(self.label_to_idx[t])])
        idxs = torch.from_numpy(np.array(idxs, dtype=int))
        return idxs

    def get_kg_rel_dists(self, pairs, ht2vec):
        dists = []
        for i in range(pairs.shape[0]):
            head = self.idx_to_label[str(int(pairs[i, 0].item()))]
            tail = self.idx_to_label[str(int(pairs[i, 1].item()))]
            if (head, tail) in ht2vec.keys():
                dists.append(ht2vec[(head, tail)])
            else:
                dists.append(np.zeros(self.num_rel_cls))
        dists = torch.from_numpy(np.array(dists, dtype=np.float64))
        return dists


    def id_to_idx(self, id, obj_pred):
        idxs = torch.zeros_like(id)
        for i in range(id.shape[0]):
            h_idx = (obj_pred == id[i][0]).nonzero(as_tuple=True)[0]
            t_idx = (obj_pred == id[i][1]).nonzero(as_tuple=True)[0]
            idxs[i][0] = h_idx[0]
            idxs[i][1] = t_idx[0]
        return idxs

    def convert_rel_pair_idxs(self, rel_pair_idx_prop, rel_pair_id, obj_pred):
        new_pair_idx = []
        new_pair_id = []
        pair_id_prop = torch.column_stack((obj_pred[rel_pair_idx_prop[:, 0]], obj_pred[rel_pair_idx_prop[:, 1]]))
        for i in range(pair_id_prop.shape[0]):
            for j in range(rel_pair_id.shape[0]):
                if pair_id_prop[i, 0] == rel_pair_id[j, 0] and pair_id_prop[i, 1] == rel_pair_id[j, 1]:
                    new_pair_idx.append([rel_pair_idx_prop[i, 0].item(), rel_pair_idx_prop[i, 1].item()])
                    new_pair_id.append([pair_id_prop[i, 0].item(), pair_id_prop[i, 1].item()])
                    break
        new_pair_idx = torch.from_numpy(np.array(new_pair_idx))
        new_pair_id = torch.from_numpy(np.array(new_pair_id))
        return new_pair_idx, new_pair_id

    def infer_new_rel_pair_using_kg(self, rel_pair_idx_prop, rel_pair_id, obj_pred, ht2vec):
        new_pair_idx = []
        dists = []
        pair_id_prop = torch.column_stack((obj_pred[rel_pair_idx_prop[:, 0]], obj_pred[rel_pair_idx_prop[:, 1]]))
        for i in range(pair_id_prop.shape[0]):
            head_dists = []
            tail_dists = []
            for j in range(rel_pair_id.shape[0]):
                if pair_id_prop[i, 0] == rel_pair_id[j, 0]: # prop head is also kg dict head
                    head = self.idx_to_label[str(rel_pair_id[j, 0].item())]
                    tail = self.idx_to_label[str(rel_pair_id[j, 1].item())]
                    head_dists.append(ht2vec[(head, tail)])
                elif pair_id_prop[i, 1] == rel_pair_id[j, 1]: # prop tail is also kg dict tail
                    head = self.idx_to_label[str(rel_pair_id[j, 0].item())]
                    tail = self.idx_to_label[str(rel_pair_id[j, 1].item())]
                    tail_dists.append(ht2vec[(head, tail)])
            if head_dists == [] and tail_dists == []:
                continue
            elif head_dists != [] and tail_dists == []:
                dist = sum(np.array(head_dists)) / len(head_dists)
            elif head_dists == [] and tail_dists != []:
                dist = sum(np.array(tail_dists)) / len(head_dists)
            else:
                h = sum(np.array(head_dists)) / len(head_dists)
                t = sum(np.array(tail_dists)) / len(head_dists)
                dist = h * t # element-wise
            assert dist.shape[0] > 0, "Infered relation distribution is None!"
            new_pair_idx.append([rel_pair_idx_prop[i, 0].item(), rel_pair_idx_prop[i, 1].item()])
            dists.append(dist)
        new_pair_idx = torch.from_numpy(np.array(new_pair_idx))
        dists = torch.from_numpy(np.array(dists, dtype=np.float64))
        assert new_pair_idx.shape[0] == dists.shape[0], "Infered relation and distribution shape doesn't match."
        return new_pair_idx, dists


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Input:
            proposals (list[BoxList])
            roi_features (batch_num_roi, 4096)
            union_features (batch_num_rel_pair, 4096)
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
        """
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_preds = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_logits = to_onehot(obj_preds, self.num_obj_cls)
            obj_dists = F.softmax(obj_logits, dim=1)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_dists = F.softmax(obj_logits, dim=1)
            obj_preds = obj_dists.max(dim=1)

        num_objs = [len(b) for b in proposals]
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_dists = obj_dists.split(num_objs, dim=0)

        prod_reps = []
        if self.mode == 'context_freq':
            for obj_pred, pair_idx in zip(obj_preds, rel_pair_idxs): #image-level
                unique_obj_pred = torch.unique(obj_pred)
                # input should be a list of unique objects (not index)
                icontext = [self.idx_to_label[str(int(i))] for i in unique_obj_pred]
                # get the predicate distributions prediction for all the pairs (allow same-object pair, e.g., (car, car))
                ht2vec = self.predictor.predict(icontext)  # {(head, tail): vector}
                # rel_pair_id = self.get_kg_rel_pair_id(ht2vec)
                # assert rel_pair_id.shape[0] != 0

                # rel_pair_idx, new_pair_id = self.convert_rel_pair_idxs(pair_idx, rel_pair_id, obj_pred)
                # if rel_pair_idx.shape[0] == 0:
                #     rel_pair_idx = self.id_to_idx(rel_pair_id, obj_pred)
                #     new_pair_id = rel_pair_id
                # dists = self.get_kg_rel_dists(new_pair_id, ht2vec)

                rel_pair_id = torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1)
                dists = self.get_kg_rel_dists(rel_pair_id, ht2vec)

                # if rel_pair_idx.shape[0] == 0:
                #     rel_pair_idx, dists= self.infer_new_rel_pair_using_kg(pair_idx, rel_pair_id, obj_pred, ht2vec)
                # else:
                #     dists = self.get_kg_rel_dists(new_pair_id, ht2vec)

                dists = dists.to(obj_pred.device)
                prod_reps.append(dists)
        elif self.mode == 'lp_no_context':
            for obj_pred, pair_idx in zip(obj_preds, rel_pair_idxs): #image-level
                prior = self.predictor.predict(obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]).to(pair_idx.device)
                prod_reps.append(prior)
        elif self.mode == 'lp_context':
            for obj_pred, pair_idx in zip(obj_preds, rel_pair_idxs): #image-level
                unique_obj_pred = torch.unique(obj_pred)
                icontext = [self.idx_to_label[str(i.item())] for i in unique_obj_pred]
                normalize_mapping = {'by_tail': 1, 'by_predicate': 2}
                # prepare the normalized matrix (only need to run once for each input context)
                self.predictor.prepare_pht_matrix(icontext, matrix_normalize_dim=normalize_mapping['by_predicate'], verbose=False)
                prior = self.predictor.predict(obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]).to(pair_idx.device)
                prod_reps.append(prior)
        elif self.mode == 'fc_lp_no_context_tail':
            for obj_pred, pair_idx in zip(obj_preds, rel_pair_idxs): #image-level
                unique_obj_pred = torch.unique(obj_pred)
                icontext = [self.idx_to_label[str(i.item())] for i in unique_obj_pred]
                # prepare ic-based matrix
                self.predictor.prepare_matrix(icontext, verbose=False)
                # obtain prediction
                prior = self.predictor.predict(obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]).to(pair_idx.device)
                prod_reps.append(prior)

        rel_dists = prod_reps
        add_losses = {}
        return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
