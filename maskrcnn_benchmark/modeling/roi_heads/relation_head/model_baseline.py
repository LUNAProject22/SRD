from turtle import forward
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, encode_box_info


class WordContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(WordContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        self.nl_edge = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER
        assert self.nl_obj > 0 and self.nl_edge > 0

        self.ctx_rnn = torch.nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_obj,
                dropout=self.dropout_rate if self.nl_obj > 1 else 0,
                bidirectional=True)
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)


    def sort_rois(self, proposals):
        c_x = center_x(proposals)
        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(proposals, scores)

    def word_ctx(self, obj_embed, proposals, obj_labels=None, ctx_average=False):
        """
        Object name context.
        :param obj_embed: [num_obj, object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :return: 
                word_ctx: [object embedding0 dim] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(proposals)
        # Pass word embedding, sorted by score, into the encoder LSTM
        emb_inp_rep = obj_embed[perm].contiguous()
        input_packed = PackedSequence(emb_inp_rep, ls_transposed)
        encoder_rep, (h_n, c_n) = self.ctx_rnn(input_packed)
        encoder_rep = encoder_rep[0]
        encoder_rep = self.lin_obj_h(encoder_rep) # map to hidden_dim
        encoder_rep = encoder_rep[inv_perm]
        h_n = torch.sum(h_n, dim=0) # (batch, hidden_dim)
        return encoder_rep, h_n, perm, inv_perm, ls_transposed


    def forward(self, proposals, logger=None, ctx_average=False):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_preds = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_logits = to_onehot(obj_preds, self.num_obj_classes)
            obj_dists = F.softmax(obj_logits, dim=1)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_dists = F.softmax(obj_logits, dim=1)
            obj_preds = obj_dists.max(dim=1)

        # retrieve word embedding
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_preds.long())
        else:
            obj_embed = obj_dists @ self.obj_embed1.weight
        
        assert proposals[0].mode == 'xyxy'

        # image level word embedding
        embed_rep, word_ctx, perm, inv_perm, ls_transposed = self.word_ctx(obj_embed, proposals, obj_preds, ctx_average=ctx_average)

        return obj_dists, obj_preds, embed_rep, word_ctx, perm, inv_perm, ls_transposed


class PosEmbed(nn.Module):
    def __init__(self):
        super(PosEmbed, self).__init__()

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

    def forward(self, proposals):
        pos_embed = self.pos_embed(encode_box_info(proposals))
        return pos_embed