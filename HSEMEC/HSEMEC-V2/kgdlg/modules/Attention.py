import math, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttention(nn.Module):
    def __init__(self, dim, attn_type="dot", input_emb_size=0, emo_emb_size=0, mode='train'):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
            "Please select a valid attention type.")
        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)

            # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2 + emo_emb_size, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.mask = None

        self.mode = mode  # if mode=='test': adaptive_gate=1
        assert (self.mode in ['train', 'test']), (
            "Please select a valid mode type.")
        self.adptive_gate = nn.Sequential(
            nn.Linear(dim * 2 + input_emb_size, 1, bias=True),
            nn.Sigmoid()
        )

    def applyMask(self, mask):
        self.mask = mask

    def score(self, h_t, h_s):
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)

    def forward(self, input, context, emo_emb=None, input_emb=None):

        # one step input
        # print('input.dim: ',input.dim())  == 3
        # print('input.size: ',input.size())    [batch, tgt_len, hid]
        # print('context.size: ',context.size())    [batch, stc_len, hid]
        # print('emo_emb.size: ',emo_embedding.size())  [batch, tgt_len, emo_emb_size]
        # exit()
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()

        # compute attention scores, as in Luong et al.
        align = self.score(input, context)

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch * targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)

        # print('dim: ', dim)
        # print('c_t.size: ', c.size())
        # print('s_t.size: ', input.size())
        # print('emo_emb.size: ', emo_emb.size())
        # print('input_emb.size: ', input_emb.size())

        # calculate adaptive emotion embedding
        adaptive_emo_emb = None
        emo_emb_size, input_emb_size = 0, 0
        if (not emo_emb is None) and (not input_emb is None):
            _, _, emo_emb_size = emo_emb.size()
            _, _, input_emb_size = input_emb.size()
            if self.mode == 'train':
                # print('mode==train')
                gate_input = torch.cat([c, input, input_emb], 2).view(batch * targetL, dim * 2 + input_emb_size)
                adaptive_gate = self.adptive_gate(gate_input).view(batch, targetL, 1)
                adaptive_emo_emb = adaptive_gate * emo_emb
                # print('gate_input.size: ', gate_input.size())
                # print('adaptive_gate.size: ', adaptive_gate.size())
                # print('adaptive_emo_emb.size: ', adaptive_emo_emb.size())
                # exit()
            else:
                # print('mode=test')
                adaptive_emo_emb = emo_emb

            # concatenate
            concat_c = torch.cat([c, input, adaptive_emo_emb], 2).view(batch * targetL, dim * 2 + emo_emb_size)
        else:
            concat_c = torch.cat([c, input], 2).view(batch * targetL, dim * 2)


        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        else:

            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

        # Check output sizes
        # targetL_, batch_, dim_ = attn_h.size()

        # targetL_, batch_, sourceL_ = align_vectors.size()
        return attn_h, align_vectors


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.linear_out = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        raw_query = query
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        attn_out = torch.cat((self.linears[-1](x), raw_query), dim=-1)
        attn_out = self.linear_out(attn_out)
        return attn_out
