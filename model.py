#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.nn.init as init
from numba import jit
from entmax import entmax_bisect



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size * 2
        self.input_size = self.hidden_size * 2
        self.gate_size = 3 * self.hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        # input_in = A_in * linear_edge+b_iah
        # A:100*8*16
        # hidden = 100*8*100
        # b_iah 10
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)  # (100*8*100)*(300*100)+(300) = 100*8*300
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)  # *： element-wise product
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):  # [b,s,s] [b,s,d]
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class FindSimilarIntentSess(Module):
    def __init__(self, hidden_size):
        super(FindSimilarIntentSess, self).__init__()
        self.hidden_size = hidden_size
        self.neighbor_n = 5
        self.dropout40 = nn.Dropout(0.40)

    def compute_sim(self, sess_emb):
        fenzi = torch.matmul(sess_emb, sess_emb.permute(1, 0))  # 512*512
        fenmu_l = torch.sum(sess_emb * sess_emb + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu  # 512*512
        cos_sim = nn.Softmax(dim=-1)(cos_sim)
        return cos_sim

    def forward(self, sess_emb):
        k_v = self.neighbor_n
        cos_sim = self.compute_sim(sess_emb)
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_emb[topk_indice]

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.hidden_size)

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)  # [b,d]
        return neighbor_sess


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.dataset = opt.dataset
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0, max_norm=1.5)
        self.pos_embedding = nn.Embedding(300, self.hidden_size, padding_idx=0, max_norm=1.5)
        self.gnn = GNN(self.hidden_size, step=opt.step)

        # Sparse Attention
        self.is_dropout = True
        self.w = 20
        dim = self.hidden_size * 2
        self.dim = dim
        self.LN = nn.LayerNorm(dim)
        self.LN2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.activate = F.relu
        self.atten_w0 = nn.Parameter(torch.Tensor(1, dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_bias = nn.Parameter(torch.Tensor(dim))
        self.attention_mlp = nn.Linear(dim, dim)
        self.alpha_w = nn.Linear(dim, 1)
        self.self_atten_w1 = nn.Linear(dim, dim)
        self.self_atten_w2 = nn.Linear(dim, dim)
        self.linear2_1 = nn.Linear(2*dim, dim, bias=True)

        # Similar-Intent Collaboration Component
        self.FindNeighbor = FindSimilarIntentSess(self.hidden_size)
        self.n_target = 5
        self.w_ne = opt.w_ne

        self.gama = opt.gama

        self.w_f = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.linear_one = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=True)
        self.linear_two = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=True)
        self.linear_three = nn.Linear(2 * self.hidden_size, 1, bias=False)

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.LayerNorm = LayerNorm(2*self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.2)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)



    def add_position_embedding(self, sequence):

        batch_size = sequence.shape[0]  # b
        len = sequence.shape[1]  # s

        position_ids = torch.arange(len, dtype=torch.long, device=sequence.device)  # [s,]
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # [b,s]
        position_embeddings = self.pos_embedding(position_ids)  # [b,s,d]
        item_embeddings = self.embedding(sequence)

        # sequence_emb = self.linear_transform(torch.cat((item_embeddings, position_embeddings), -1))
        # sequence_emb = item_embeddings + position_embeddings
        sequence_emb = torch.cat((item_embeddings, position_embeddings), -1)
        sequence_emb = self.LayerNorm(sequence_emb)
        # sequence_emb = self.dropout(sequence_emb)

        return sequence_emb


    def denoise(self, alpha):  #[b, s+1, s+1]
        batch_size = alpha.shape[0]
        seq_len = alpha.shape[1]
        alpha_avg = torch.mean(alpha, 2, keepdim=True).expand(batch_size, seq_len, seq_len)
        alpha_mask = alpha - 0.1 * alpha_avg

        alpha_out = torch.where(alpha_mask > 0, alpha, trans_to_cuda(torch.tensor([0.])))
        return alpha_out

    def enhanceTarget(self, last_emb, target_emb): #[b,d],[b,d]

        def compute_sim(last_emb, target_emb):
            fenzi = torch.matmul(last_emb, target_emb.permute(1, 0))  # 512*512
            fenmu_l1 = torch.sum(last_emb * last_emb + 0.000001, 1)
            fenmu_l2 = torch.sum(target_emb * target_emb + 0.000001, 1)
            fenmu_l1 = torch.sqrt(fenmu_l1).unsqueeze(1)
            fenmu_l2 = torch.sqrt(fenmu_l2).unsqueeze(1)
            fenmu = torch.matmul(fenmu_l1, fenmu_l2.permute(1, 0))
            cos_sim = fenzi / fenmu  # 512*512
            cos_sim = nn.Softmax(dim=-1)(cos_sim)
            return cos_sim  # [b,b]

        def compute_pos(batch_size, cos_sim):
            gama = self.gama
            scores = torch.sum(cos_sim, 1)  #[b,]
            value = torch.mean(scores) * gama  #[1,]
            for index in range(batch_size):
                target_emb[index] = torch.where(scores[index] - value > 0, self.linear2_1(torch.cat([target_emb[index], last_emb[index]], 0)),
                                   target_emb[index])

        target_emb = target_emb.squeeze() #[b,2d]
        cos_sim = compute_sim(last_emb, target_emb) #[b,b]
        batch_size = last_emb.shape[0] #b
        mask = trans_to_cuda(torch.Tensor(np.diag([1] * batch_size))) #[b,b]
        scores = cos_sim * mask #[b,b] [0.4,0,0,0,0][0,0.5,0,0,0]
        compute_pos(batch_size, scores)
        up_target = target_emb.unsqueeze(1) #[b,1,d]
        return up_target



    def get_alpha(self, x=None, seq_len=70, number=None):  # x[b,1,d]
        if number == 0:
            alpha_ent = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
            alpha_ent = self.add_value(alpha_ent).unsqueeze(1)
            alpha_ent = alpha_ent.expand(-1, seq_len, -1)  # [b,s+1,1]
            return alpha_ent
        if number == 1:  # x[b,1,d]
            alpha_global = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
            alpha_global = self.add_value(alpha_global)
            return alpha_global


    def add_value(self, value):

        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value

    # q,k,v = [b,s+1,d] ,alpha_ent = [b,s+1,1]
    def self_attention(self, q, k, v, mask=None, alpha_ent=1):
        is_dropout = True
        if is_dropout:
            q_ = self.dropout(self.activate(self.attention_mlp(q)))  # [b,s+1,d]
        else:
            q_ = self.activate(self.attention_mlp(q))
        scores = torch.matmul(q_, k.transpose(1, 2)) / math.sqrt(self.dim)  # [b,s+1,d]x[b,d,s+1] = [b,s+1,s+1]
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
            scores = scores.masked_fill(mask == 0, -np.inf)
        alpha = entmax_bisect(scores, alpha_ent, dim=-1)  # [b,s+1,s+1]

        att_v = torch.matmul(alpha, v)  # B, seq, dim [b,s+1,s+1]x[b,s+1,d] = [b,s+1,d]
        if is_dropout:
            att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
        else:
            att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v
        # att_v = self.LN(att_v) #[b,s+1,d]
        att_v = self.LN(att_v)
        c = att_v[:, -1, :].unsqueeze(1)  # [b,d]->[b,1,d]
        x_n = att_v[:, :-1, :]  # [b,s,d]
        return c, x_n


    def global_attention(self, target, k, v, mask=None, alpha_ent=1):
        alpha = torch.matmul(
            torch.relu(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias),
            # [b,s,d]+[b,1,d]+[d] = [b,s,d]x[d,1] = [b,s,1]
            self.atten_w0.t())  # [b,s,d] x [d,1] = [b,s,1]
        if mask is not None: #[b,s]
            mask = mask.unsqueeze(-1)
            alpha = alpha.masked_fill(mask == 0, -np.inf)
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)  # [b,s,1],[b,1]
        c = torch.matmul(alpha.transpose(1, 2), v)  # (B, 1, dim) = [b,1,s]x[b,s,d] = [b,1,d]
        return c


    # [b,d], [b,d]
    def decoder(self, global_s, target_s):
        if self.is_dropout:
            c = self.dropout(torch.selu(self.w_f(torch.cat((global_s, target_s), 2))))
        else:
            c = torch.selu(self.w_f(torch.cat((global_s, target_s), 2)))  # [b,1,4d]

        c = c.squeeze() #[b,d]
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        return l_c

    def compute_scores(self, hidden, mask, target_emb, att_hidden):
        # ht is local_embedding
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size

        sess_global = torch.sigmoid(q1 + q2) #[b,s,d]

        #target_enhance
        target_emb = self.enhanceTarget(ht, target_emb) #[b,1,2d]

        # Noise-reducing target-aware global attention layer
        alpha_global = self.get_alpha(x=target_emb, number=1) #[b,1,2d]
        q = target_emb
        k = att_hidden #[b,s,2d]
        v = sess_global #[b,s,2d]
        global_c = self.global_attention(q, k, v, mask=mask, alpha_ent=alpha_global)

        sess_final = self.decoder(global_c, target_emb)  # 将global与target送入decoder中进行拼接

        neighbor_sess = self.FindNeighbor(sess_final)

        sess_final = sess_final + self.w_ne * neighbor_sess

        b = self.embedding.weight[1:] / torch.norm(self.embedding.weight[1:], dim=-1).unsqueeze(
            1)
        scores = self.w * torch.matmul(sess_final, b.transpose(1, 0))  # [b,d]x[d,n] = [b,n]
        return scores

    def forward(self, inputs, A, alias_inputs):  # inputs[b,s], A[b,s,s]
        seq_emb = self.add_position_embedding(inputs)  # [b,s,2d]
        hidden = self.gnn(A, seq_emb)  # (b,s,2d)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden_gnn = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  # [b,s,2d]

        # Get Target
        zeros = torch.cuda.FloatTensor(seq_hidden_gnn.shape[0], 1, self.dim).fill_(0)  # [b,1,d]
        session_target = torch.cat([seq_hidden_gnn, zeros], 1)  # [b,s+1,d]

        # Noise-reducing target-aware self-attention layer
        sess_len = session_target.shape[1] #s+1
        alpha_ent = self.get_alpha(x=session_target[:, -1, :], seq_len=sess_len, number=0)  # [b,s+1,1]
        target_emb, x_n = self.self_attention(session_target, session_target, session_target, mask=None,
                                              alpha_ent=alpha_ent)

        return seq_hidden_gnn, target_emb, x_n #[b,s,2d], [b,1,2d], [b,s,2d]


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


# 1
def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden, target_emb, att_hidden = model(items, A, alias_inputs)

    scores = model.compute_scores(hidden, mask, target_emb, att_hidden)

    return targets, scores


# 1
def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss = loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr