import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dgl.nn.pytorch.conv import GATConv
import dgl
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer, GATLayer
from src.model import BaseRGCN
from src.decoder import *



class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc,
                                  rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class SdTKGGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times,
                 time_interval, h_dim, opn, history_rate, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False, use_hete_graph=False):
        super(SdTKGGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm  #perform layer normalization in every layer of gcn
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.sin = torch.sin
        self.linear_0 = nn.Linear(num_times, 1)
        self.linear_1 = nn.Linear(num_times, self.h_dim - 1)
        self.tanh = nn.Tanh()
        self.use_cuda = None
        self.use_hete_graph = use_hete_graph

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels * 2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False,
                                                    skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.gat= GATLayer(h_dim, h_dim)

        self.apha=0.8

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        # add
        self.global_weight = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.global_weight, gain=nn.init.calculate_gain('relu'))
        self.global_bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.global_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        self.subgraph_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "timeconvtranse":
            self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder = Contrastive(h_dim)
        else:
            raise NotImplementedError

    def forward(self, g_list, hete_glist, use_cuda):
        gate_list = []
        degree_list = []

        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
        self.h_sub = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
        history_embs = []

        for i, g in enumerate(g_list):

            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(
                self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])

            if self.use_hete_graph:
                if len(hete_glist) != 0:
                    hete_g=hete_glist[i]
                    for key, h_g in hete_g.items():
                        h_g = h_g.to(self.gpu)
                        # node_id = g.ndata['id'].squeeze()
                        # feat = self.h_sub.clone()
                        self.h_sub = torch.squeeze(self.gat.forward(h_g, self.h_sub), dim=1)
                        # 只更新有边的节点
                current_h = self.subgraph_cell_1(current_h, self.h_sub)

            current_h = F.normalize(current_h) if self.layer_norm else current_h

            self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h)
            if torch.isnan(self.h).any():
                print("self.h========NaN detected in the tensor, stopping execution.")
                raise ValueError("NaN detected in the tensor")


        return history_embs, self.h_sub,  self.h_0, gate_list, degree_list

    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        self.use_cuda = use_cuda
        # print(test_triplets.size())
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            evolve_embs, _,  r_emb, _, _ = self.forward(test_graph,  [], use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            time_embs = self.get_init_time(all_triples)
            score_rel = self.rel_raw_mode(embedding, r_emb,  all_triples)
            score = self.raw_mode(embedding, r_emb,  all_triples)
            score_rel = torch.log(score_rel)
            #score = torch.log(score)

            return all_triples, score, score_rel

    def get_loss(self, glist, triples, hete_glist, use_cuda):
        self.use_cuda = use_cuda
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_sub = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        #inverse_triples = triples[:, [2, 1, 0, 3]]
        #inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        #all_triples = torch.cat([triples, inverse_triples])
        all_triples = triples.to(self.gpu)

        evolve_embs, evolve_sub_embs, r_emb, _, _ = self.forward(glist, hete_glist ,use_cuda)
        if torch.isnan(evolve_embs[-1]).any():
            print("NaN detected in the tensor, stopping execution.")
            raise ValueError("NaN detected in the tensor")



        #pre_emb = torch.mul(,evolve_sub_embs)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        #pre_sub_emb = F.normalize(evolve_sub_embs)


        if self.entity_prediction:
            #score = self.decoder(pre_emb,r_emb, pre_g)
            #loss = nn.CrossEntropyLoss(score,all_triples[:,2])
            #loss_ent += loss
            score_en = self.raw_mode(pre_emb, r_emb,  all_triples)
            scores_en = torch.log(score_en)
            loss_ent += F.nll_loss(scores_en, all_triples[:, 2])


        if self.relation_prediction:
            score_rel_r = self.rel_raw_mode(pre_emb, r_emb, all_triples)
            score_re = score_rel_r
            scores_re = torch.log(score_re)
            loss_rel += F.nll_loss(scores_re, all_triples[:, 1])


        if self.use_hete_graph and len(hete_glist) != 0:
            loss_sub = self.decoder.forward(pre_emb,hete_glist)



        return loss_ent, loss_rel, loss_sub

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2


    def raw_mode(self, pre_emb, r_emb,  all_triples):
        scores_ob = self.decoder_ob1.forward_slow(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
        score = F.softmax(scores_ob, dim=1)
        return score

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples):
        score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def rel_raw_mode(self, pre_emb, r_emb,  all_triples):
        scores_re = self.rdecoder_re1.forward(pre_emb, r_emb, all_triples).view(-1, 2 * self.num_rels)
        score = F.softmax(scores_re, dim=1)
        return score

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples):
        score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h







