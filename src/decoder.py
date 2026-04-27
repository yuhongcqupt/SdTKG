from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
import math
import os
import dgl
path_dir = os.getcwd()

class TimeConvTransR(torch.nn.Module):
    def __init__(self, num_relations, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):
        super(TimeConvTransR, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(4, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.conv2 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                                     padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(4)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(2)
        self.register_parameter('b', Parameter(torch.zeros(num_relations*2)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)



    def forward(self, embedding, r_rel, triplets, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):

        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)

        stacked_inputs = torch.cat([e1_embedded, e2_embedded], 1)
        stacked_inputs = self.bn3(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if partial_embeding is None:
            x = torch.mm(x, r_rel.transpose(1, 0))
        else:
            x = torch.mm(x, r_rel.transpose(1, 0))
            x = torch.mul(x, partial_embeding)
        return x

    def forward_path(self, embedding, emb_rel, path, num_path, num_rels, test=0, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        e1_embedded_all = F.tanh(embedding)
        batch_size = len(path)
        if test==0:
            #path:[rule_body_id, rule_head_id]
            r=path[:,0]-num_rels
            e1_embedded = e1_embedded_all[r].unsqueeze(1)
            e2_embedded = e1_embedded_all[r + num_path].unsqueeze(1)
        else:
            list_of_tensor = torch.chunk(e1_embedded_all, dim=0, chunks=2)
            e1_embedded = list_of_tensor[0].unsqueeze(1)
            e2_embedded = list_of_tensor[1].unsqueeze(1)
            batch_size=num_path
        stacked_inputs = torch.cat([e1_embedded, e2_embedded], 1)
        stacked_inputs = self.bn3(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if partial_embeding is None:
            x = torch.mm(x, emb_rel.transpose(1, 0))
        else:
            x = torch.mm(x, emb_rel.transpose(1, 0))
            x = torch.mul(x, partial_embeding)
        return x


class TimeConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(TimeConvTransE, self).__init__()

        self.num_path=num_entities
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(4, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))
        self.conv2 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                                     padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(4)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(2)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, emb_time, triplets, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        #print(embedding.shape)
        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)  # batch_size,1,h_dim
        #print(emb_rel.shape)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)  # batch_size,1,h_dim
        emb_time_1, emb_time_2 = emb_time
        emb_time_1 = emb_time_1.unsqueeze(1)
        emb_time_2 = emb_time_2.unsqueeze(1)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded, emb_time_1, emb_time_2], 1)  # batch_size,2,h_dim
        stacked_inputs = self.bn0(stacked_inputs)  # batch_size,2,h_dim
        x = self.inp_drop(stacked_inputs)  # batch_size,2,h_dim
        x = self.conv1(x)  # batch_size,2,h_dim
        x = self.bn1(x)  # batch_size,channels,h_dim
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)  # batch_size,channels*h_dim
        x = self.fc(x)  # batch_size,channels*h_dim
        x = self.hidden_drop(x)  # batch_size,h_dim
        #print(batch_size)
        #print(x.shape)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        #print(x.shape)
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        else:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
            x = torch.mul(x, partial_embeding)
        #print(x.shape)
        return x

    def forward_slow(self, embedding, emb_rel, triplets):

        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn3(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))

        return x

    def forward_path(self, embedding, emb_rel, triplets):

        e1_embedded_all = F.tanh(embedding[0:self.num_path])
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 1]-230].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn3(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        return x


class Decode(torch.nn.Module):
    def __init__(self, path_dict,num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3):
        super(Decode, self).__init__()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1, padding=4)
        self.conv2 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,padding=int(math.floor(kernel_size / 2)))
        self.conv3 = torch.nn.Conv1d(3, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)))
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.bn0 = torch.nn.BatchNorm1d(4)

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.num_nodes=num_entities
        self.bn0 = torch.nn.BatchNorm1d(4)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(2)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

        self.path_dict=path_dict

    def forward(self,pre_emb,r_embed,g):
        #g.ndata['h']=pre_emb
        #g.apply_nodes(lambda nodes:{'h':pre_emb[nodes.data['id']]})
        self.propagate(g,r_embed,pre_emb)
        return g.edata['msg'],g.edate['o']

    def propagate(self,g,r_emb,pre_emb):
        g.apply_edges(lambda x: self.msg_func(x,r_emb,pre_emb))

    def msg_func(self,edges,r_emb,pre_emb):

        src = pre_emb[edges.src['id']]
        p = edges.data['path']
        o = edges.data['o']

        p1=[]
        p2=[]
        p3=[]
        for i in p:
            p11 = []
            p12 = []
            p13 = []
            for j in i:
                if j.item()==-1:
                    break
                path=eval(self.path_dict[str(j.item())])
                if len(path)==1:
                    p11.append(path)
                elif len(path)==2:
                    p12.append(path)
                else:
                    p13.append(path)
            p1.append(p11)
            p2.append(p12)
            p3.append(p13)
        p_embed=self.decode(src,r_emb,o,p1,p2,p3)
        r_embed=self.cov(torch.cat(src,r_emb[edges['type']]),2)
        cos_sim = [F.cosine_similarity(feature, r_embed, dim=0) for feature in p_embed]

        s = torch.zeros((len(edges),self.num_nodes))

        for j in range(len(edges)):
            # 使用 torch.tensor 将特征列表转换为张量，确保数据类型一致
            feature_tensor = torch.tensor(cos_sim[j], dtype=torch.float32)
            # 使用 torch.tensor 将索引列表转换为张量，确保数据类型一致
            index_tensor = torch.tensor(o[j])
            # 使用 torch.scatter 将特征值散布到张量 s 的指定位置上
            s[j].scatter_(0, index_tensor, feature_tensor)
        return {'scores':s}

    def decode(self,s,r,o,p1,p2,p3):
        rel_embed = r
        o_len=len(o)
        x1=None
        x2=None
        x3=None
        for i in range(len(p1)):
            if len(p1[i])!=0:
                s_embed = torch.tile(s, (len(p1[i]), 1))
                #print(s_embed.size())
                #print(p1)
                #print(len(p1[i]))
                stacked_inputs1=torch.cat([s_embed,rel_embed[p1[i]]], 1)
                x1 = self.cov(stacked_inputs1, 2, o_len)
            if len(p2[i])!=0:
                s_embed = torch.tile(s, (len(p2[i]), 1))
                stacked_inputs2 = torch.cat([s_embed, rel_embed[p2[i][0]], rel_embed[p2[i][1]]], 1)
                x2 = self.cov(stacked_inputs2, 3, o_len)
            if len(p3[i])!=0:
                s_embed = torch.tile(s, (len(p3[i]), 1))
                stacked_inputs3 = torch.cat([s_embed, rel_embed[p3[i][0]], rel_embed[p3[i][1]], rel_embed[p3[i][2]]], 1)
                x3 = self.cov(stacked_inputs3, 4, o_len)
            x = [x1, x2, x3]

            valid_tensors = [i for i in x if i is not None]
            combined_tensor = torch.cat(valid_tensors, dim=0)  # 假设在第一维度上进行合并

        return combined_tensor

    def cov(self,stacked_inputs,dim,batch_size):
        x = self.inp_drop(stacked_inputs)  # batch_size,2,h_dim
        if dim == 2:
            x = self.conv1(x)  # batch_size,2,h_dim
        elif dim == 4:
            x = self.conv2(x)
        elif dim == 3:
            x = self.conv3(x)
        x = self.bn1(x)  # batch_size,channels,h_dim
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x



class Contrastive(torch.nn.Module):

    def __init__(self,embedding_dim):
        """ init additional BN used after head """
        super(Contrastive, self).__init__()
        self.bn_last = torch.nn.BatchNorm1d(embedding_dim)


    def bt_loss(self,h1, h, lambda_, batch_norm=True, eps=1e-15, *args, **kwargs):
        batch_size = h.size(0)
        feature_dim = h.size(1)
        loss=0
        if lambda_ is None:
            lambda_ = 1. / feature_dim

        for h2 in h:
            if batch_norm:
                z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
                z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
                c = (z1_norm.T @ z2_norm) / batch_size
            else:
                c = h1.T @ h2 / batch_size
            loss += lambda_ * c.pow(2)

        return loss


    def forward(self,pre_emb,g):
        loss = 0
        for i, g in enumerate(g):
            for key, h_g in g.items():
                for s in h_g.point_set:
                    #print(len(h_g.successors(s)))
                    loss += self.bt_loss(pre_emb[s],pre_emb[h_g.successors(s)],None)
        loss = loss/len(g)
        return loss