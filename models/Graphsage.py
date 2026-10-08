import torch
import torch.nn as nn
# import math
import torch.nn.functional as F
import random


# from models.BasicModel import BasicModel


class SageLayer(nn.Module):
    '''
    一层SageLayer
    '''
    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()
        
        self.input_size = input_size
        self.out_size = out_size
        
        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))
        # self.weight = nn.Parameter(torch.zeros(size=(2 * input_size, out_size)))
        self.init_params()
    
    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
    
    def forward(self, self_feats, aggregate_feats, neighs=None):
        '''
        
        :param self_feats: 源节点的特征向量
        :param aggregate_feats: 聚合后的邻居特征
        :param neighs:
        :return:
        '''
        if not self.gcn: # 如果聚合层不是gcn，需要concatenate
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined


class Graphsage(nn.Module):
    # def __init__(self, num_layers, input_size, out_size, raw_features, adj_list, gcn=False, agg_func='MEAN'):
    def __init__(self, num_layers, input_size, out_size, device, gcn=False, agg_func='MEAN'):
        super(Graphsage, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.out_size = out_size
        self.gcn = gcn
        self.agg_func = agg_func
        self.device = device
        
        # self.raw_features = raw_features
        # self.adj_list = adj_list
        # 设置每一层输入和输出
        for index in range(1, num_layers+1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))
    
    def forward(self, nodes_bath, adj_list, raw_features):
        '''
        为一批节点生成embedding表示
        :param nodes_bath:
        :return:
        '''
        lower_layer_nodes = list(nodes_bath) # 初始化第一层节点
        nodes_batch_layers = [(lower_layer_nodes,)] # 存放每一层的节点信息
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(lower_layer_nodes, adj_list)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))
        
        assert len(nodes_batch_layers) == self.num_layers + 1
        
        pre_hidden_embs = raw_features # 初始的节点特征h0
        # pre_hidden_embs = torch.LongTensor(raw_features).to(self.device)
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]    # 聚合自己和邻居的节点
            pre_neighs = nodes_batch_layers[index-1]    # 涉及到的所有节点，自己和邻居节点，邻居节点编号->字典中编号
            
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)   # 聚合函数。聚合的节点， 节点特征，集合节点邻居信息
            sage_layer = getattr(self, 'sage_layer'+str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs) # 第一层的batch节点，没有进行转换
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb], aggregate_feats=aggregate_feats) # 进入SageLayer。weight*concat(node,neighbors)
            pre_hidden_embs = cur_hidden_embs
        return pre_hidden_embs
    
    def _get_unique_neighs_list(self, nodes, adj_list, num_sample=10):
        _set = set
        to_neighs = [adj_list[int(node)] for node in nodes]  # 获取目标节点集的所有邻居节点[[v0的邻居],[v1的邻居],[v2的邻居]]
        
        if not num_sample is None: # 如果num_sample为实数的话
            _sample = random.sample
            # 遍历所有邻居集合如果邻居节点数>=num_sample，就从邻居节点集中随机采样num_sample个邻居节点，否则直接把邻居节点集放进去
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        
        # if self.gcn:
        #     samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)] # 把源节点也放进去
        _unique_nodes_list = list(set.union(*samp_neighs)) # 展平 得到这个batch涉及到的所有节点
        i = list(range(len(_unique_nodes_list))) # 建立编号
        unique_nodes = dict(list(zip(_unique_nodes_list, i))) # 节点编号->当前字典中顺序index
        # unique_nodes = {n:i for i, n in enumerate(_unique_nodes_list)}
        return samp_neighs, unique_nodes, _unique_nodes_list # 聚合自己和邻居节点，点的dict，batch涉及到的所有节点
    
    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        '''
        聚合邻居节点信息
        :param nodes: 从最外成开始的节点聚合
        :param pre_hidden_embs: 上一层的节点嵌入
        :param pre_neighs: 上一层的节点
        :param num_sample:
        :return:
        '''
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs # 上一层的源节点，...,....,
        
        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))] # 判断每个节点是否出现在邻居节点中
        assert (False not in indicator)
        if not self.gcn:
            # 如果不适用gcn就要把源节点去除
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) if len(samp_neighs[i])>1 else (samp_neighs[i]) for i in range(len(samp_neighs))]
        
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1 # 每个源节点为一行，一行元素中1对应的就是邻居节点的位置
        
        # if self.cuda():
        #     mask = mask.cuda()
        
        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True) # 计算每个源节点有多少个邻居节点
            mask = mask.div(num_neigh).to(embed_matrix.device)
            # mask = mask.div(num_neigh)
            aggregate_feats = mask.mm(embed_matrix)
        
        elif self.agg_func == 'MAX':
            indexs = [x.nonzero() for x in mask==1]
            aggregate_feats = []
            
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)
        
        return aggregate_feats
    
    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes] # 记录将上一层的节点编号。
        return index

class Classification(nn.Module):
    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()
        self.layer = nn.Sequential(nn.Linear(emb_size, num_classes))
        self.init_params()
        
    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
    
    def forward(self, embeds):
        logists = F.log_softmax(self.layer(embeds), dim=1)
        # logists = self.layer(embeds)
        return logists
    
class UnsupervisedLoss(object):
    def __init__(self, adj_lists, train_nodes, device):
        super(UnsupervisedLoss, self).__init__()
        self.Q = 10
        self.N_WALKS = 6
        self.WALK_LEN = 1
        self.N_WALK_LEN = 5
        self.MARGIN = 3
        self.adj_lists = adj_lists
        self.device = device
        # self.train_nodes = torch.FloatTensor(train_nodes).to(device)
        self.train_nodes = train_nodes
        
        self.target_nodes = None
        self.positive_pairs = []
        self.negtive_pairs = []
        self.node_positive_pairs = {}
        self.node_negtive_pairs = {}
        self.unique_nodes_batch = []
    
    def get_loss_sage(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i] == self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i, n in enumerate(self.node_negtive_pairs)}
        
        nodes_score = []
        assert len(self.node_negtive_pairs) == len(self.node_positive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Q * Exception(negative score)
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
            # print(neg_score)
            # multiple positive score
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))
            # print(pos_score)
            nodes_score.append(torch.mean(- pos_score - neg_score).view(1, -1))

        loss = torch.mean(torch.cat(nodes_score, 0))
        
        return loss

    def get_loss_margin(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i] == self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}
    
        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue
        
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)
        
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)
        
            nodes_score.append(
                torch.max(torch.tensor(0.0).to(self.device), neg_score - pos_score + self.MARGIN).view(1, -1))
            # nodes_score.append((-pos_score - neg_score).view(1,-1))
    
        loss = torch.mean(torch.cat(nodes_score, 0), 0)
    
        # loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))
    
        return loss

    def extend_nodes(self, nodes, num_neg=6):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negtive_pairs = []
        self.node_negtive_pairs = {}
    
        self.target_nodes = nodes
        self.get_positive_nodes(nodes)
        # print(self.positive_pairs)
        self.get_negtive_nodes(nodes, num_neg)
        # print(self.negtive_pairs)
        self.unique_nodes_batch = list(
            set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
        assert set(self.target_nodes) < set(self.unique_nodes_batch)
        return self.unique_nodes_batch

    def get_positive_nodes(self, nodes):
        return self._run_random_walks(nodes)

    def get_negtive_nodes(self, nodes, num_neg):
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= self.adj_lists[int(outer)]
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.train_nodes) - neighbors
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negtive_pairs

    def _run_random_walks(self, nodes):
        for node in nodes:
            if len(self.adj_lists[int(node)]) == 0:
                continue
            cur_pairs = []
            for i in range(self.N_WALKS):
                curr_node = node
                for j in range(self.WALK_LEN):
                    neighs = self.adj_lists[int(curr_node)]
                    next_node = random.choice(list(neighs))
                    # self co-occurrences are useless
                    if next_node != node and next_node in self.train_nodes:
                        self.positive_pairs.append((node, next_node))
                        cur_pairs.append((node, next_node))
                    curr_node = next_node
        
            self.node_positive_pairs[node] = cur_pairs
        return self.positive_pairs