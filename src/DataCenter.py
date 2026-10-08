import sys
import os

from collections import defaultdict
import numpy as np
import torch

class DataCenter(object):
    def __init__(self, config, cuda, device):
        super(DataCenter, self).__init__()
        self.config = config
        self.cuda = cuda
        self.device = device
        
    def load_dataSet(self, dataSet='cora'):
        if dataSet == 'cora':
            cora_content_file = self.config['file_path.cora_content']
            cora_cite_file = self.config['file_path.cora_cite']
        
            feat_data = []
            labels = []  # label sequence of node
            node_map = {}  # map node to Node_ID
            label_map = {}  # map label to Label_ID
            with open(cora_content_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    feat_data.append([float(x) for x in info[1:-1]])
                    node_map[info[0]] = i
                    if not info[-1] in label_map:
                        label_map[info[-1]] = len(label_map)
                    labels.append(label_map[info[-1]])
            feat_data = np.asarray(feat_data)
            labels = np.asarray(labels, dtype=np.int64)
        
            adj_lists = defaultdict(set)
            with open(cora_cite_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    assert len(info) == 2
                    paper1 = node_map[info[0]]
                    paper2 = node_map[info[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)
        
            assert len(feat_data) == len(labels) == len(adj_lists)
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
        
            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)
        
            setattr(self, dataSet + '_feats', feat_data)
            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_lists', adj_lists)
    
        elif dataSet == 'pubmed':
            pubmed_content_file = self.config['file_path.pubmed_paper']
            pubmed_cite_file = self.config['file_path.pubmed_cites']
        
            feat_data = []
            labels = []  # label sequence of node
            node_map = {}  # map node to Node_ID
            with open(pubmed_content_file) as fp:
                fp.readline()
                feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
                for i, line in enumerate(fp):
                    info = line.split("\t")
                    node_map[info[0]] = i
                    labels.append(int(info[1].split("=")[1]) - 1)
                    tmp_list = np.zeros(len(feat_map) - 2)
                    for word_info in info[2:-1]:
                        word_info = word_info.split("=")
                        tmp_list[feat_map[word_info[0]]] = float(word_info[1])
                    feat_data.append(tmp_list)
        
            feat_data = np.asarray(feat_data)
            labels = np.asarray(labels, dtype=np.int64)
        
            adj_lists = defaultdict(set)
            with open(pubmed_cite_file) as fp:
                fp.readline()
                fp.readline()
                for line in fp:
                    info = line.strip().split("\t")
                    paper1 = node_map[info[1].split(":")[1]]
                    paper2 = node_map[info[-1].split(":")[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)
        
            assert len(feat_data) == len(labels) == len(adj_lists)
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
        
            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)
        
            setattr(self, dataSet + '_feats', feat_data)
            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_lists', adj_lists)
        
        elif dataSet == 'ht':
            ht_path = self.config['file_path.HT_path']
            ht_content_files = []
            ht_cites_files = []
            for _, _, files in os.walk(ht_path):
                for file in files:
                    if os.path.splitext(file)[1] == ".content":
                        ht_content_files.append(file)
                    elif os.path.splitext(file)[1] == ".cites":
                        ht_cites_files.append(file)
        
            # feat_data_all = {}
            # labels_all = {}
            # node_map_all = {}
            # label_map_all = {}
        
            for f in range(len(ht_content_files)):
                feat_data = []
                labels_one = []
                labels_zero = []
                labels = []  # label sequence of node
                node_map = {}  # map node to Node_ID
                label_map = {}  # map label to Label_ID
                with open(ht_path + '/' + ht_content_files[f]) as fp:
                    for i, line in enumerate(fp):
                        info = line.strip().split()
                        feat_data.append([float(x) for x in info[1:-1]])
                        node_map[info[0]] = i
                        if not info[-1] in label_map:
                            label_map[info[-1]] = len(label_map)
                        if label_map[info[-1]] == 0:
                            labels_zero.append(label_map[info[-1]])
                        else:
                            labels_one.append(label_map[info[-1]])
                        labels.append(label_map[info[-1]])
                weight = [ len(labels_zero) / len(labels_one) if labels[i] == 1 else 1 for i in range(len(labels))]
                weight = np.asarray(weight)
                feat_data = np.asarray(feat_data)
                labels = np.asarray(labels, dtype=np.int64)
                
                
                
                adj_lists = defaultdict(set)
                adj_lists_2 = defaultdict(set)
                with open(ht_path + '/' + ht_cites_files[f]) as fp:
                    for i, line in enumerate(fp):
                        info = line.strip().split()
                        assert len(info) == 2
                        u = node_map[info[0]]
                        v = node_map[info[1]]
                        # 结构是u->v，如果是无向图，则添加两条边，如果是有向图，则添加一条边
                        adj_lists[v].add(u)
                        adj_lists_2[u].add(v)
                # 对无向图来说，adj_list中一定包含所有节点；但是对有向图来说，这个结论是不成立的（因为PI没有邻居节点）
                # assert len(feat_data) == len(labels) == len(adj_lists)
                test_indexs, val_indexs, train_indexs, all_nodes = self._split_data(feat_data.shape[0])
                
                if self.cuda:
                    setattr(self, ht_content_files[f].split('.')[0] + '_test', torch.LongTensor(test_indexs).to(device=self.device))
                    setattr(self, ht_content_files[f].split('.')[0] + '_val', torch.LongTensor(val_indexs).to(device=self.device))
                    setattr(self, ht_content_files[f].split('.')[0] + '_train', torch.LongTensor(train_indexs).to(device=self.device))
                    setattr(self, ht_content_files[f].split('.')[0] + '_all_nodes', torch.LongTensor(all_nodes).to(device=self.device))
    
                    setattr(self, ht_content_files[f].split('.')[0] + '_feats', torch.FloatTensor(feat_data).to(device=self.device))
                    setattr(self, ht_content_files[f].split('.')[0] + '_labels', torch.LongTensor(labels).to(device=self.device))
                    setattr(self, ht_content_files[f].split('.')[0] + '_weights',
                            torch.LongTensor(weight).to(device=self.device))
                    # setattr(self, ht_content_files[f].split('.')[0] + '_adj_lists', adj_lists)
                    for k, v in adj_lists.items():
                        # test = torch.FloatTensor(list(v)).to(device=self.device)
                        adj_lists[k] = set(torch.LongTensor(list(v)).to(device=self.device))
                    setattr(self, ht_content_files[f].split('.')[0] + '_adj_lists', adj_lists)
                    for k, v in adj_lists_2.items():
                        # test = torch.FloatTensor(list(v)).to(device=self.device)
                        adj_lists[k] = set(torch.LongTensor(list(v)).to(device=self.device))
                    setattr(self, ht_content_files[f].split('.')[0] + '_adj_lists_2', adj_lists_2)
                else:
                    setattr(self, ht_content_files[f].split('.')[0] + '_test', test_indexs)
                    setattr(self, ht_content_files[f].split('.')[0] + '_val', val_indexs)
                    setattr(self, ht_content_files[f].split('.')[0] + '_train', train_indexs)
                    setattr(self, ht_content_files[f].split('.')[0] + '_all_nodes', all_nodes)
        
                    setattr(self, ht_content_files[f].split('.')[0] + '_feats', feat_data)
                    setattr(self, ht_content_files[f].split('.')[0] + '_labels', labels)
                    setattr(self, ht_content_files[f].split('.')[0] + '_adj_lists', adj_lists)
                    setattr(self, ht_content_files[f].split('.')[0] + '_adj_lists_2', adj_lists_2)
                    setattr(self, ht_content_files[f].split('.')[0] + '_weights', weight)
            
    
    def _split_data(self, num_nodes, test_split=3, val_split=6):
        rand_indices = np.random.permutation(num_nodes)
        
        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)
        
        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size + val_size)]
        train_indexs = rand_indices[(test_size + val_size):]
        
        all_nodes = rand_indices[:]
        
        return test_indexs, val_indexs, train_indexs, all_nodes