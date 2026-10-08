import sys
import os
import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, cur_epoch):
    # test_nodes = getattr(dataCenter, ds+'_test')
    # val_nodes = getattr(dataCenter, ds+'_val')
    all_nodes = getattr(dataCenter, ds + '_all_nodes')
    labels = getattr(dataCenter, ds + '_labels')
    
    adj_list = getattr(dataCenter, ds + '_adj_lists')
    raw_features = torch.FloatTensor(getattr(dataCenter, ds + '_feats')).to(device)
    
    models = [graphSage, classification]
    
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)
    
    embs = graphSage(all_nodes, adj_list, raw_features)
    logists = classification(embs)
    _, predicts = torch.max(logists, 1)
    labels_val = labels[all_nodes]
    assert len(labels_val) == len(predicts)
    comps = zip(labels_val, predicts.data)
    
    vali_f1 = f1_score(labels_val, predicts.cpu().data, average="binary")
    print("Validation F1:", vali_f1)
    
    if vali_f1 > max_vali_f1:
        max_vali_f1 = vali_f1
        # embs = graphSage(test_nodes, adj_list, raw_features)
        # logists = classification(embs)
        # _, predicts = torch.max(logists, 1)
        # labels_test = labels[test_nodes]
        # assert len(labels_test) == len(predicts)
        # comps = zip(labels_test, predicts.data)
        
        # test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
        # print("Test F1:", test_f1)
        
        for param in params:
            param.requires_grad = True
        
        torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, vali_f1))
    
    for param in params:
        param.requires_grad = True
    
    return max_vali_f1, predicts, labels


def evaluate_2(dataCenter, ds, graphSage, graphSage_2, classification, device, max_vali_f1, name, cur_epoch):
    # test_nodes = getattr(dataCenter, ds+'_test')
    # val_nodes = getattr(dataCenter, ds+'_val')
    all_nodes = getattr(dataCenter, ds + '_all_nodes')
    labels = getattr(dataCenter, ds + '_labels')
    
    adj_list = getattr(dataCenter, ds + '_adj_lists')
    adj_list_2 = getattr(dataCenter, ds + '_adj_lists_2')
    raw_features = torch.FloatTensor(getattr(dataCenter, ds + '_feats')).to(device)
    
    models = [graphSage, graphSage_2, classification]
    
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = False
                params.append(param)
    
    embs_1 = graphSage(all_nodes, adj_list, raw_features)
    embs_2 = graphSage_2(all_nodes, adj_list_2, raw_features)
    # embs = torch.cat((embs_1, embs_2), 1)
    embs = torch.add(embs_1, embs_2)
    logists = classification(embs)
    _, predicts = torch.max(logists, 1)
    labels_val = labels[all_nodes]
    assert len(labels_val) == len(predicts)
    comps = zip(labels_val, predicts.data)
    
    vali_f1 = f1_score(labels_val, predicts.cpu().data, average="binary")
    print("Validation F1:", vali_f1)
    
    if vali_f1 > max_vali_f1:
        max_vali_f1 = vali_f1
        # embs = graphSage(test_nodes, adj_list, raw_features)
        # logists = classification(embs)
        # _, predicts = torch.max(logists, 1)
        # labels_test = labels[test_nodes]
        # assert len(labels_test) == len(predicts)
        # comps = zip(labels_test, predicts.data)
        
        # test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
        # print("Test F1:", test_f1)
        
        for param in params:
            param.requires_grad = True
        
        torch.save(models, 'models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, vali_f1))
    
    for param in params:
        param.requires_grad = True
    
    return max_vali_f1, predicts, labels


def get_gnn_embeddings(gnn_model, dataCenter, ds, device):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds + '_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds + '_labels'))).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    adj_list = getattr(dataCenter, ds + '_adj_lists')
    raw_features = torch.FloatTensor(getattr(dataCenter, ds + '_feats'))
    for index in range(batches):
        nodes_batch = nodes[index * b_sz:(index + 1) * b_sz]
        embs_batch = gnn_model(nodes_batch, adj_list, raw_features)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')
    
    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()


def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
    # test_nodes = getattr(dataCenter, ds+'_test')
    # val_nodes = getattr(dataCenter, ds+'_val')
    # train_nodes = getattr(dataCenter, ds+'_train')
    all_nodes = getattr(dataCenter, ds + '_all_nodes')
    labels = getattr(dataCenter, ds + '_labels')
    
    adj_list = getattr(dataCenter, ds + '_adj_lists')
    raw_features = torch.FloatTensor(getattr(dataCenter, ds + '_feats')).to(device)
    
    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        # num_neg = 100
        num_neg = 10
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)
    
    train_nodes = shuffle(all_nodes)
    
    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)
    
    optimizer = torch.optim.SGD(params, lr=0.001)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()
    
    batches = math.ceil(len(train_nodes) / b_sz)
    
    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
        
        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        # if torch.cuda.is_available():
        #     nodes_batch = torch.LongTensor(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg))).to(device)
        # else:
        #     nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)
        
        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]
        
        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # 需要注意：如果采用ABGNN，classification中的embs_batch的维度是两个geraphSage的维度之和，需要更改config文件
        embs_batch = graphSage(nodes_batch, adj_list, raw_features)
        
        if learn_method == 'sup':
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            loss = loss_sup
        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net
        
        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                         len(visited_nodes), len(train_nodes)))
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
    
    return graphSage, classification


def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=800):
    print('Training Classification ...')
    c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.01)
    # train classification, detached from the current graph
    # classification.init_params()
    b_sz = 50
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    features = get_gnn_embeddings(graphSage, dataCenter, ds, device)
    for epoch in range(epochs):
        train_nodes = shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)
        visited_nodes = set()
        for index in range(batches):
            nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
            visited_nodes |= set(nodes_batch)
            labels_batch = labels[nodes_batch]
            embs_batch = features[nodes_batch]
            
            logists = classification(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(nodes_batch)
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))
            
            loss.backward()
            
            nn.utils.clip_grad_norm_(classification.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()
        
        max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
    return classification, max_vali_f1


def apply_model_2(dataCenter, ds, graphSage, graphSage_2, classification, unsupervised_loss, b_sz, unsup_loss, device,
                  learn_method):
    # test_nodes = getattr(dataCenter, ds+'_test')
    # val_nodes = getattr(dataCenter, ds+'_val')
    # train_nodes = getattr(dataCenter, ds+'_train')
    all_nodes = getattr(dataCenter, ds + '_all_nodes')
    labels = getattr(dataCenter, ds + '_labels')
    
    adj_list = getattr(dataCenter, ds + '_adj_lists')
    adj_list_2 = getattr(dataCenter, ds + '_adj_lists_2')
    raw_features = torch.FloatTensor(getattr(dataCenter, ds + '_feats')).to(device)

    weight_array = getattr(dataCenter, ds + '_weights')
    
    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        # num_neg = 100
        num_neg = 10
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)
    
    train_nodes = shuffle(all_nodes)
    
    models = [graphSage, graphSage_2, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)
    
    optimizer = torch.optim.SGD(params, lr=0.001)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()
    
    batches = math.ceil(len(train_nodes) / b_sz)
    
    visited_nodes = set()
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
        
        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        # if torch.cuda.is_available():
        #     nodes_batch = torch.LongTensor(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg))).to(device)
        # else:
        #     nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        # nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)
        
        # get ground-truth for the nodes batch
        labels_batch = labels[nodes_batch]
        weight_array_batch = torch.FloatTensor(weight_array[nodes_batch]).to(device)
        # labels_batch = torch.LongTensor(labels_batch).to(device)
        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        # 需要注意：如果采用ABGNN，classification中的embs_batch的维度是两个geraphSage的维度之和，需要更改config文件
        embs_batch_1 = graphSage(nodes_batch, adj_list, raw_features)
        embs_batch_2 = graphSage_2(nodes_batch, adj_list_2, raw_features)
        # embs_batch = torch.cat((embs_batch_1, embs_batch_2), 1)
        embs_batch = torch.add(embs_batch_1, embs_batch_2)

        
        # embs_batch = torch.add(embs_batch_1, embs_batch_2)
        if learn_method == 'sup':
            logists = classification(embs_batch)

            # loss = F.cross_entropy(logists, labels_batch)
            
            # loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            
            logists = logists[range(logists.size(0)), labels_batch]
            logists_weight = torch.mul(logists, weight_array_batch)
            
            loss_sup = -torch.sum(logists_weight, 0)

            loss_sup /= len(nodes_batch)
            loss = loss_sup
        elif learn_method == 'plus_unsup':
            # superivsed learning
            logists = classification(embs_batch)
            loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss_sup /= len(nodes_batch)
            # unsuperivsed learning
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_sup + loss_net
        else:
            if unsup_loss == 'margin':
                loss_net = unsupervised_loss.get_loss_margin(embs_batch, nodes_batch)
            elif unsup_loss == 'normal':
                loss_net = unsupervised_loss.get_loss_sage(embs_batch, nodes_batch)
            loss = loss_net
        
        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, batches, loss.item(),
                                                                         len(visited_nodes), len(train_nodes)))
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
    
    return graphSage, graphSage_2, classification
