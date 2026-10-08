import sys
import os
import torch
import argparse
import pyhocon
import random
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from src.DataCenter import *
from models.Graphsage import *
from src.utils import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='ht')
parser.add_argument('--agg_func', type=str, default='MAX')
parser.add_argument('--epochs', type=int, default=63)
parser.add_argument('--b_sz', type=int, default=64)
parser.add_argument('--seed', type=int, default=636)
parser.add_argument('--cuda', default=False, help='use CUDA')
parser.add_argument('--gcn', default=False)
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='./src/experiments.conf')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

# device = torch.device("cuda" if args.cuda else "cpu")
device = torch.device("cuda")
print('DEVICE:', device)

def result(y_test, y_pre, result_path, net_dt):
    #  print(testname)
    # 混淆矩阵
    conf_mat = confusion_matrix(y_test, y_pre)
    print('混淆矩阵：')
    print(conf_mat)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pre).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    prec = tp / (tp + fp)
    f1 = 2 / (1 / prec + 1 / tpr)

    with open(result_path + "\\" + net_dt + ".txt", "a", encoding="utf-8") as f:
        f.write('TPR:' + str(tpr) + '\n')
        f.write('TNR:' + str(tnr) + '\n')
        f.write('F1:' + str(f1) + '\n')
        f.write(classification_report(y_test, y_pre, digits=4))
    print('TPR:', tpr)
    print('TNR:', tnr)
    print('f-score:', f1)

    # 分类指标文本报告（精确率、召回率、F1值等）
    print('分类指标报告：')
    print(classification_report(y_test, y_pre, digits=4))

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)

    # load data
    ds = args.dataSet
    dataCenter = DataCenter(config, args.cuda, device)
    dataCenter.load_dataSet(ds)
    
    graphSAGE = Graphsage(config['setting.num_layers'], config['setting.feature_size'], config['setting.hidden_emb_size'], device)
    graphSAGE.to(device)

    graphSAGE_2 = Graphsage(config['setting.num_layers'], config['setting.feature_size'],
                          config['setting.hidden_emb_size'], device)
    graphSAGE_2.to(device)
    
    classification = Classification(config['setting.hidden_emb_size_after_graphsage'], config['setting.num_labels'])
    classification.to(device)

    ht_path = config['file_path.HT_path']
    result_path = config['file_path.result_path']
    ht_file_name = []
    for _, _, files in os.walk(ht_path):
        for file in files:
            if os.path.splitext(file)[1] == ".content":
                ht_file_name.append(os.path.splitext(file)[0])
    
    for val_name in ht_file_name:
        print('---------------------------FILE %s-------------------------' % val_name)
        for epoch in range(args.epochs):
            print('----------------------EPOCH %d-----------------------' % epoch)
            for train_name in ht_file_name:
                if train_name == val_name:
                    continue
                else:
                    unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, train_name + '_adj_lists'),
                                                         getattr(dataCenter, train_name + '_train'), device)
                    graphSAGE, graphSAGE_2, classification = apply_model_2(dataCenter, train_name, graphSAGE, graphSAGE_2, classification,
                                                            unsupervised_loss, args.b_sz, args.unsup_loss, device,
                                                            args.learn_method)
                    # graphSAGE, classification = apply_model(dataCenter, train_name, graphSAGE,
                    #                                                        classification,
                    #                                                        unsupervised_loss, args.b_sz,
                    #                                                        args.unsup_loss, device,
                    #                                                        args.learn_method)
                    # if (epoch + 1) % 2 == 0 and args.learn_method == 'unsup':
                    #     classification, args.max_vali_f1 = train_classification(dataCenter, graphSAGE, classification,
                    #                                                             train_name, device, args.max_vali_f1, args.name)
            if args.learn_method != 'unsup':
                args.max_vali_f1, predicts, labels = evaluate_2(dataCenter, val_name, graphSAGE, graphSAGE_2, classification, device, args.max_vali_f1,
                                                    args.name, epoch)
                # args.max_vali_f1, predicts, labels = evaluate(dataCenter, val_name, graphSAGE,
                #                                                 classification, device, args.max_vali_f1,
                #                                                 args.name, epoch)
            
            result(labels, predicts.cpu().data, result_path, val_name)