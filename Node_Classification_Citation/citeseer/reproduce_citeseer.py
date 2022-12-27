#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
from torch_geometric.datasets import Planetoid
from model import *
from utils import *
from framework_GLACMT_WA_mod import train
import gc

def main(config):
    args = my_args(config)
    if args.emb_size >= 128:
        target_util = 0.5
    elif args.hidden_size >=128:
        target_util = 0.5
    else:
        target_util = 0.5

    args.reconstruct = 'A'
    args.num_samples = 500


    dataset = Planetoid(root='./data', name=args.dataset)
    topk_val_result_list = []
    best_val_result_list = []
    last_result_list = []
    adj_acc_list = []
    for seed in range(0,args.repeat):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        test_acc_at_topk_val,test_acc_at_best_val,test_acc_at_last,best_adj_acc = train(args,dataset)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        topk_val_result_list.append(test_acc_at_topk_val)
        best_val_result_list.append(test_acc_at_best_val)
        last_result_list.append(test_acc_at_last)
        adj_acc_list.append(best_adj_acc)
        

    mean_topk = np.mean(topk_val_result_list)
    std_topk = np.std(topk_val_result_list)
    
    mean_best = np.mean(best_val_result_list)
    std_best = np.std(best_val_result_list)
    
    mean_last = np.mean(last_result_list)
    std_last = np.std(last_result_list)
    
    mean_adj = np.mean(adj_acc_list)
    std_adj = np.std(adj_acc_list)

    print('Test @ topk Val Acc mean:',mean_topk,'std:',std_topk)
    
if __name__ =='__main__':
    config =  {'dataset': 'Citeseer', 'epochs': 300, 'classifier_epochs': 100, 'repeat': 10, 'best_k_epochs': 1, 'tune': False, 'classifier': 'MLP', 'training_method': 'Separate', 'num_encoder_layers': 2, 'hidden_size': 256, 'emb_size': 512, 'dropout': 0.5, 'normalize': False, 'wa': 5, 'base_model': 'GATv2Conv', 'activation': 'Mish', 'enable_encoder_skip': False, 'l2': 0.01, 'vgae_l2': 0, 'classifier_l2': 0.0001, 'lr': 0.0005, 'vgae_lr': 0.01, 'classifier_lr': 0.005, 'lr_step_rate': 0.85, 'flood': 0, 'tau': 5.0, 'gamma': 1, 'proj_head': 'Linear'}
    main(config)
