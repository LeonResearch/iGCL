#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
from torch_geometric.datasets import Planetoid,Amazon
import torch_geometric.transforms as T
from model import *
from args_mod import init_args
from loss import *
from utils import *
from torch.nn import LeakyReLU, ReLU, GELU, RReLU, Mish
import gc

from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

def train(args,dataset):
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset in ['Computers','Photo']:
        data.train_mask, data.test_mask, data.val_mask = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    # Build Adjacency Matrix
    adj, adj_ori, adj_label, pos_weight, norm = Build_ADJ(data)
    adj = adj.to(device)
    adj_label = adj_label.to(device)

    
    # ------------------ Initialize Models and Optimizers -------------------
    # Train a VGAE to generate latent distributions
    VGAE = GCNModelVAE(data.x.shape[1],args.hidden_size,args.emb_size,0,args.reconstruct).to(device)
    
    '''
    # ENCODER: Use a 1-layer/2-layer GCN encoder
    if args.num_encoder_layers == 1:
        Encoder = GCN_Encoder_1(nfeat=data.x.shape[1], nemb=args.emb_size, 
                                     dropout=args.dropout).to(device)
    elif args.num_encoder_layers == 2:
        Encoder = GCN_Encoder_2(nfeat=data.x.shape[1], nhid=args.hidden_size, 
                                     nemb=args.emb_size, dropout=args.dropout).to(device)
    Encoder = GAT(data.x.shape[1], args.emb_size, 4).to(device)

    '''

    ## 新encoder hidden size不起作用


    Encoder = EncoderMod(in_channels=data.x.shape[1], out_channels=args.emb_size, \
            base_model=eval(args.base_model), activation=eval(args.activation), \
            k=args.num_encoder_layers, skip=args.enable_encoder_skip, dropout=args.dropout).to(device)


    #Encoder = GAT(data.x.shape[1], args.emb_size, 4).to(device)
    

    if args.proj_head == 'FF':
        Projection_Head = FF(args.emb_size,args.emb_size).to(device)
    elif args.proj_head == 'Linear':
        Projection_Head = nn.Sequential(nn.Linear(args.emb_size, args.emb_size), 
                                        nn.ReLU(), 
                                        nn.Linear(args.emb_size, args.emb_size)).to(device)
    
    # CLASSIFIER: use an MLP as Classifier
    Classifier = Linear_Classifier(args.emb_size,dataset.num_classes).to(device) 
    
    # OPTIMIZER:
    optimizer_encoder = torch.optim.Adam(Encoder.parameters(), lr=args.lr, weight_decay=args.l2)
    optimizer_classifier =  torch.optim.Adam(Classifier.parameters(), lr=args.classifier_lr, weight_decay=args.classifier_l2)
    optimizer_vgae = torch.optim.Adam(VGAE.parameters(), lr=args.vgae_lr, weight_decay=args.vgae_l2)
    
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=10, gamma=args.lr_step_rate)
    
    Encoder_PQ = PriorityQueue(size=args.wa)
    Clser_PQ = PriorityQueue(size=args.wa)

    # Other stuff
    data = data.to(device)
    adj_label = adj_label.to(device)
    val_list = []
    test_list = []
    best_adj_acc = 0
    adj_acc = 0.1
    
    for epoch in range(args.epochs):
        Encoder.train()
        Classifier.train()
        VGAE.train()
        
        pred, mu, log_sigma = VGAE(data.x,adj)
        emb = Encoder(data)
        emb = Projection_Head(emb)

        if args.normalize:
            emb = F.normalize(emb)
        
        #---------------------VGAE Training-----------------------------
        if epoch < args.epochs * 0.5:    
            VGAE_loss = VGAE_Loss(preds=pred, labels=adj_label, mu=mu, logvar=log_sigma,
                                  n_nodes=data.x.shape[0], norm=norm, pos_weight=pos_weight)
            optimizer_vgae.zero_grad()
            VGAE_loss.backward()
            optimizer_vgae.step()

        # --------------------- Contrastive Learning -----------------------     
        # Use latent distributions from the best VGAE so far.
        if adj_acc > best_adj_acc:
            pred, mu, log_sigma = VGAE(data.x,adj)
            mu_best = mu.detach()
            sigma = torch.exp(log_sigma.detach())
            best_adj_acc = adj_acc
        
        # Get the Contrastive Loss 
        Contrastive_loss = Fast_Contrastive_Loss(Z=emb, mu=mu_best, sigma2=sigma**2, 
                                                 tau=args.tau, num_samples=args.num_samples, 
                                                 device = device)
        # Use flooding loss for training
        Flooding_loss = (Contrastive_loss-args.flood).abs()+args.flood
        
        optimizer_encoder.zero_grad()
        Flooding_loss.backward(retain_graph=True)
        optimizer_encoder.step()
        scheduler_encoder.step()
    
        # ----------------------- Classifier Training -----------------------
        # if (epoch+1) % 100 == 0:
        if True:
            Encoder.eval()
            #emb = Encoder(data.x, adj)
            emb = Encoder(data)
            for iteration in range(args.classifier_epochs):
                optimizer_classifier.zero_grad()
                # NOTE: Here we detach the embeddings from computation graph
                y_pred = Classifier(emb.detach())
            
                classifier_loss = F.nll_loss(y_pred[data.train_mask], data.y[data.train_mask])
                classifier_loss.backward()
                optimizer_classifier.step()
        
            #--------------------- Evaluation -------------------------
            Encoder.eval()
            Classifier.eval()
            #emb = Encoder(data.x, adj)
            emb = Encoder(data)
            out = Classifier(emb.detach())
            _, val_pred = out.max(dim=1)
            
            # Get Val Results
            val_acc = pred_result(val_pred, data.y, data.val_mask)
            test_acc = pred_result(val_pred, data.y, data.test_mask)
            val_list.append(val_acc)
            test_list.append(test_acc)
            
            # Adj Acc
            adj_acc = Reconstruct_Ratio(pred.cpu(),adj_ori)

            if not args.tune:
                print('Epoch:{}, Loss:{:.4f}, Adj Acc:{:.4f}, Val Acc:{:.4f}, Test Acc:{:.4f}'\
                        .format(epoch,Contrastive_loss.item(),best_adj_acc,val_acc,test_acc),'\r')
            
            # --------------------- Weight Averaging ----------------------
            #Encoder_PQ.insert((val_acc, gtr2dict(Encoder.named_parameters()), epoch))
            Encoder_PQ.insert((val_acc, Encoder.state_dict(), epoch))
            Clser_PQ.insert((val_acc, Classifier.state_dict(), epoch))

    # ----------------------------- Return Results------------------------
    # Best Val Epoch Results
    con_val_list = np.array(val_list)
    con_test_list = np.array(test_list)
    
    best_epoch = np.where(con_val_list == max(con_val_list))[0][-1]
    test_acc_at_best_val = con_test_list[best_epoch]
    
    # Top k Val Epochs Results after Weight Averaging
    print(f"#"*6+f"Weight Average Result from Top {args.wa} Val Acc Epochs"+"#"*6)
    print(f'Best {args.wa} Val Acc Result:',[_[0] for _ in Encoder_PQ.queue])
    print(f'Best {args.wa} Val Acc Epochs:',[_[2] for _ in Encoder_PQ.queue])
    EPmw = Encoder_PQ.merge_weights()
    Encoder.load_state_dict(EPmw, strict=False)
    Classifier.load_state_dict(Clser_PQ.merge_weights(), strict=False)
        
    Encoder.eval()
    #emb = Encoder(data.x, adj)
    emb = Encoder(data)
    Classifier.eval()
    out = Classifier(emb.detach())
    _, test_pred = out.max(dim=1)
    
    val_acc = pred_result(test_pred, data.y, data.val_mask)
    test_acc = pred_result(test_pred, data.y, data.test_mask)
    
    if not args.tune:
        print(f'Test Acc @ Top {args.wa} Val Epochs',test_acc, f'with Val Acc: {val_acc}')
        print('Test Acc @ best Val Epoch',test_acc_at_best_val,'from Epoch:',best_epoch)
        print('Test Acc @ Last Epoch',test_list[-1])
    return test_acc ,test_acc_at_best_val, test_list[-1], best_adj_acc

def main(args):
    if args.dataset in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(root='data/'+args.dataset, name=args.dataset)
    elif args.dataset in ['Computers','Photo']:
        dataset = Amazon(root='data/Amazon', name=args.dataset, transform=T.NormalizeFeatures())
    print(f'Using {args.dataset} Dataset ...')
    #dataset = Reddit(root='/tmp/'+args.dataset)
    topk_val_result_list = []
    best_val_result_list = []
    last_result_list = []
    adj_acc_list = []
    for seed in range(0,args.repeat):
        set_random_seed(seed)
        
        test_acc_at_topk_val,test_acc_at_best_val,test_acc_at_last,best_adj_acc = train(args,dataset)
        
        torch.cuda.empty_cache()
        gc.collect()

        topk_val_result_list.append(test_acc_at_topk_val)
        best_val_result_list.append(test_acc_at_best_val)
        last_result_list.append(test_acc_at_last)
        adj_acc_list.append(best_adj_acc)
    
    return topk_val_result_list, best_val_result_list, last_result_list, adj_acc_list

if __name__ =='__main__':
    args = init_args()
    topk_val_result_list, best_val_result_list, last_result_list, adj_acc_list = main(args)
    
    mean_topk = np.mean(topk_val_result_list)
    std_topk = np.std(topk_val_result_list)
    
    mean_best = np.mean(best_val_result_list)
    std_best = np.std(best_val_result_list)
    
    mean_last = np.mean(last_result_list)
    std_last = np.std(last_result_list)
    
    mean_adj = np.mean(adj_acc_list)
    std_adj = np.std(adj_acc_list)
    
    print("Test @ topk debug", topk_val_result_list)
    print('Test @ topk Val Acc mean:',mean_topk,'std:',std_topk)
    print("Test @ best debug", best_val_result_list)
    print('Test @ best Val Acc mean:',mean_best,'std:',std_best)
    print("Test @ last debug", last_result_list)
    print('Test @ last Acc mean:',mean_last,'std:',std_last)
    print('Adj Acc mean:',mean_adj,'std:',std_adj)
