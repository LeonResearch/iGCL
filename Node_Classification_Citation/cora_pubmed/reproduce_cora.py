#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.datasets import Planetoid
from model import GCN_Encoder,Linear_Classifier
from VGAE import GCNModelVAE
from cora_args import init_args
from loss import VGAE_Loss, Implicit_Contrastive_Loss
from utils import *

def train(args,dataset):
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Build Adjacency Matrix
    adj, adj_ori, adj_label, pos_weight, norm = Build_ADJ(data)
    adj = adj.to(device)
    adj_label = adj_label.to(device)

    # ------------------ Initialize Models and Optimizers -------------------
    # Train a VGAE to generate latent distributions
    VGAE = GCNModelVAE(data.x.shape[1],args.hidden_size,args.emb_size,0).to(device)
    
    # ENCODER: Use a 2-layer GCN encoder
    Encoder = GCN_Encoder(nfeat=data.x.shape[1], nhid=args.hidden_size, 
                          nemb=args.emb_size, dropout=args.dropout).to(device)
    
    # CLASSIFIER: use Logistic Regression as Classifier
    Classifier = Linear_Classifier(args.emb_size,dataset.num_classes).to(device) 
    
    # OPTIMIZER:
    optimizer_encoder = torch.optim.Adam(Encoder.parameters(), lr=args.lr, weight_decay=args.l2)
    optimizer_classifier =  torch.optim.Adam(Classifier.parameters(), lr=5e-3, weight_decay=5e-4)
    optimizer_vgae = torch.optim.Adam(VGAE.parameters(), lr=0.01, weight_decay=0)
    
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=10, gamma=args.lr_step_rate)
    
    # For weight average
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
        emb = Encoder(data.x,adj)
        
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
        Contrastive_loss = Implicit_Contrastive_Loss(Z=emb, mu=mu_best, sigma2=sigma**2, 
                                                 tau=args.tau, num_samples=args.num_samples, 
                                                 device = device)

        Flooding_loss = (Contrastive_loss-args.flood).abs()+args.flood
        
        optimizer_encoder.zero_grad()
        Flooding_loss.backward(retain_graph=True)
        optimizer_encoder.step()
        scheduler_encoder.step()
    
        # ----------------------- Classifier Training -----------------------
        #Encoder.eval()
        emb = Encoder(data.x,adj)
        for iteration in range(100):
            optimizer_classifier.zero_grad()
            # NOTE: Here we detach the embeddings from computation graph
            y_pred = Classifier(emb.detach())
            
            classifier_loss = F.nll_loss(y_pred[data.train_mask], data.y[data.train_mask])
            classifier_loss.backward()
            optimizer_classifier.step()
        
        #--------------------- Evaluation -------------------------
        Encoder.eval()
        Classifier.eval()
        emb = Encoder(data.x, adj)
        out = Classifier(emb.detach())
        _, val_pred = out.max(dim=1)
        
        # Get Val Results
        val_acc = pred_result(val_pred, data.y, data.val_mask)
        test_acc = pred_result(val_pred, data.y, data.test_mask)
        val_list.append(val_acc)
        test_list.append(test_acc)
        
        # Adj Acc
        adj_acc = Reconstruct_Ratio(pred.cpu(),adj_ori)

        print('Epoch:{}, Loss:{:.4f}, Adj Acc:{:.4f}, Val Acc:{:.4f}, Test Acc:{:.4f}'\
                .format(epoch,Contrastive_loss.item(),best_adj_acc,val_acc,test_acc),'\r')
        
        # --------------------- Weight Averaging ----------------------
        Encoder_PQ.insert((val_acc, gtr2dict(Encoder.named_parameters()), epoch))
        Clser_PQ.insert((val_acc, gtr2dict(Classifier.named_parameters()), epoch))

    # ----------------------------- Return Results------------------------
    # Best Val Epoch Results
    con_val_list = np.array(val_list)
    con_test_list = np.array(test_list)
    
    best_epoch = np.where(con_val_list == max(con_val_list))[0][-1]
    test_acc_at_best_val = con_test_list[best_epoch]
    
    # Top k Val Epochs Results after Weight Averaging
    print(f"#"*6+"Weight Average Result from Top {args.wa} Val Acc Epochs"+"#"*6)
    print(f'Best {args.wa} Val Acc Result:',[_[0] for _ in Encoder_PQ.queue])
    print(f'Best {args.wa} Val Acc Epochs:',[_[2] for _ in Encoder_PQ.queue])
    Encoder.load_state_dict(Encoder_PQ.merge_weights(), strict=False)
    Classifier.load_state_dict(Clser_PQ.merge_weights(), strict=False)
        
    Encoder.eval()
    emb = Encoder(data.x, adj)
    Classifier.eval()
    out = Classifier(emb.detach())
    _, test_pred = out.max(dim=1)
    
    val_acc = pred_result(test_pred, data.y, data.val_mask)
    test_acc = pred_result(test_pred, data.y, data.test_mask)
    
    print(f'Test Acc @ Top {args.wa} Val Epochs',test_acc)
    print('Test Acc @ best Val Epoch',test_acc_at_best_val,'from Epoch:',best_epoch)
    return test_acc, best_adj_acc

def main(args):
    if args.dataset in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(root='data/'+args.dataset, name=args.dataset)
    print(f'Using {args.dataset} Dataset ...')
    topk_val_result_list = []
    adj_acc_list = []
    if args.repeat > 0:
        for seed in range(0,args.repeat):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            test_acc_at_topk_val,best_adj_acc = train(args,dataset)
            
            topk_val_result_list.append(test_acc_at_topk_val)
            adj_acc_list.append(best_adj_acc)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        test_acc_at_topk_val,best_adj_acc = train(args,dataset)
        
        topk_val_result_list.append(test_acc_at_topk_val)
        adj_acc_list.append(best_adj_acc)
    
    return topk_val_result_list, adj_acc_list


if __name__ =='__main__':
    args = init_args()
    topk_val_result_list, adj_acc_list = main(args)
    
    mean_topk = np.mean(topk_val_result_list)
    std_topk = np.std(topk_val_result_list)
    
    mean_adj = np.mean(adj_acc_list)
    std_adj = np.std(adj_acc_list)
    
    print("Test @ topk Results", topk_val_result_list)
    print('Test @ topk Val Acc mean:',mean_topk,'std:',std_topk)
