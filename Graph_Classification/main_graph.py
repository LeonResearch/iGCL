#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from VGAE import GCNModelVAE_BATCH 
from loss import Implicit_Contrastive_Loss, VGAE_Loss
from utils import Build_ADJ, Reconstruct_Ratio, set_random_seed
from gin import GIN_Encoder,FF
from evaluate_embedding import evaluate_embedding
from omegaconf import OmegaConf

def train(args, dataset, seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(dataset, batch_size=args.batch_size)
    
    if dataset[0].x is not None: # If we have access to the node features
        feat_dim=dataset[0].x.shape[1]
    else: # Use node degree as the feature
        feat_dim=1
    
    # ------------------ Initialize Models and Optimizers -------------------
    # Train a VGAE to generate latent distributions
    VGAE = GCNModelVAE_BATCH(input_feat_dim=feat_dim, hidden_dim1=args.emb_size,
                             hidden_dim2=args.emb_size, dropout=0).to(device)
    # Using the standard GIN Encoder
    Encoder = GIN_Encoder(num_features=feat_dim, dim=args.emb_size, 
                          num_gc_layers=args.num_encoder_layers).to(device)

    if args.proj_head_skip: # If to use the projection_head with skip-connection
        Projection_Head = FF(input_dim=args.emb_size*args.num_encoder_layers, 
                             output_dim=args.emb_size) .to(device)  
    else:
        Projection_Head = nn.Sequential(nn.Linear(args.emb_size*args.num_encoder_layers, args.emb_size), 
                                        nn.ReLU(), 
                                        nn.Linear(args.emb_size, args.emb_size)).to(device)

    # OPTIMIZER:
    optimizer_encoder = torch.optim.Adam(list(Encoder.parameters())+\
            list(Projection_Head.parameters()), lr=args.lr, weight_decay=args.l2)
    optimizer_vgae = torch.optim.Adam(VGAE.parameters(), lr=0.01, weight_decay=0)
    
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder,\
            step_size=10, gamma=args.lr_step_rate)
    # Other stuff
    adj_batch = []
    adj_label_batch = []
    pos_weight_batch = []
    norm_batch = []
    accuracies = {'svc':[]} 
    
    for epoch in range(args.epochs):
        Encoder.train()
        Projection_Head.train()
        VGAE.train()
        loss_all = 0
        adj_acc = 0
        for step, data in enumerate(loader):
            # Build Adjacency Matrix
            if epoch == 0:
                adj, adj_ori, adj_label, pos_weight, norm = Build_ADJ(data)
                adj = adj.to(device)
                adj_label = adj_label.to(device)

                adj_batch.append(adj)
                adj_label_batch.append(adj_label)
                pos_weight_batch.append(pos_weight)
                norm_batch.append(norm)
            
            data = data.to(device)
            if data.x is None:
                data.x = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)
            
            #---------------------VGAE Training-----------------------------
            # Get latent distributions for augmentation
            if epoch < args.epochs*0.5:    
                for iter_vgae in range(10):
                    pred, mu, log_sigma, mu_graph, log_sigma_graph = VGAE(data.x, adj_batch[step], data.batch)
                    
                    
                    VGAE_loss = VGAE_Loss(preds=pred, labels=adj_label_batch[step], mu=mu, \
                                    logvar=log_sigma, n_nodes=data.num_nodes, \
                                    norm=norm_batch[step], pos_weight=pos_weight_batch[step])
                    
                    optimizer_vgae.zero_grad()
                    VGAE_loss.backward()
                    optimizer_vgae.step()
                
            # --------------------- Contrastive Learning -----------------------     
            # Get embeddings from Encoder
            emb, _ = Encoder(data.x, data.edge_index, data.batch)
            emb = Projection_Head(emb) 
            
            # Use latent distributions from the updated VGAE
            pred, mu, log_sigma, mu_graph, log_sigma_graph = VGAE(data.x, adj_batch[step], data.batch)
            mu_best = mu_graph.detach()
            sigma = torch.exp(log_sigma_graph.detach())

            # Get the Contrastive Loss 
            Contrastive_loss = Implicit_Contrastive_Loss(Z=emb, mu=mu_best, sigma2=sigma**2, 
                                                     tau=args.tau, num_samples=None,
                                                     mode='graph', device=device)
            
            optimizer_encoder.zero_grad()
            Contrastive_loss.backward()
            optimizer_encoder.step()
            scheduler_encoder.step()
            
            loss_all += Contrastive_loss.item() * data.num_graphs 
    
        if (epoch+1)%10==0:
            print('Epoch:{}, Loss:{:.4f}'.format(epoch,loss_all/len(loader)))  
        
    # ----------------------- Classifier Training -----------------------
    Encoder.eval()
    Projection_Head.eval()
    
    # Check Reconstruction Ratio 
    VGAE.eval()
    adj_acc = 0
    for step, data in enumerate(loader):
        adj, adj_ori, adj_label, pos_weight, norm = Build_ADJ(data)
        data = data.to(device)
        if data.x is None:
            data.x = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)

        pred, mu, log_sigma, mu_graph, log_sigma_graph = VGAE(data.x, adj_batch[step], data.batch)
        # Adj Acc
        adj_acc_current = Reconstruct_Ratio(pred.cpu(), adj_ori) 
        adj_acc += adj_acc_current * 1/len(loader) 

    emb, y = Encoder.get_embeddings(loader)
    res = evaluate_embedding(emb, y)
    accuracies['svc'].append(res)
    print('Testing Result for Seed:{} is: Adj Acc:{:.4f}, SVC Acc:{:.4f}'\
          .format(seed, adj_acc, res))  
    return accuracies['svc'][-1]


if __name__ == "__main__":
    args = OmegaConf.load(sys.argv[1])

    dataset = TUDataset('data/TUDataset', name = args.dataset)
    print(f'Using {args.dataset} dataset ...') 
    acc_list = []
    for seed in range(0, args.repeat):
        set_random_seed(1)        
        dataset = dataset.shuffle()
        acc = train(args,dataset,seed)
        acc_list.append(acc)
 
    mean_last = np.mean(acc_list)
    std_last = np.std(acc_list)
    
    print('Test @ last Acc mean:',mean_last,'std:',std_last)
