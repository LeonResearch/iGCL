#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from layers import GraphConvolution

class EncoderMod(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False, dropout=0.5):
        super(EncoderMod, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            #~self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            self.conv = [base_model(in_channels, 2 * out_channels)]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation()
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation()
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        if not self.skip:
            for i in range(self.k):
                x = F.dropout(self.activation(self.conv[i](x, edge_index)), self.dropout, training=self.training)
            return x
        else:
            h = F.dropout(self.activation(self.conv[0](x, edge_index)), self.dropout, training=self.training)
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(F.dropout(self.activation(self.conv[i](u, edge_index)), self.dropout, training=self.training))
            return hs[-1]

class GCN_Encoder_1(nn.Module):
    def __init__(self, nfeat, nemb, dropout):
        super(GCN_Encoder_1, self).__init__()

        self.gc = GraphConvolution(nfeat, nemb)

    def forward(self, x, adj):
        x = self.gc(x, adj)
        return x

class GCN_Encoder_2(nn.Module):
    def __init__(self, nfeat, nhid, nemb, dropout):
        super(GCN_Encoder_2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nemb)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class Linear_Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Linear_Classifier, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        #seq = F.relu(seq)
        ret = self.fc(seq)
        return F.log_softmax(ret,dim=1)

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, reconstruct_type):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2)
        self.dropout = dropout
        self.reconstruct_type = reconstruct_type
        if reconstruct_type == 'A':
            self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        elif reconstruct_type == 'X':
            self.dc = GCNDecoder(hidden_dim2, hidden_dim1, input_feat_dim, dropout)
        else:
           raise Exception("Specify the Reconstruction Type as A or X") 
        
    def encode(self, x, adj):
        hidden1 = F.relu(self.gc1(x, adj))
        hidden1 = F.dropout(hidden1,self.dropout,training=self.training)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        if self.reconstruct_type == 'X':
            pred = self.dc(z,adj)
        else:
            pred = self.dc(z)
        return pred, mu, logvar

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class MLPDecoder(nn.Module):
    def __init__(self, n_emb, n_hidden, n_feature, dropout):
        super(MLPDecoder, self).__init__()
        self.dc1 = nn.Linear(n_emb, n_hidden)
        self.dc2 = nn.Linear(n_hidden, n_feature)
        self.dropout = dropout

    def forward(self, z):
        z = F.relu(self.dc1(z))
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.dc2(z)
        return z

class GCNDecoder(nn.Module):
    def __init__(self, n_emb, n_hidden, n_feature, dropout):
        super(GCNDecoder, self).__init__()
        self.dc1 = GraphConvolution(n_emb, n_hidden)
        self.dc2 = GraphConvolution(n_hidden, n_feature)
        self.dropout = dropout

    def forward(self, z, adj):
        z = F.relu(self.dc1(z,adj))
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.dc2(z,adj)
        return z

class FF(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class GAT(torch.nn.Module):
    def __init__(self,nfeat,nhid,nhead):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels=nfeat, 
                             out_channels=nhid, 
                             heads=nhead, 
                             dropout=0.6)
        self.conv2 = GATConv(in_channels=nhid * nhead, 
                             out_channels=nhid, 
                             concat=False,
                             heads=1, 
                             dropout=0.6)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        #return F.log_softmax(x, dim=1)
        return x
