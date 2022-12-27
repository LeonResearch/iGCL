#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from layers import GraphConvolution

class GCNModelVAE_BATCH(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE_BATCH, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2)
        self.dropout = dropout
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        
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

    def forward(self, x, adj,batch):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        mu_graph = global_mean_pool(mu, batch)
        logvar_graph = global_mean_pool(logvar, batch)
        pred = self.dc(z)
        
        return pred, mu, logvar, mu_graph, logvar_graph

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
