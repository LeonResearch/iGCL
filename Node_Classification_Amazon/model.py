#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

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
