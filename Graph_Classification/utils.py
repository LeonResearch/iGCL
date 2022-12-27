#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import random

def Build_ADJ(data):
    adj = sp.coo_matrix((np.ones(data.num_edges), (data.edge_index[0], data.edge_index[1])),
                        shape=(data.num_nodes, data.num_nodes),
                        dtype=np.float32)
    adj_ori = torch.Tensor(adj.toarray())
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
                
    adj_label = adj_ori + torch.eye(adj_ori.shape[0])
                
    pos_weight = (adj_ori.shape[0]**2-adj_ori.sum())/adj_ori.sum()
    norm = adj_ori.shape[0]**2/(2*(adj_ori.shape[0]**2-adj_ori.sum()))
    return adj, adj_ori, adj_label, pos_weight, norm

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def Reconstruct_Ratio(pred,adj_ori):
    adj_pred = pred.reshape(-1)
    adj_pred = (sigmoid(adj_pred)>0.5).float()
    adj_true = (adj_ori+torch.eye(adj_ori.shape[0])).reshape(-1)
    adj_acc = float(adj_pred.eq(adj_true).sum().item())/adj_pred.shape[0]
    return adj_acc
            
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class my_args:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def pred_result(y_pred, y_true, mask):
        correct  = float(y_pred[mask].eq(y_true[mask]).sum().item())
        acc = correct / mask.sum().item()
        return acc

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
