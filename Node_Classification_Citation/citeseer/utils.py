#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from torch.utils.data import random_split
import random

def mask_nodes():
    train_index = []
    test_index = []
    for j in range(lable.max().item() + 1):
        # num = ((lable == j) + 0).sum().item()
        index = torch.range(0, len(lable) - 1)[(lable == j).squeeze()]
        x_list0 = random.sample(list(index), int(len(index) * 0.1))
        for x in x_list0:
            train_index.append(int(x))
    for c in range(len(lable)):
        if int(c) not in train_index:
            test_index.append(int(c))
    train_lbls = lable[train_index].squeeze()
    test_lbls = lable[test_index]
    val_lbls = lable[train_index]
    return train_mask, test_mask

def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask

def sample_z(mu, logvar):
    """Reparameterisation trick."""
    #torch.manual_seed(seed)
    std = torch.exp(logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

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

def Reconstruct_Ratio(pred,adj_ori):
    adj_pred = pred.reshape(-1)
    adj_pred = (sigmoid(adj_pred)>0.5).float()
    adj_true = (adj_ori+torch.eye(adj_ori.shape[0])).reshape(-1)
    adj_acc = float(adj_pred.eq(adj_true).sum().item())/adj_pred.shape[0]
    return adj_acc

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

class PriorityQueue(object):
    def __init__(self,size):
        self.queue = []
        self.weights = []
        self._size = size
    def __str__(self):
        return ' '.join([str(i) for i in self.queue])
 
    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0
 
    # for inserting an element in the queue
    def insert(self, info_tuple):
        assert len(self.queue) <= self._size
        self.queue.append(info_tuple)
        try:
            self.queue = sorted(self.queue, key=lambda s:-1*s[0])
        except:
            __import__('pdb').set_trace()
        if len(self.queue) > self._size:
            self.queue.pop(-1)
    def merge_weights(self,):
        assert len(self.queue) > 0
        if len(self.queue) == 1:
            return self.queue[0][1]
        params_dict = defaultdict(list)
        for _, weight, _ in self.queue:
            for k,v in weight.items():
                params_dict[k].append(v)
        avged_params_dict = dict()
        for k,v in params_dict.items():
            avg_v = torch.stack(v, dim=0)
            avged_params_dict[k] = avg_v.mean(dim=0)
        return avged_params_dict

def gtr2dict(param_gtr):
    return {k:v.data.clone() for k,v in param_gtr}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

