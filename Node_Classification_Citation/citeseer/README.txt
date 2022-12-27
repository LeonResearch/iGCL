train_CL: 2-layer GCN Encoder -> mu,sigma; 1-layer MLP Classifier; Cov from class
train_cov: 1-layer GCN Encoder -> emb; 1-layer GCN Classifier; Cov from class
train_variational: 2-layer GCN Encoder -> mu,sigma; 1-layer GCN Classifier; Cov from individual

IMPORTANT
1. Do not use discriminative + cov_by_class
2. Do not use KNN Classifier + Combined
