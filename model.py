# File: model.py
# Description: Defines model architecture, loss functions, training and testing procedures

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score
from sklearn import metrics
from anndata import AnnData
import scanpy as sc

# Contrastive loss for embedding space regularization
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, tau=1.0):
        super().__init__()
        self.margin = margin
        self.tau = tau

    def forward(self, z, labels):
        dist_matrix = -torch.exp(torch.matmul(z, z.T) / self.tau)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        positive_pairs = mask * dist_matrix
        negative_pairs = - (1 - mask) * dist_matrix
        loss = (positive_pairs.sum() + negative_pairs.sum()) / z.size(0)
        return loss

# Simple decoder for feature reconstruction
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=[128, 256]):
        super().__init__()
        layers = []
        input_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)

# Main model integrating an encoder, GCN, and decoder
class Model(nn.Module):
    def __init__(self, input_dim=541, hidden_dim=128, loc_dim=2, loc_hidden_dim=128, output_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        self.loc_encoder = nn.Sequential(nn.Linear(loc_dim, loc_hidden_dim))
        self.decoder = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.w_imp = Decoder(hidden_dim, input_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.contrast_loss_fn = ContrastiveLoss()
        self.mae_loss = nn.L1Loss()

    def GCN(self, batch):
        embs = []
        for data in batch:
            x = F.relu(self.conv1(data['x'], data['edge_index']))
            x = self.conv2(x, data['edge_index'])
            embs.append(x.mean(0))
        return torch.stack(embs)

    def forward(self, x, labels, loc, batch_subgraph, test=False):
        z = self.encoder(x)
        node_emb = F.normalize(self.GCN(batch_subgraph), p=2, dim=1)
        x_imp = self.w_imp(node_emb)
        combined = torch.cat((z, node_emb), dim=1)

        if test:
            return F.normalize(z, p=2, dim=1), x_imp

        logits = self.decoder(z)
        contrastive_loss = self.contrast_loss_fn(combined, labels)
        cross_entropy_loss = self.loss_fn(logits, labels)
        return combined, x_imp, contrastive_loss + cross_entropy_loss

# Clustering evaluation metrics

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def cluster_acc(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) / y_pred.size

def evaluate(y_true, y_pred):
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    return acc, 0, nmi, ari, homo, comp, purity
