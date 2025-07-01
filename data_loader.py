# File: data_loader.py
# Description: Contains data loading, preprocessing, and graph construction logic

import os
import torch
import pickle
import numpy as np
import scanpy as sc
import networkx as nx
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# PyTorch dataset for storing spatial features, labels, locations, and subgraphs
class CellDataset(Dataset):
    def __init__(self, X, y, loc, subgraphs):
        self.X = X
        self.y = y
        self.loc = loc
        self.subgraphs = subgraphs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.loc[index], self.subgraphs[index]

# Collate function for DataLoader to manage batched subgraphs
def collate_fn(batch):
    batch_x = torch.stack([item[0] for item in batch])
    batch_y = torch.stack([item[1] for item in batch])
    loc = torch.stack([item[2] for item in batch])
    batch_subgraph = [item[3] for item in batch]
    return batch_x, batch_y, loc, batch_subgraph

# Main loader function to handle H5AD input, preprocessing, and graph generation
def loader_construction(data_name, data_path, batch_size, device='cuda', num_workers=4):
    data = sc.read_h5ad(data_path)
    X_all = data.X.toarray()

    # Try multiple label sources
    for key in ['Cluster', 'cluster', 'region', 'layer_guess']:
        try:
            y_all = np.array(data.obs.loc[:, key])
            break
        except:
            continue

    loc = data.obsm['spatial']

    # Encode labels
    y_all = pd.Series(y_all).replace({
        'WM': 0, 'Layer1': 1, 'Layer2': 2, 'Layer3': 3, 'Layer4': 4,
        'Layer5': 5, 'Layer6': 6, 'Layer7': 7, 'Layer8': 8,
        'Layer9': 9, 'Layer10': 10, 'Layer11': 11
    }).fillna(12).astype(int).to_numpy()

    n_clusters = len(np.unique(y_all))
    n_clusters_test = n_clusters + 100

    # Dimensionality reduction
    X_all = PCA(n_components=200, random_state=42).fit_transform(X_all)

    # Build similarity graph using cosine similarity
    degree_threshold = 3
    G = nx.Graph()
    X_norm = X_all / np.linalg.norm(X_all, axis=1, keepdims=True)

    for i in tqdm(range(X_all.shape[0]), desc='Constructing graph'):
        sim_scores = X_norm[i] @ X_norm.T
        neighbors = np.argsort(sim_scores)[-degree_threshold - 1:-1]
        for idx in neighbors:
            if idx != i:
                G.add_edge(i, idx)

    # Extract subgraphs (2-hop neighborhood)
    subgraph_data_list = []
    max_neighbors = 20
    for node in tqdm(G.nodes, desc="Preparing subgraphs"):
        one_hop = list(nx.neighbors(G, node))
        total_nodes = [node] + random.sample(one_hop, min(len(one_hop), max_neighbors))
        subgraph_nodes = list(set(total_nodes))
        subgraph = G.subgraph(subgraph_nodes)

        node_map = {n: i for i, n in enumerate(subgraph_nodes)}
        edge_list = [[node_map[u], node_map[v]] for u, v in subgraph.edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

        x = X_all[subgraph_nodes]
        subgraph_data_list.append({
            'x': torch.tensor(x).float().to(device),
            'edge_index': edge_index.to(device)
        })

    # Save graph
    save_path = f"./saved_graph/{data_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(f"{save_path}_graph_PCA200.pkl", "wb") as f:
        pickle.dump(G, f)
    with open(f"{save_path}_2-hop-subgraph.pkl", "wb") as f:
        pickle.dump(subgraph_data_list, f)

    # Convert arrays to tensors
    X_all = torch.tensor(X_all).float()
    y_all = torch.tensor(y_all).long()
    loc = torch.tensor(loc).float()

    # Split into train, val, test
    X_train, X_val, y_train, y_val, loc_train, loc_val, sub_train, sub_val = train_test_split(
        X_all, y_all, loc, subgraph_data_list, test_size=0.2, random_state=1
    )
    X_val, X_test, y_val, y_test, loc_val, loc_test, sub_val, sub_test = train_test_split(
        X_val, y_val, loc_val, sub_val, test_size=0.5, random_state=1
    )

    # Create dataset and dataloaders
    train_set = CellDataset(X_train.to(device), y_train.to(device), loc_train.to(device), sub_train)
    val_set = CellDataset(X_val.to(device), y_val.to(device), loc_val.to(device), sub_val)
    test_set = CellDataset(X_test, y_test, loc_test, sub_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    input_dim = X_all.shape[1]
    return train_loader, val_loader, test_loader, input_dim, n_clusters, n_clusters_test
