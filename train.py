import os

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import time
import random
import numpy as np
import scanpy as sc
import anndata
from anndata import AnnData
from sklearn.model_selection import train_test_split
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import sort_edge_index, degree
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment as linear_assignment
import optuna
import json
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import networkx as nx
import pickle


import warnings

warnings.filterwarnings("ignore")


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, loc, subgraphs):
        self.X = X #torch.tensor(X)
        self.y = y
        self.loc = loc
        self.subgraphs = subgraphs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.loc[index], self.subgraphs[index]


def collate_fn(batch):
    batch_x = torch.stack([item[0] for item in batch])
    batch_y = torch.stack([item[1] for item in batch])
    loc = torch.stack([item[2] for item in batch])
    batch_subgraph = [item[3] for item in batch]  # Keep subgraphs as a list
    return batch_x, batch_y, loc, batch_subgraph


def loader_construction(data_name, data_path, batch_size, device='cuda', num_workers=4):
    data = sc.read_h5ad(data_path)
    X_all = data.X.toarray()

    #if data_name=='151507_processed':
    try:
        y_all = np.array(data.obs.loc[:, 'Cluster'])
    except:
        try:
            y_all = np.array(data.obs.loc[:, 'cluster'])
        except:
            try:
                y_all = np.array(data.obs.loc[:, 'region'])
            except:
                y_all = np.array(data.obs.loc[:, 'layer_guess'])


    loc = data.obsm['spatial']



    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_all)
    n_clusters_test = len(np.unique(y_all))+100 #*5



    cluster_label='raw'

    if cluster_label=='leiden':
        sc.pp.normalize_total(data, target_sum=1e4)
        sc.pp.log1p(data)
        sc.pp.scale(data, max_value=10)
        sc.tl.pca(data)
        sc.pp.neighbors(data, n_neighbors=10, n_pcs=25)
        sc.tl.umap(data)
        sc.tl.leiden(data)
        print(data.obs['leiden'].nunique())

        leiden_clusters = data.obs['leiden']

        y_all_pseudo= torch.tensor(leiden_clusters.to_numpy().astype(int)).long().cuda()
        n_clusters = data.obs['leiden'].nunique()+100

    elif cluster_label=='raw':
        try:
            y_all = np.array(data.obs.loc[:, 'layer_guess'])
        except:
            try:
                y_all = np.array(data.obs.loc[:, 'Cluster'])
            except:
                try:
                    y_all = np.array(data.obs.loc[:, 'cluster'])
                except:
                    y_all = np.array(data.obs.loc[:, 'region'])

        y_all = pd.Series(y_all).replace({
            'WM': 0,
            'Layer1': 1,
            'Layer2': 2,
            'Layer3': 3,
            'Layer4': 4,
            'Layer5': 5,
            'Layer6': 6,
            'Layer7': 7,
            'Layer8': 8,
            'Layer9': 9,
            'Layer10': 10,
            'Layer11': 11,
        }).fillna(12).astype(int).to_numpy()

        #     label_encoder = LabelEncoder()
        #     y_all = label_encoder.fit_transform(y_all)
        n_clusters_test = len(np.unique(y_all))+100 #- 1  # *5

        n_clusters=n_clusters_test


    else:
        n_clusters=len(np.unique(y_all))
        kmeans = KMeans(n_clusters=n_clusters_test, random_state=0)
        y_all_pseudo = kmeans.fit_predict(X_all)
        y_all_pseudo = torch.tensor(y_all_pseudo).long().cuda()


    from sklearn.decomposition import PCA  # sklearn PCA is used
    X_all = PCA(n_components=200, random_state=42).fit_transform(X_all)
    #X_all = torch.tensor(X_all).float()

    print(X_all.shape)

    direct_load=False

    if not direct_load:
        degree_threshold = 3 #3
        G = nx.Graph()

        X_norm = X_all / np.linalg.norm(X_all, axis=1, keepdims=True)

        bar = tqdm(range(X_all.shape[0]), desc=f'construct graph')
        for i in range(X_all.shape[0]):
            sim_scores = X_norm[i] @ X_norm.T
            sorted_indices = np.argsort(sim_scores)[-degree_threshold - 1:-1]


            for idx in sorted_indices:
                if idx != i:
                    G.add_edge(i, idx)
            bar.update(1)

        subgraph_data_list = []

        max_neighbors=  20 #20 #10

        for node in tqdm(G.nodes, desc="Preparing subgraphs"):
            # Get the one-hop neighbors
            one_hop_neighbors = list(nx.neighbors(G, node))

            # Get the two-hop neighbors
            two_hop_neighbors = []
            for neighbor in one_hop_neighbors:
                #print(len(list(nx.neighbors(G, neighbor))))
                two_hop_neighbors.extend(list(nx.neighbors(G, neighbor)))


            # Combine the node, one-hop neighbors, and two-hop neighbors
            total_nodes=[node] + random.sample(one_hop_neighbors , min(len(one_hop_neighbors),max_neighbors))
            #total_nodes=total_nodes[:max_neighbors]
            subgraph_nodes = list(set(total_nodes))


            subgraph = G.subgraph(subgraph_nodes)

            #print(len(subgraph_nodes))

            # Map nodes to indices for PyTorch Geometric compatibility
            node_mapping = {n: i for i, n in enumerate(subgraph_nodes)}

            edge_list = [[node_mapping[u], node_mapping[v]] for u, v in subgraph.edges]
            if len(edge_list) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            x = X_all[subgraph_nodes]

            # Create a Data object
            data = {'x': torch.tensor(x).float().cuda(), 'edge_index': edge_index.cuda()}
            subgraph_data_list.append(data)

            bar.update(1)


        # Save the graph and subgraph data
        save_path = "./saved_graph/{}".format(data_name)
        with open(save_path + "_graph_PCA200.pkl", "wb") as f:
            pickle.dump(G, f)

        with open(save_path + "_2-hop-subgraph.pkl", "wb") as f:
            pickle.dump(subgraph_data_list, f)

        print(f"Graph and subgraph data saved in {save_path}")
    else:
        # Load the graph and subgraph data
        save_path = "./saved_graph/{}".format(data_name)

        with open(save_path + "_2-hop-subgraph.pkl", "rb") as f:
            subgraph_data_list = pickle.load(f)


    X_all = torch.tensor(X_all).float()
    y_all = torch.tensor(y_all).long()
    loc = torch.tensor(loc).float()

    #X_all=F.normalize(X_all, p=2, dim=1)



    input_dim = X_all.shape[1]

    use_pseudo_y=False

    # Split the data and subgraph_data_list using a single `train_test_split`
    X_train, X_val, y_train, y_val, loc_train, loc_val, subgraphs_train, subgraphs_val = train_test_split(
        X_all, y_all_pseudo if use_pseudo_y else y_all, loc, subgraph_data_list, test_size=0.2, random_state=1
    )

    X_val, X_test, y_val, y_test, loc_val, loc_test, subgraphs_val, subgraphs_test = train_test_split(
        X_val, y_val, loc_val, subgraphs_val, test_size=0.5, random_state=1
    )




    train_set = CellDataset(X_train.cuda(), y_train.cuda(), loc_train.cuda(), subgraphs_train)
    val_set = CellDataset(X_val.cuda(), y_val.cuda(), loc_val.cuda(), subgraphs_val)
    test_set = CellDataset(X_test, y_test, loc_test, subgraphs_test)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                             collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, input_dim, n_clusters, n_clusters_test


def format_time(seconds):
    if seconds <= 60:
        time_str = '%.1fs' % seconds
    elif seconds <= 3600:
        time_str = '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        time_str = '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    return time_str


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import *


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate(y_true, y_pred):
    acc = cluster_acc(y_true, y_pred)
    f1 = 0
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    return acc, f1, nmi, ari, homo, comp, purity


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, tau=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.tau = tau

    def forward(self, z, labels):
        batch_size = z.size(0)

        # dist_matrix = torch.cdist(z, z, p=2)
        dist_matrix = -torch.exp(torch.matmul(z, z.T) / self.tau)

        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()

        positive_pairs = mask * dist_matrix
        # negative_pairs = (1 - mask) * F.relu(self.margin - dist_matrix)
        negative_pairs = - (1 - mask) * dist_matrix

        loss = positive_pairs.sum() + negative_pairs.sum()
        loss /= batch_size
        return loss


def mixup_same_label(input_tensor, labels):
    # Convert to numpy for easier indexing (optional)
    #input_np = input_tensor.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    unique_labels = list(set(labels_np))
    mixed_inputs = []

    for round in range(5):
        for label in unique_labels:
            # Get all indices with this label
            indices = [i for i, l in enumerate(labels_np) if l == label]

            # Skip if not enough samples to mix
            if len(indices) < 2:
                continue

            # Randomly select two distinct indices
            i1, i2 = random.sample(indices, 2)

            # Random alpha for mix-up
            alpha = random.uniform(0, 1)

            # Perform mixup
            mixed = alpha * input_tensor[i1] + (1 - alpha) * input_tensor[i2]
            mixed_inputs.append((mixed, label))

    if mixed_inputs:
        mixed_data, mixed_labels = zip(*mixed_inputs)
        mixed_data = torch.stack(mixed_data)
        mixed_labels = torch.tensor(mixed_labels, dtype=labels.dtype, device=labels.device)
        return mixed_data, mixed_labels
    else:
        return None, None  # or raise an exception if needed

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=[128, 256]):
        super(Decoder, self).__init__()

        layers = []
        input_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon

class Model(nn.Module):
    def __init__(self, input_dim=541, hidden_dim=128, loc_dim=2, loc_hidden_dim=128, output_dim=10):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        self.loc_encoder = nn.Sequential(
            nn.Linear(loc_dim, loc_hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim , output_dim)
        )

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        #self.w_imp=nn.Linear(hidden_dim, input_dim)
        self.w_imp=Decoder(hidden_dim, input_dim, hidden_dims=[128, 256])

        self.loss_fn = nn.CrossEntropyLoss()

        self.contrast_loss_fn = ContrastiveLoss(margin=1.0)

        self.mae_loss= torch.nn.L1Loss(reduction='mean')
        self.l1loss = torch.nn.L1Loss(reduction='none')

    def GCN(self, batch):
        embs = []
        for data in batch:
            x = self.conv1(data['x'], data['edge_index'])
            x = F.relu(x)
            x = self.conv2(x, data['edge_index'])
            embs.append(x.mean(0))

        return torch.stack(embs, 0)

    def forward(self, x, labels, loc, batch_subgraph, test=False):


        z = self.encoder(x)

        node_emb = self.GCN(batch_subgraph)
        node_emb = F.normalize(node_emb, p=2, dim=1)

        x_imp=self.w_imp(node_emb)

        mask = torch.where(x != 0, torch.ones(x.shape).to(x.device),
                           torch.zeros(x.shape).to(x.device))

        combined = torch.cat((z, node_emb), dim=1)

        if test:
            return F.normalize(z, p=2, dim=1), x_imp

        logits = self.decoder(z)

        # Split the input for contrastive and cross-entropy loss
        split_idx = int(0.1 * x.size(0))
        contrastive_input = combined#[:split_idx]
        contrastive_labels = labels#[:split_idx]
        cross_entropy_input = logits
        cross_entropy_labels = labels#

        contrastive_loss = self.contrast_loss_fn(contrastive_input, contrastive_labels)

        cross_entropy_loss = self.loss_fn(cross_entropy_input, cross_entropy_labels)


        loss=cross_entropy_loss+contrastive_loss
        return combined, x_imp, loss


def train(train_loader,
          valid_loader,
          lr,
          seed,
          epochs,
          n_clusters,
          input_dim,
          save_model_path,
          device='cuda'):
    model = Model(input_dim, output_dim=n_clusters).cuda()

    opt_model = torch.optim.Adam(model.parameters(), lr=lr)

    setup_seed(seed)
    train_loss = []
    valid_loss = []
    best_epoch = 0

    steps = len(train_loader)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    min_loss = 1e6
    patience = 10
    patience_counter = 0

    for each_epoch in range(epochs):

        if patience_counter >= patience:
            print(f"Early stopping at epoch {each_epoch + 1}")
            break

        # print('Epoch %d / %d:' % (each_epoch + 1, epochs))
        batch_loss = []
        model.train()

        with tqdm(total=steps, desc=f'Epoch {each_epoch + 1}/{epochs}', unit='batch') as pbar:
            for step, (batch_x, batch_y, loc, batch_subgraph) in enumerate(train_loader):
                batch_z, x_imp, loss = model(batch_x, batch_y, loc, batch_subgraph)

                opt_model.zero_grad()
                loss.backward()
                opt_model.step()

                batch_loss.append(loss.cpu().detach().numpy())

                # Update tqdm progress bar
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

                end_time = time.time()
                #remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))

                batch_loss.append(loss.cpu().detach().numpy())

        train_loss.append(np.mean(np.array(batch_loss)))


        with torch.no_grad():
            batch_loss = []
            model.eval()

            for step, (batch_x, batch_y, loc, batch_subgraph) in enumerate(valid_loader):
                batch_z, x_imp, loss = model(batch_x, batch_y, loc, batch_subgraph)

                batch_loss.append(loss.cpu().detach().numpy())
    
        valid_loss.append(np.mean(np.array(batch_loss)))
        cur_loss = valid_loss[-1]


        print(cur_loss, min_loss)
        if cur_loss < min_loss or each_epoch==0:
            print('Saving model at Epoch %d with loss %.4f' % (each_epoch, cur_loss))
            min_loss = cur_loss
            best_epoch = each_epoch
            state = {
                'net': model.state_dict(),
                'optimizer': opt_model.state_dict()
            }
            torch.save(state, save_model_path)

            patience_counter = 0
        else:
            patience_counter += 1

    return best_epoch, min_loss


def pearson_corr(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data

    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)

    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
        np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2))
    )
    return corr

# DropData function to mask and preprocess the input data
def DropData(batch_x, d_rate):
    zero_idx = torch.where(batch_x != 0, torch.ones(batch_x.shape).to(batch_x.device), torch.zeros(batch_x.shape).to(batch_x.device))
    batch_x_nonzero = torch.where(batch_x == 0, torch.zeros(batch_x.shape).to(batch_x.device) - 999, batch_x)
    sample_mask = torch.rand(batch_x_nonzero.shape).to(batch_x.device) <= d_rate
    batch_x_drop = torch.where(sample_mask, torch.zeros(batch_x_nonzero.shape).to(batch_x.device), batch_x_nonzero)

    final_mask = torch.where(
        batch_x_drop == 0, torch.ones(batch_x_drop.shape).to(batch_x.device),
        torch.zeros(batch_x_drop.shape).to(batch_x.device) * zero_idx
    )
    final_x = torch.where(batch_x_drop == -999, torch.zeros(batch_x.shape).to(batch_x.device), batch_x_drop)

    return final_mask, final_x

def test(test_loader,
         n_clusters,
         n_clusters_test,
         input_dim,
         save_model_path,
         seed, device='cuda'):
    model = Model(input_dim, output_dim=n_clusters).cuda()
    weights = torch.load(save_model_path)['net']
    model.load_state_dict(weights)
    model.eval()

    z_test = []
    y_test = []
    all_pccs = {0.1: [], 0.2: [], 0.3: []}
    all_loss_maes = {0.1: [], 0.2: [], 0.3: []}
    for step, (batch_x, batch_y, loc, batch_subgraph) in enumerate(test_loader):
        batch_z,x_imp = model(batch_x, batch_y, loc, batch_subgraph, test=True)
        z_test.append(batch_z.cpu().detach().numpy())
        y_test.append(batch_y.cpu().detach().numpy())

        # evaluate for imputation


        for d_rate in [0.1, 0.2, 0.3]:
            final_mask, final_x = DropData(batch_x, d_rate)

            loss_mae = model.mae_loss(final_mask * x_imp, final_mask * batch_x)
            pcc = pearson_corr(
                (final_mask * x_imp).cpu().detach().numpy(),
                (final_mask * batch_x).cpu().detach().numpy()
            )
            all_pccs[d_rate].append(pcc)
            all_loss_maes[d_rate].append(loss_mae.cpu().detach().numpy())

    avg_pccs = {d_rate: float(np.mean(pccs)) for d_rate, pccs in all_pccs.items()}
    avg_loss_maes = {d_rate: float(np.mean(loss_maes)) for d_rate, loss_maes in all_loss_maes.items()}

    z_test = np.vstack(z_test)
    y_test = np.hstack(y_test)



    kmeans = KMeans(n_clusters=n_clusters_test, random_state=seed).fit(z_test) #, n_init=20
    y_kmeans_test = kmeans.labels_



    best_acc=-1
    best_ARI=-1
    best_NMI=-1
    best_homo=-1
    best_comp=-1
    best_purity=-1

    total_results=[]

    for method in ['louvain', 'leiden']:

        for k in [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 50, 100]:
            data = AnnData(z_test)
            # Run clustering
            for n_neighbors in [10]: #10
                #for n_pcs in [10, 25, 50]:
                sc.pp.neighbors(data, n_neighbors=n_neighbors, n_pcs=25) #
                if method == 'leiden':
                    sc.tl.leiden(data, resolution=k)
                else:
                    sc.tl.louvain(data, resolution=k)

                # Extract predicted labels
                leiden_clusters = data.obs[method].astype(int).to_numpy()

                # Evaluate metrics
                acc, f1, nmi, ari, homo, comp, purity = evaluate(y_test, leiden_clusters)

                total_results.append({
                    'method': method,
                        'Resolution': k,
                        'ACC': acc,
                        'ARI': ari,
                        'NMI': nmi,
                        'Purity': purity,
                        'Homogeneity': homo,
                        'Completeness': comp,
                        })


                if nmi > best_NMI:
                    best_NMI = nmi
                if acc > best_acc:
                    best_acc = acc
                if ari > best_ARI:
                    best_ARI = ari
                if homo > best_homo:
                    best_homo=homo
                if comp > best_comp:
                    best_comp=comp
                if purity > best_purity:
                    best_purity=purity

    best_results = {
        'Resolution': k,
        'ACC': best_acc,
        'ARI': best_ARI,
        'NMI': best_NMI,
        'Purity': best_purity,
        'Homogeneity': best_homo,
        'Completeness': best_comp,
    }


    return total_results


# Define paths
data_folder = "./data"
save_results_path = "./result.json"
save_model_folder = "./saved_models"

if not os.path.exists(save_model_folder):
    os.makedirs(save_model_folder)


test_only=False

# Dictionary to store results for all data files
all_results = {}#json.load(open(save_results_path))

for file_name in os.listdir(data_folder):

    if file_name.endswith(".h5ad"):


        data_name = os.path.splitext(file_name)[0]

        if data_name not in ['15_processed_all']:
            continue


        if data_name in all_results:
            continue
        if 'Xenium' in data_name:
            continue

        print('Start Running {}'.format(data_name))

        # Hyperparameters
        epochs = 50 #200 #50  # 20
        batch_size = 20
        # Define paths
        data_path = os.path.join(data_folder, file_name)
        # Load data
        train_loader, val_loader, test_loader, input_dim, n_clusters, n_clusters_test= loader_construction(data_name, data_path,
                                                                                           batch_size)



        for run in range(10):
            seed = run  # Different seed for each run

            start_time = time.time()


            save_model_path = os.path.join(save_model_folder, f"{data_name}_model_run_{run}")


            if not test_only:
                # Train the model
                best_epoch, min_loss = train(train_loader, val_loader, lr=0.0001*10, seed=seed, epochs=epochs,
                                             n_clusters=n_clusters, input_dim=input_dim, save_model_path=save_model_path)

            elapsed_time=time.time() - start_time

            # Save results to the dictionary
            if data_name not in all_results:
                all_results[data_name] = []
            all_results[data_name].append(results)

        # Save all results to a JSON file
        with open(save_results_path, "w") as json_file:
            json.dump(all_results, json_file, indent=4)

        print(f"Results saved to {save_results_path}")
