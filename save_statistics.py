# %% [markdown]
# # Benchmark on OGB, https://ogb.stanford.edu/docs/home/

# %%
import torch_geometric.utils as torch_utils
import importlib
import random
import argparse
import configparser
import numpy as np
import networkx as nx
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch import Tensor
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim

from torch_geometric.utils import negative_sampling, to_networkx
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


import networkx as nx
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import scipy
import math


from dataset_utils import node_feature_utils
from dataset_utils.node_feature_utils import *
import my_utils as utils

importlib.reload(utils)

# %%

class MyIter(object):
    def __init__(self, ite_obj) -> None:
        self.ite_obj = ite_obj
        self.ite = None
        
    def __iter__(self):
        self.ite = iter(self.ite_obj)
        return self.ite
    
    def __next__(self):
        if self.ite is None:
            self.__reset__()
        try:
            res = next(self.ite)
            return res
        except StopIteration as e:
            self.__reset__()
            
        return next(self.ite)
    
    def __reset__(self):
        self.ite = iter(self.ite_obj)
    
    


# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


x_y_label_font = 5
x_y_legend_font = 5
# %%
import pickle as pk
def save_pkl_data(pkl_data, file_name):
    with open(file_name, 'wb') as f:
        pk.dump(pkl_data, f)

# %%
import my_utils as utils
importlib.reload(utils)

import networkx as nx
from functools import reduce
import models
importlib.reload(models)


def graphs_statistics(adjs:list, labels:list):
    
    statistics = []
    for i, A in enumerate(adjs):
        nx_g = nx.from_numpy_array(A)
        avg_cc = nx.average_clustering(nx_g)
        degree_set = node_feature_utils.node_degree_feature(adj=A)
        avg_degree = np.mean(degree_set).item()
        tris = np.mean(node_feature_utils.node_tri_cycles_feature(adj=A)).item()
        cycles = nx.cycle_basis(nx_g)
        N = A.shape[0]
        minD = np.min(degree_set)
        maxD = np.max(degree_set)
        statistics.append((N, avg_cc, avg_degree, tris, cycles, labels[i], minD, maxD))
    
    return statistics

    
    

# Load all datasets:

import sys,os
sys.path.append(os.getcwd())


import my_utils

from PrepareDatasets import DATASETS, DATASETS_used

import dataset_utils
print(DATASETS_used.keys())


group1 = [
    'REDDIT-BINARY',
    'COLLAB',
    'IMDB-BINARY',
    'IMDB-MULTI',
    'NCI1',
    'AIDS',
    'ENZYMES'
]
    
group2 = [
    'PROTEINS',
    'DD',
    'MUTAG',
    'CIFAR10',
    'MNIST'
]

group3 = [
    'ogbg_molhiv',
    'ogbg_ppa',
    'ogbg_moltox21',
    'ogbg-molbace'
]

datasets_obj = {}
for k, v in DATASETS_used.items():
    if k not in group3:
        continue
    print('dataset name:', k)
    dat = v()
    datasets_obj[k] = dat

# %%
# Load specific dataset:

import sys,os
sys.path.append(os.getcwd())


from PrepareDatasets import DATASETS
import my_utils
import dataset_utils

# %%
def is_pyg_dataset(d_name: str):
    return d_name.startswith('ogb') or d_name.startswith('syn')

def get_dense_adjs(dataset, dataset_name):
    adjs = []
    if is_pyg_dataset(dataset_name):
        if hasattr(dataset, 'dataset'):
            for d in dataset.dataset:
                if d.edge_index.numel() < 1:
                    N = d.x.shape[0]
                    adj = np.ones(shape=(N, N))
                else:
                    adj = torch_utils.to_dense_adj(d.edge_index).numpy()[0]
                adjs.append(adj)
        else:
            for d in dataset:
                if d.edge_index.numel() < 1:
                    N = d.x.shape[0]
                    adj = np.ones(shape=(N, N))
                else:
                    adj = torch_utils.to_dense_adj(d.edge_index).numpy()[0]
                adjs.append(adj)
    else:
        # NOTE: not correct, need to be fixed
        if hasattr(dataset, 'dataset'):
            adjs = [d.to_numpy_array() for d in dataset.dataset]
        else:
            adjs = [d.to_numpy_array() for d in dataset]

    return adjs



def get_pyg_dataset_stats(pyg_data, name):
    adjs = []
    # TODO: transform into networkx.
    labels = []

    adjs = get_dense_adjs(pyg_data, name)
    labels = pyg_data.get_labels()
    labels = [y.item() for y in labels]
    # for graph in pyg_data.dataset.get_data():
    #     row = graph.edge_index[0]
    #     col = graph.edge_index[1]
    #     graph.y
    #     N = graph.x.shape[0]
    #     dense_A = torch.zeros((N, N))
    #     dense_A[row, col] = 1
    #     A = dense_A.detach().numpy()
    #     adjs.append(A)
    #     labels.append(graph.y.item())
    print('got statistis done')
    return graphs_statistics(adjs, labels)


# %%


# generate node degree feature:
import dataset_utils.node_feature_utils as nfu
# TODO: check shuffle func:

def shuffle2(data, data_name):
    node_num_total = 0
    node_index = {}
    start_id = 0
    copy_degree_sequence = []
    for i, d in enumerate(data):
        node_num = d.x.shape[0]
        node_num_total += node_num
        for j in range(node_num):
            node_index[start_id] = (i, j)
            start_id += 1
            copy_degree_sequence.append(d.x[j].item())
    
    # dump copy_degree_sequece:
    cds = np.array(copy_degree_sequence)
    print('max csd:', np.max(cds))
    np.save(f'{data_name}_degree_dist.npy', cds)
    print(f'dumped {data_name}_degree_dist.npy!!!')
    
    np.random.shuffle(copy_degree_sequence)
    cds_shuffle = np.array(copy_degree_sequence)
    print('cds_shuffle max 10000 csd:', np.max(cds_shuffle))

    np.save(f'{data_name}_degree_dist_shuffled.npy', cds_shuffle)
    print(f'dumped {data_name}_degree_dist_shuffled.npy!!!')
    
    shuf_idx = list(np.arange(node_num_total))
    pre_value = data[0].x
    sample_ids = [s for s in np.random.choice(shuf_idx, size=node_num_total, replace=True)].__iter__()
    # sample_ids = np.random.randint(1, int(4), size=len(shuf_idx)).__iter__()
    
    for d in data:
        new_x = []
        N = d.x.shape[0]
        for i in range(N):
            new_x.append(copy_degree_sequence[sample_ids.__next__().item()])
            
        d.x = np.array(new_x).reshape(N, 1)
        
def shuffle(data):
    node_num_total = 0
    node_index = {}
    start_id = 0
    for i, d in enumerate(data):
        node_num = d.x.shape[0]
        node_num_total += node_num
        for j in range(node_num):
            node_index[start_id] = (i, j)
            start_id += 1
            
    shuf_idx = list(np.arange(node_num_total))
    np.random.shuffle(shuf_idx)
    np.random.shuffle(shuf_idx)
    np.random.shuffle(shuf_idx)
    # construct pairs
    pairs = []
    for i in range(0, len(shuf_idx), 2):
        if i + 1 < len(shuf_idx):
            pairs.append((shuf_idx[i], shuf_idx[i+1]))

    print(f'shuffle feature!, total len: {node_num_total}, pair len: {len(pairs)}')
    # reconstruct:
    for (p1, p2) in pairs:
        # swich p1 p2 in place
        p1_node, p1_x_id = node_index[p1]
        p2_node, p2_x_id = node_index[p2]
        tmp = data[p1_node].x[p1_x_id]
        data[p1_node].x[p1_x_id] = data[p2_node].x[p2_x_id]
        data[p2_node].x[p2_x_id] = tmp
        
        if p1_node == 3 or p2_node == 3:
            print('node 1:', p1_node, p1_x_id, ' to node2:', p2_node, p2_x_id)
        
        
def plot_st(cur_fea):
    
    max_dd = [s[0] for s in cur_fea]
    min_dd = [s[1] for s in cur_fea]
    mean_dd = [s[2] for s in cur_fea]
    
    plt.figure()
    plt.plot(max_dd)
    plt.plot(min_dd)
    plt.plot(mean_dd)
    plt.ylim(0, 3.1)
    plt.show()
   

import dataset_utils.node_feature_utils as nfu



# %%

if __name__ == '__main__':
        # TODO: plot on one figure:
    datasets_stats = {}
    for k, v in datasets_obj.items():
        d_stats = get_pyg_dataset_stats(v, k)
        print('save stats: ', k)
        save_pkl_data(d_stats, f'{k}_all_datasets_stats.pkl')
        

