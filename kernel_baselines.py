import importlib
import numpy as np
import networkx as nx

# load splits as datasets:
import torch_geometric.utils as torch_utils
import torch

import networkx as nx


from dataset_utils.node_feature_utils import *


# Load specific dataset:

import sys,os
sys.path.append(os.getcwd())


from PrepareDatasets import DATASETS


print(DATASETS.keys())
"""
DATASETS = {
    'REDDIT-BINARY': RedditBinary,
    'REDDIT-MULTI-5K': Reddit5K,
    'COLLAB': Collab,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-MULTI': IMDBMulti,
    'NCI1': NCI1,
    'AIDS': AIDS,
    'ENZYMES': Enzymes,
    'PROTEINS': Proteins,
    'DD': DD,
    "MUTAG": Mutag,
    'CSL': CSL,
    'CIFAR10': CIFAR10,
    'MNIST': MNIST,
    'PPI': PPI,
    'hiv': HIV,
    'bace':BACE,
    'bbpb':BBPB,
    'ogbg_molhiv':OGBHIV,
    'ogbg_ppa':OGBPPA
}
"""

data_names = ['PROTEINS']
data_names = ['DD']
data_names = ['ENZYMES']
data_names = ['NCI1']
data_names = ['IMDB-MULTI']
data_names = ['CIFAR10']
data_names = ['ogbg_molhiv']

# NOTE:new kernel:
data_names = ['MUTAG']

data_names = ['REDDIT-BINARY', 'COLLAB', 'IMDB-BINARY','IMDB-MULTI']

data_names = ['REDDIT-BINARY', 'COLLAB']

data_names = ['PROTEINS', 'ENZYMES', 'NCI1', 'DD', 'MUTAG']
data_names = []

datasets_obj = {}


def get_each_folder_syn(dataset, fold_id, batch_size=1):
    
    fold_test = dataset.get_test_fold(fold_id, batch_size=batch_size, shuffle=True).dataset
    fold_train, fold_val = dataset.get_model_selection_fold(fold_id, inner_idx=None,
                                                                          batch_size=batch_size, shuffle=True)
    
    
    
    fold_train = fold_train.dataset
    fold_val = fold_val.dataset
    
    # train_G = [pyg_utils.to_networkx(d, node_attrs=['x']) for d in fold_train.get_subset()]
    # test_G = [pyg_utils.to_networkx(d, node_attrs=['x']) for d in fold_test.get_subset()]
    # print('x: ',train_G[0].nodes[0]['x'])
    
    train_adjs, test_adjs = [], []
    train_y, test_y = [], []
    
    def node_fea_to_dict(node_fea):
        res = {}
        for i in range(node_fea.shape[0]):
            res[i] = node_fea[i]
        return res
    
    if hasattr(fold_train, "get_subset"):
        for d in fold_train.get_subset():

            train_y.append(d.y.item())
            train_adjs.append([d.to_numpy_array()])

        for d in fold_test.get_subset():
            test_y.append(d.y.item())
            test_adjs.append([d.to_numpy_array()])
            
    else:
        train_adjs = dataset.get_dense_adjs(fold_train)
        test_adjs = dataset.get_dense_adjs(fold_test)
        
        for d in fold_train:
            train_y.append(d.y)
        # is_labeled = data.y == data.y
        for d in fold_test:
            test_y.append(d.y)
            
        train_y = torch.cat(train_y, dim=0)
        test_y = torch.cat(test_y, dim=0)
        
    return train_adjs, test_adjs, train_y, test_y
    # do not use val for kernel methods.
#     for d in fold.dataset.get_subset():


def get_each_folder(data_name, fold_id, batch_size=1):
    
    fold_test = datasets_obj[data_name].get_test_fold(fold_id, batch_size=batch_size, shuffle=True).dataset
    
    fold_train, fold_val = datasets_obj[data_name].get_model_selection_fold(fold_id, inner_idx=None,
                                                                          batch_size=batch_size, shuffle=True)
    fold_train = fold_train.dataset
    fold_val = fold_val.dataset
    
    train_adjs = get_dense_adjs(fold_train, datasets_obj[data_name].name)
    test_adjs = get_dense_adjs(fold_test, datasets_obj[data_name].name)
    
    train_y = [d.y for d in fold_train]
    
    test_y = [d.y for d in fold_test]
    
    return train_adjs, test_adjs, train_y, test_y


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath
import numpy as np
from grakel.kernels import WeisfeilerLehman,SubgraphMatching
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from grakel import Graph
from grakel import utils as g_utils

import networkx as nx
# Loads the MUTAG dataset



# Define the Weisfeiler-Lehman kernel

def train_with_wl_kernel(wl_kernel, train_adj_matrices, test_adj_matrices, train_labels, test_labels):
    y_train = train_labels
    y_test = test_labels
    
    
    def transform_to_gr_graphs(adjs):
        nx_gs = []
        all_node_labels = []
        for m in adjs:
            nx_g = nx.from_numpy_array(m)
            N = m.shape[0]
            node_labels = {i:0 for i in range(N)}
            nx_gs.append(nx_g)
            all_node_labels.append(node_labels)
        
        gr_graphs =  [g for g in g_utils.graph_from_networkx(nx_gs, as_Graph=True)]
        
        for i, g in enumerate(gr_graphs):
            g.node_labels = all_node_labels[i]
            
        return gr_graphs
    
    
    train_graphs = transform_to_gr_graphs(train_adj_matrices)
    test_graphs = transform_to_gr_graphs(test_adj_matrices)
    
    wl_kernel.fit(train_graphs)

    # Transform the graphs using the Weisfeiler-Lehman kernel
    X_train = wl_kernel.transform([graph for graph in train_graphs])
    X_test = wl_kernel.transform([graph for graph in test_graphs])

    # Train an SVM classifier on the transformed training data
    svm = SVC()
    svm.fit(X_train, y_train)

    # Predict labels on the validation and test data using the trained SVM classifier
    y_test_pred = svm.predict(X_test)

    # Calculate the accuracy of the SVM classifier on the validation and test data
    test_accuracy = accuracy_score(y_test, y_test_pred)
    return test_accuracy



# MUTAG = fetch_dataset("MUTAG", verbose=False)
# G, y = MUTAG.data, MUTAG.target
# print('G10:', G[0])
def is_pyg_dataset(d_name:str):
    return d_name.startswith('ogb') or d_name.startswith('syn')

def get_dense_adjs(dataset, dataset_name):
    adjs = []

    if is_pyg_dataset(dataset_name):
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


def train_with_kernel(gk, dataset_name):
    res=[]
    for i in range(10):
        if isinstance(dataset_name, str):
            G_train, G_test, y_train, y_test = get_each_folder(dataset_name, i)
        else:
            G_train, G_test, y_train, y_test = get_each_folder_syn(dataset_name, i)
            
        # G_train = [g for g in graph_from_networkx(G_train,node_labels_tag='x')]
        # G_test = [g for g in graph_from_networkx(G_test,node_labels_tag='x')]
        # print('G_train 10:',G_train[:10])
        
        # G_train, G_test, y_train, y_test = train_test_split(G_train, y_train, test_size=0.1)
        # Uses the shortest path kernel to generate the kernel matrices
        if isinstance(gk, WeisfeilerLehman) or isinstance(gk, SubgraphMatching):
            res.append(train_with_wl_kernel(gk,  G_train, G_test, y_train, y_test))
        else:
            K_train = gk.fit_transform(G_train)
            K_test = gk.transform(G_test)

            # Uses the SVM classifier to perform classification
            clf = SVC(kernel="precomputed")
            clf.fit(K_train, y_train)
            y_pred = clf.predict(K_test)

            # Computes and prints the classification accuracy
            acc = accuracy_score(y_test, y_pred)
            res.append(acc)
            # print("Accuracy:", str(round(acc*100, 2)) + "%")
        
    res = np.array(res)
    print(f'Acc, mean: {round(np.mean(res)*100, 4)}, std: {round(100*np.std(res),4)}')
    
    
    
# run:

from grakel.kernels import ShortestPath, WeisfeilerLehman, SubgraphMatching

# load syn cc datasets:


def get_new_config():
    return {'model': 'GIN', 'device': 'cuda:0', 'batch_size': 128, 'learning_rate': 0.001, 'classifier_epochs':
    200, 'hidden_units': [64, 300, 300, 64], 'layer_num': 5, 'optimizer': 'Adam', 
    'scheduler': {'class': 'StepLR', 'args': {'step_size': 50, 'gamma': 0.5}}, 
    'loss': 'MulticlassClassificationLoss', 'train_eps': False, 'l2': 0.0, 'aggregation': 'mean', 'gradient_clipping': None, 
    'dropout': 0.5, 'early_stopper': {'class': 'Patience', 'args': {'patience': 30, 'use_loss': False}},
    'shuffle': True, 'resume': False,
    'additional_features': 'degree', 'node_attribute': False,
    'shuffle_feature': False, 'roc_auc': True, 'use_10_fold': True, 
    'mol_split': False, 'dataset': 'syn_degree', 
    'config_file': 'gnn_comparison/config_GIN_degree.yml', 
    'experiment': 'endtoend', 
    'result_folder': 'results/result_0530_GIN_degree_syn_degree_0.1_class2', 
    'dataset_name': 'syn_degree', 'dataset_para': '0.1_class2', 'outer_folds': 10, 
    'outer_processes': 2, 'inner_folds': 5, 'inner_processes': 1, 'debug': True, 'ogb_evl': False, 
    'model_name': 'GIN', 'device': 'cuda:0', 'batch_size': 128,
    'learning_rate': 0.001, 'classifier_epochs': 200,
    'hidden_units': [64, 300, 300, 64], 'layer_num': 5,
    'train_eps': False, 'l2': 0.0,
    'aggregation': 'mean',
    'gradient_clipping': None, 'dropout': 0.5, 
    'shuffle': True, 'resume': False, 'additional_features': 'degree',
    'node_attribute': False, 
    'shuffle_feature': False, 'roc_auc': True, 'use_10_fold': True, 
    'mol_split': False, 'dataset_name': 'syn_degree', 
    'experiment': 'endtoend', 'result_folder': 'results/result_0530_GIN_degree_syn_degree_0.1_class2',
    'dataset_para': '0.1_class2', 'outer_folds': 10, 
    'outer_processes': 2, 'inner_folds': 5, 'inner_processes': 1, 'debug': True, 'ogb_evl': False}



import sys


if __name__ == '__main__':
    cc_datasets = []
    
    for i in range(9, 10):
        configs = get_new_config()
        corrs = round(i/10.0, 1)
        print('corrs: ', corrs)
        configs['dataset_para'] = f'{corrs}_class10'
        cc_datasets.append(DATASETS['syn_cc'](config=configs))
        
                
    for d in cc_datasets:
        train_with_kernel(WeisfeilerLehman(n_iter=25), d)
        