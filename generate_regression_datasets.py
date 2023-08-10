# %%
# load datasets:
import pickle as pk
import json
from scipy.stats import pearsonr
import dataset_utils.node_feature_utils as nfu
from PrepareDatasets import DATASETS
import numpy as np

import matplotlib.pyplot as plt
import torch_geometric.utils as torch_utils

from dataset_utils.node_feature_utils import *
import sys
import os
sys.path.append(os.getcwd())


print(DATASETS.keys())
"""
    'REDDIT-BINARY': RedditBinary,
    'REDDIT-MULTI-5K': Reddit5K,
    'COLLAB': Collab,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-MULTI': IMDBMulti,
    'ENZYMES': Enzymes,
    'PROTEINS': Proteins,
    'NCI1': NCI1,
    'DD': DD,
    "MUTAG": Mutag,
    'CSL': CSL
"""

data_names = ['PROTEINS']
data_names = ['DD']
data_names = ['ENZYMES']
data_names = ['NCI1']
data_names = ['IMDB-MULTI']
data_names = ['REDDIT-BINARY']
data_names = ['CIFAR10']
data_names = ['ogbg_molhiv']

# NOTE:new kernel:
data_names = ['DD', 'PROTEINS', 'ENZYMES']

data_names = ['ogbg_moltox21', 'ogbg-molbace']

data_names = ['MUTAG']
data_names = []
datasets_obj = {}
for k, v in DATASETS.items():
    if k not in data_names:
        continue
    print('loaded dataset, name:', k)
    dat = v(use_node_attrs=True)
    datasets_obj[k] = dat


"""
each folder:

{"best_config": {"config": {"model": "GIN", "device": "cuda:1", "batch_size": 64,
"learning_rate": 0.0001, "classifier_epochs": 200, "hidden_units": [64, 64, 64, 64],
"layer_num": 5, "optimizer": "Adam", "scheduler": {"class": "StepLR", "args": {"step_size": 50, "gamma": 0.5}}, 
"loss": "MulticlassClassificationLoss", "train_eps": false, "l2": 0.0, "aggregation": "sum",
"gradient_clipping": null, "dropout": 0.5, "early_stopper": {"class": "Patience", 
"args": {"patience": 50, "use_loss": false}}, "shuffle": true, "resume": false, "additional_features": "degree", 
"node_attribute": false, "shuffle_feature": false, "roc_auc": false, "mol_split": false, "dataset": "syn_cc",
"config_file": "gnn_comparison/config_GIN_degree.yml", "experiment": "endtoend", 
"result_folder": "results/result_0422_GIN_degree_syn_cc_0.1", "dataset_name": "syn_cc",
"dataset_para": "0.1", "outer_folds": 10, "outer_processes": 2, "inner_folds": 5, "inner_processes": 1, 
"debug": true, "ogb_evl": false}, "TR_score": 16.183574925298277, "VL_score": 21.505376272304083, 
"TR_roc_auc": -1, "VL_roc_auc": -1}, "OUTER_TR": 14.774557204254199, "OUTER_TS": 11.003236511378612,
"OUTER_TR_ROCAUC": -1, "OUTER_TE_ROCAUC": -1}

whole_assessment:
{"avg_TR_score": 97.172316605532, "std_TR_score": 0.158351532933239, "avg_TS_score": 96.85607610347206, 
"std_TS_score": 0.13199798398933663, "avg_TR_ROCAUC": 0.6630882047950513, "std_TR_ROCAUC": 0.025717343617869023,
"avg_TE_ROCAUC": 0.6298793940035297, "std_TE_ROCAUC": 0.031157193352788347}#

"""

_OUTER_RESULTS_FILENAME = 'outer_results.json'

def get_test_acc(data_root_path, fold=10, as_whole=False, roc_auc=False):
    if data_root_path is None:
        return None if as_whole else [None for _ in range(fold)]

    if as_whole:
        assess_path = os.path.join(data_root_path, 'assessment_results.json')
        # load file to json, and return avg_TS_score and std_TS_score from json
        with open(assess_path, 'r') as fp:
            assess_results = json.load(fp)
            return float(assess_results['avg_TS_score']) if not roc_auc else float(assess_results['avg_TE_ROCAUC'])

    outer_TR_scores, outer_TS_scores, outer_TR_ROCAUC, outer_TE_ROCAUC = [], [], [], []
    for i in range(1, fold+1):
        config_filename = os.path.join(
            data_root_path, f'OUTER_FOLD_{i}', _OUTER_RESULTS_FILENAME)

        with open(config_filename, 'r') as fp:
            outer_fold_scores = json.load(fp)

            outer_TR_scores.append(outer_fold_scores['OUTER_TR'])
            outer_TS_scores.append(outer_fold_scores['OUTER_TS'])

            if 'OUTER_TE_ROCAUC' in outer_fold_scores:
                outer_TR_ROCAUC.append(outer_fold_scores['OUTER_TR_ROCAUC'])
                outer_TE_ROCAUC.append(outer_fold_scores['OUTER_TE_ROCAUC'])

    return outer_TE_ROCAUC if roc_auc else outer_TS_scores


def extract_features(adjs, labels):

    def get_mean_std_corr(features, labels):

        features = np.array(features).squeeze()
        mean = np.array(np.mean(features))
        std = np.array(np.std(features))
        # print(np.isnan(mean).any(), np.isnan(std).any())
        x = np.array(features).reshape(-1)
        # NOTE: if multilabel, use the average of all labels:
        y = np.array(labels).squeeze()
        if y.ndim > 1:
            corrs = []
            print(y.shape)
            for i in range(y.shape[1]):
                # ignore nan in y[:, i]
                not_nan = ~np.isnan(y[:, i])
                x_i = x[not_nan]
                y_i = y[not_nan]
                corr, _ = pearsonr(x_i, y_i[:, i].squeeze())
                if np.isnan(corr):
                    corr = np.array([0])
                corrs.append(corr)
            corr = np.mean(corrs)
        else:
            not_nan = ~np.isnan(y)
            x_nn = x[not_nan]
            y_nn = y[not_nan]

            corr, _ = pearsonr(x_nn, y_nn)
            if np.isnan(corr):
                corr = np.array([0])

            if not isinstance(corr, np.ndarray):
                corr = np.array([corr])

        return np.array([mean.item(), std.item(), corr.item()])

    # F1: avgD:
    avg_d = [nfu.graph_avg_degree(adj=adj) for adj in adjs]
    f_avgD = get_mean_std_corr(avg_d, labels)
    # F2: avgCC:
    avg_cc = [nfu.node_cc_avg_feature(adj=adj) for adj in adjs]
    f_avgCC = get_mean_std_corr(avg_cc, labels)

    # F3: avgD/N:
    avg_DN = [nfu.graph_avgDN_feature(adj=adj) for adj in adjs]
    f_avgDN = get_mean_std_corr(avg_DN, labels)

    # F4: node num N:
    avg_N = [adj.shape[0] for adj in adjs]
    f_avgN = get_mean_std_corr(avg_N, labels)

    # F5: labels
    # calculate each dimension of labels:
    Y = np.array(labels).squeeze()

    if Y.ndim > 1:
        f_Ys = []
        for i in range(Y.shape[1]):
            y_i = Y[:, i]
            print('y_i: ', y_i.shape)
            y_i = y_i[~np.isnan(y_i)]
            print('y_i not nan: ', y_i.shape)
            f_Ys.append(get_mean_std_corr(y_i, y_i)[:2])
        f_Y = np.concatenate(f_Ys)
        f_Y = np.mean(f_Ys, axis=0)
    else:
        f_Y = get_mean_std_corr(Y, Y)[:2]

    # F6: cycles:
    avg_cyc = [nfu.graph_cycle_feature(adj=adj, k='4-5-6-7') for adj in adjs]
    f_cyc4 = get_mean_std_corr([c[0] for c in avg_cyc], labels)
    f_cyc5 = get_mean_std_corr([c[1] for c in avg_cyc], labels)
    f_cyc6 = get_mean_std_corr([c[2] for c in avg_cyc], labels)
    f_cyc7 = get_mean_std_corr([c[3] for c in avg_cyc], labels)

    feas = np.concatenate(
        [f_avgD, f_avgCC, f_avgDN, f_avgN, f_Y, f_cyc4, f_cyc5, f_cyc6, f_cyc7], axis=0)
    return feas


# construct E of each fold, and plot

# Effectiveness


# NOTE: get E for each dataset:

def plot_E(es, ax=None, title='E=(E_struct+E_attr)/2'):

    # e_res = sorted(es, key=lambda x:x[0])
    e_res = es
    labels = [e[1] for e in e_res]

    if ax is None:
        fig, ax = plt.subplots(dpi=100)

    for e in e_res:
        bars = ax.bar(e[1], e[0], label=e[1], hatch='\\',
                      edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='center')
    ax.set_axisbelow(True)
    ax.grid(linestyle='dashed', zorder=0)
    ax.set_title(title)



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


def get_new_E(Acc_MLP_attr, Acc_GNN_attr, Acc_MLP_avg_degree, Acc_GNN_degree,
              Y_num, is_abs=True, is_roc=False, factor=0.5):
    # class_factor = np.log2(Y_num)
    
    if not is_roc:
        Acc_MLP_avg_degree /= 100.0
        Acc_GNN_degree /= 100.0
    
    class_factor = 1.0
    min_ab_struct = min(Acc_GNN_degree, Acc_MLP_avg_degree)
    if is_abs:
        E_struct = abs(Acc_GNN_degree - Acc_MLP_avg_degree) / min_ab_struct / (Y_num -1) \
            * (1 - min_ab_struct)/(1 - 1.0/Y_num) * class_factor
    else:
        E_struct = (Acc_GNN_degree - Acc_MLP_avg_degree) / min_ab_struct  / (Y_num -1) \
            * (1 - min_ab_struct)/(1 - 1.0/Y_num) * class_factor
    
    if Acc_MLP_attr is None:
        E_attribute = 0
        factor = 1
    else:
        if not is_roc:
            Acc_MLP_attr /= 100.0
            Acc_GNN_attr /= 100.0
            
        min_ab_attr = min(Acc_MLP_attr, Acc_GNN_attr)
        if is_abs:
            E_attribute = abs(Acc_GNN_attr - Acc_MLP_attr) / min_ab_attr / (Y_num -1) \
            * (1 - min_ab_attr) / (1 - 1.0/Y_num) * class_factor
        else:
            E_attribute = (Acc_GNN_attr - Acc_MLP_attr) / min_ab_attr / (Y_num -1) \
            * (1 - min_ab_attr) / (1 - 1.0/Y_num) * class_factor
            
    return (E_struct+E_attribute) * factor, E_struct * factor

# %%

def E_datasets(dataset,
               MLP_log_path_attr=None, GNN_log_path_attr=None,
               MLP_log_path_struct=None, GNN_log_path_struct=None, fold=10, as_whole=False, roc_auc=False):

    MLP_test_acc_attr = get_test_acc(
        MLP_log_path_attr, fold=fold, as_whole=as_whole, roc_auc=roc_auc)
    GNN_test_acc_attr = get_test_acc(
        GNN_log_path_attr, fold=fold, as_whole=as_whole, roc_auc=roc_auc)

    MLP_test_acc_struct = get_test_acc(
        MLP_log_path_struct, fold=fold, as_whole=as_whole, roc_auc=roc_auc)
    GNN_test_acc_struct = get_test_acc(
        GNN_log_path_struct, fold=fold, as_whole=as_whole, roc_auc=roc_auc)

    print('_dim_targets: ', dataset._dim_target)
    mutag_splits = []

    if as_whole:
        print('__as whole__')
        adjs = get_dense_adjs(dataset, dataset.name)
        labels = dataset.get_labels()
        feas = extract_features(adjs=adjs, labels=labels)
        e = get_new_E(MLP_test_acc_attr, GNN_test_acc_attr,
                      MLP_test_acc_struct, GNN_test_acc_struct, dataset._dim_target)
        mutag_splits.append((feas, e))
        return mutag_splits
        # labels = [d.y for d in train_loader.dataset] + [d.y for d in val_loader.dataset]
    else:
        for i in range(fold):
            train_loader, val_loader = dataset.get_model_selection_fold(
                outer_idx=i, inner_idx=0, batch_size=1, shuffle=False)
            adjs = get_dense_adjs(train_loader.dataset, dataset.name) + \
                get_dense_adjs(val_loader.dataset, dataset.name)

            labels = [d.y for d in train_loader.dataset] + \
                [d.y for d in val_loader.dataset]
            feas = extract_features(adjs=adjs, labels=labels)
            e = get_new_E(MLP_test_acc_attr[i], GNN_test_acc_attr[i],
                          MLP_test_acc_struct[i], GNN_test_acc_struct[i], dataset._dim_target)

            mutag_splits.append((feas, e))

    return mutag_splits


# save datasets


def save_datasets(datasets, file_name):
    with open(file_name, 'wb') as f:
        pk.dump(datasets, f)


def load_datasets(file_name):
    with open(file_name, 'rb') as f:
        datasets = pk.load(f)
    return datasets

# pref = 'whole_'

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



def generate_save_regression_dataset(dataset_name: str,
                                     MLP_log_path_attr=None, GNN_log_path_attr=None,
                                     MLP_log_path_degree=None, GNN_log_path_degree=None,
                                     as_whole=False, return_E=False, dim_y=None, fold=10,
                                     roc_auc=False, factor=1.0, class_num=10, 
                                     return_acc=False, 
                                     name_pre="new_E_", dataset=None):
    data_names = [dataset_name]

    if return_E:
        MLP_test_acc_attr = get_test_acc(
            MLP_log_path_attr, fold=fold, as_whole=as_whole, roc_auc=roc_auc)
        GNN_test_acc_attr = get_test_acc(
            GNN_log_path_attr, fold=fold, as_whole=as_whole, roc_auc=roc_auc)
        MLP_test_acc_struct = get_test_acc(
            MLP_log_path_degree, fold=fold, as_whole=as_whole, roc_auc=roc_auc)
        GNN_test_acc_struct = get_test_acc(
            GNN_log_path_degree, fold=fold, as_whole=as_whole, roc_auc=roc_auc)

        print(f'{dataset_name}: MLP_test_acc_attr', MLP_test_acc_attr)
        print(f'{dataset_name}: GNN_test_acc_attr', GNN_test_acc_attr)
        print()
        print(f'{dataset_name}: MLP_test_acc_struct', MLP_test_acc_struct)
        print(f'{dataset_name}: GNN_test_acc_struct', GNN_test_acc_struct)
        if return_acc:
            
            if not as_whole:
                return get_new_E(MLP_test_acc_attr, GNN_test_acc_attr, MLP_test_acc_struct, \
                GNN_test_acc_struct, dim_y, is_roc=roc_auc, factor=factor), \
                    (MLP_test_acc_attr, GNN_test_acc_attr, MLP_test_acc_struct, GNN_test_acc_struct)
                    
            Es = []
            for i in range(len(MLP_test_acc_struct)):
                Es.append(get_new_E(MLP_test_acc_attr[i], GNN_test_acc_attr[i], MLP_test_acc_struct[i], \
                         GNN_test_acc_struct[i], dim_y, is_roc=roc_auc, factor=factor), \
                             (MLP_test_acc_attr[i], GNN_test_acc_attr[i], MLP_test_acc_struct[i], GNN_test_acc_struct[i]))
            return Es 
        
        if not as_whole:
            return get_new_E(MLP_test_acc_attr, GNN_test_acc_attr, MLP_test_acc_struct, \
                         GNN_test_acc_struct, dim_y, is_roc=roc_auc, factor=factor)
        else:
            Es = []
            for i in range(len(MLP_test_acc_struct)):
                Es.append(get_new_E(MLP_test_acc_attr, GNN_test_acc_attr, MLP_test_acc_struct, \
                         GNN_test_acc_struct, dim_y, is_roc=roc_auc, factor=factor))
            return Es
        
        
    if dataset is None:
        datasets_obj = {}
        
        for k, v in DATASETS.items():
            if k not in data_names:
                continue
            print('loaded dataset, name:', k)
            dat = v(use_node_attrs=True,)
            datasets_obj[k] = dat

        dataset = datasets_obj[dataset_name]


    cur_datasets = E_datasets(dataset, MLP_log_path_attr, GNN_log_path_attr,
                              MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole, roc_auc=False)

    if as_whole:
        pref = 'whole_'
    else:
        pref = 'new_10fold'

    save_datasets(cur_datasets, f'{name_pre}{pref}{dataset_name.lower()}_datasets.pkl')
    print('save datasets:', f'{name_pre}{pref}{dataset_name.lower()}_datasets.pkl')
    
    
data_log_path_dict = {
    # replace to the latest with roc_auc:
    'MUTAG': (
        f'./results/result_0521_Baseline_fingerprint_attr_MUTAG/MolecularFingerprint_MUTAG_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_attr_MUTAG/GIN_MUTAG_assessment/10_NESTED_CV',
        f'./results/result_0522_Baseline_mlp_MUTAG/MolecularGraphMLP_MUTAG_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_degree_MUTAG/GIN_MUTAG_assessment/10_NESTED_CV',
        './results/result_0524_GCN_attr_MUTAG/GCN_MUTAG_assessment/10_NESTED_CV', # GCN with attr,
        './results/result_0524_GCN_degree_MUTAG/GCN_MUTAG_assessment/10_NESTED_CV' # GCN with degree
    ),
    'DD': (f'./results/result_0521_Baseline_fingerprint_attr_DD/MolecularFingerprint_DD_assessment/10_NESTED_CV',
           f'./results/result_GIN_0521_GIN_attr_DD/GIN_DD_assessment/10_NESTED_CV',
           f'./results/result_0522_Baseline_mlp_DD/MolecularGraphMLP_DD_assessment/10_NESTED_CV',
           f'./results/result_GIN_0521_GIN_degree_DD/GIN_DD_assessment/10_NESTED_CV',
           './results/result_0524_GCN_attr_DD/GCN_DD_assessment/10_NESTED_CV', # GCN with attr,
           './results/result_0524_GCN_degree_DD/GCN_DD_assessment/10_NESTED_CV' # GCN with degree
        ),
    'PROTEINS': (
        f'./results/result_0521_Baseline_fingerprint_attr_PROTEINS/MolecularFingerprint_PROTEINS_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_attr_PROTEINS/GIN_PROTEINS_assessment/10_NESTED_CV',
        f'./results/result_0522_Baseline_mlp_PROTEINS/MolecularGraphMLP_PROTEINS_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_degree_PROTEINS/GIN_PROTEINS_assessment/10_NESTED_CV',
        './results/result_0524_GCN_attr_PROTEINS/GCN_PROTEINS_assessment/10_NESTED_CV', # GCN with attr,
        './results/result_0524_GCN_degree_PROTEINS/GCN_PROTEINS_assessment/10_NESTED_CV' # GCN with degree
        ),
    'ENZYMES': ( # class num = 6, no roc_auc
        f'./results/result_0516_Baseline_fingerprint_attr_ENZYMES/MolecularFingerprint_ENZYMES_assessment/10_NESTED_CV',
        f'./results/result_GIN_0404_GIN_attr_ENZYMES/GIN_ENZYMES_assessment/10_NESTED_CV',
        f'./results/result_0516_Baseline_mlp_ENZYMES/MolecularGraphMLP_ENZYMES_assessment/10_NESTED_CV',
        f'./results/result_GIN_0403_GIN_degree_ENZYMES/GIN_ENZYMES_assessment/10_NESTED_CV',
        './results/result_0524_GCN_attr_ENZYMES/GCN_ENZYMES_assessment/10_NESTED_CV', # GCN with attr,
        './results/result_0524_GCN_degree_ENZYMES/GCN_ENZYMES_assessment/10_NESTED_CV' # GCN with degree
        ),
    'AIDS': (
        f'./results/result_0521_Baseline_fingerprint_attr_AIDS/MolecularFingerprint_AIDS_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_attr_AIDS/GIN_AIDS_assessment/10_NESTED_CV',
        f'./results/result_0522_Baseline_mlp_AIDS/MolecularGraphMLP_AIDS_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_degree_AIDS/GIN_AIDS_assessment/10_NESTED_CV',
        './results/result_0524_GCN_attr_AIDS/GCN_AIDS_assessment/10_NESTED_CV', # GCN with attr,
        './results/result_0524_GCN_degree_AIDS/GCN_AIDS_assessment/10_NESTED_CV' # GCN with degree
    ),
    'NCI1': (
        f'./results/result_0524_Baseline_fingerprint_attr_NCI1/MolecularFingerprint_NCI1_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_attr_NCI1/GIN_NCI1_assessment/10_NESTED_CV',
        f'./results/result_0524_Baseline_mlp_degree_binary_NCI1/MolecularGraphMLP_NCI1_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_degree_NCI1/GIN_NCI1_assessment/10_NESTED_CV',
        './results/result_0524_GCN_attr_NCI1/GCN_NCI1_assessment/10_NESTED_CV', # GCN with attr,
        './results/result_0524_GCN_degree_NCI1/GCN_NCI1_assessment/10_NESTED_CV' # GCN with degree
    ),
    'ogbg_molhiv': (
        f'./results/result_0521_Baseline_fingerprint_attr_ogbg_molhiv/MolecularFingerprint_ogbg_molhiv_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_attr_ogbg_molhiv/GIN_ogbg_molhiv_assessment/10_NESTED_CV',
        f'./results/result_0521_Baseline_mlp_ogbg_molhiv/MolecularGraphMLP_ogbg_molhiv_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_degree_ogbg_molhiv/GIN_ogbg_molhiv_assessment/10_NESTED_CV',
        f'./results/result_0525_GCN_attr_ogbg_molhiv/GCN_ogbg_molhiv_assessment/10_NESTED_CV', # GCN with attr,
        f'./results/result_0524_GCN_degree_ogbg_molhiv/GCN_ogbg_molhiv_assessment/10_NESTED_CV' # GCN with degree
    ),
    'ogbg_moltox21': (
        f'./results/result_GIN_0411_atomencoder_attr_ogbg_moltox21/AtomMLP_ogbg_moltox21_assessment/1_NESTED_CV',
        f'./results/result_GIN_0409_EGNN_attr_ogbg_moltox21/EGNN_ogbg_moltox21_assessment/1_NESTED_CV',
        f'./results/result_0424_Baseline_mlp_mol_ogbg_moltox21/MolecularGraphMLP_ogbg_moltox21_assessment/1_NESTED_CV',
        f'./results/result_GIN_0410_GIN_degree_ogbg_moltox21/GIN_ogbg_moltox21_assessment/1_NESTED_CV',
        './results/result_0525_GCN_attr_tox21_ogbg_moltox21/GCN_ogbg_moltox21_assessment/1_NESTED_CV', # GCN with attr,
        './results/result_0525_GCN_degree_tox21_ogbg_moltox21/GCN_ogbg_moltox21_assessment/1_NESTED_CV' # GCN with degree
    ),
    'ogbg-molbace': (
        f'./results/result_0521_Baseline_fingerprint_attr_ogbg-molbace/MolecularFingerprint_ogbg-molbace_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_attr_ogbg-molbace/GIN_ogbg-molbace_assessment/10_NESTED_CV',
        f'./results/result_0521_Baseline_mlp_ogbg-molbace/MolecularGraphMLP_ogbg-molbace_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_degree_ogbg-molbace/GIN_ogbg-molbace_assessment/10_NESTED_CV',
        './results/result_0524_GCN_degree_ogbg-molbace/GCN_ogbg-molbace_assessment/10_NESTED_CV',
        './results/result_0525_GCN_attr_ogbg-molbace/GCN_ogbg-molbace_assessment/10_NESTED_CV'
    ),
    'ogbg_ppa': ( # class num = 37, no roc_auc
        f'./results/result_0508_Baseline_mlp_edge_attr_ogbg_ppa/MolecularFingerprint_ogbg_ppa_assessment/1_NESTED_CV',
        f'./results/result_0508_GIN_degree_ogbg_ppa_/GIN_ogbg_ppa_assessment/1_NESTED_CV',
        f'./results/result_0507_Baseline_mlp_ogbg_ppa/MolecularGraphMLP_ogbg_ppa_assessment/1_NESTED_CV',
        f'./results/result_0508_GIN_attr_edge_ogbg_ppa/OGBGNN_ogbg_ppa_assessment/1_NESTED_CV',
        f'./results/result_0525_GCN_degree_ogbg_ppa/GCN_ogbg_ppa_assessment/1_NESTED_CV', # GCN with attr,
        f'./results/result_0525_GCN_attr_edge_ogbg_ppa/OGBGNN_ogbg_ppa_assessment/1_NESTED_CV' # GCN with degree
    ),
    'CIFAR10': (# class num = 10, no roc_auc
        f'./results/result_0510_Baseline_fingerprint_attr_CIFAR10/MolecularFingerprint_CIFAR10_assessment/10_NESTED_CV',
        f'./results/result_GIN_0510_GIN_attr_CIFAR10/GIN_CIFAR10_assessment/10_NESTED_CV',
        f'./results/result_0510_Baseline_mlp_CIFAR10/MolecularGraphMLP_CIFAR10_assessment/10_NESTED_CV',
        f'./results/result_GIN_0510_GIN_degree_CIFAR10/GIN_CIFAR10_assessment/10_NESTED_CV',
        None, # GCN with attr,
        f'./results/result_0524_GCN_degree_CIFAR10/GCN_CIFAR10_assessment/10_NESTED_CV' # GCN with degree
    ),
    'MNIST': (# class num = 10, no roc_auc
        f'./results/result_0510_Baseline_fingerprint_attr_MNIST/MolecularFingerprint_MNIST_assessment/10_NESTED_CV',
        f'./results/result_GIN_0510_GIN_attr_MNIST/GIN_MNIST_assessment/10_NESTED_CV',
        f'./results/result_0510_Baseline_mlp_MNIST/MolecularGraphMLP_MNIST_assessment/10_NESTED_CV',
        f'./results/result_GIN_0510_GIN_degree_MNIST/GIN_MNIST_assessment/10_NESTED_CV',
        None, # GCN with attr,
        f'./results/result_0524_GCN_degree_MNIST/GCN_MNIST_assessment/10_NESTED_CV'
    ),

    'IMDB-BINARY': (
        None,
        None,
        f'./results/result_0522_Baseline_mlp_degree_IMDB-BINARY/MolecularGraphMLP_IMDB-BINARY_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_degree_IMDB-BINARY/GIN_IMDB-BINARY_assessment/10_NESTED_CV',
        None, # GCN with attr,
        './results/result_0525_GCN_degree_IMDB-BINARY/GCN_IMDB-BINARY_assessment/10_NESTED_CV' # GCN with degree
    ),
    'IMDB-MULTI': (
        None,
        None, 
        f'./results/result_0424_Baseline_mlp_IMDB-MULTI/MolecularGraphMLP_IMDB-MULTI_assessment/10_NESTED_CV',
        f'./results/result_GIN_0313_only_degree_IMDB-MULTI/GIN_IMDB-MULTI_assessment/10_NESTED_CV',
        None, # GCN with attr,
        f'./results/result_0525_GCN_degree_IMDB-MULTI/GCN_IMDB-MULTI_assessment/10_NESTED_CV' 
    ),
    'COLLAB': ( # class num = 3
        None,
        None,
        f'./results/result_0423_Baseline_mlp_COLLAB/MolecularGraphMLP_COLLAB_assessment/10_NESTED_CV',
        f'./results/result_GIN_0313_only_degree_COLLAB/GIN_COLLAB_assessment/10_NESTED_CV',
        None, # GCN with attr,
        f'./results/result_0525_GCN_degree_COLLAB/GCN_COLLAB_assessment/10_NESTED_CV' 
    ),
    'REDDIT-BINARY': (
        None, 
        None,
        f'./results/result_0522_Baseline_mlp_degree_REDDIT-BINARY/MolecularGraphMLP_REDDIT-BINARY_assessment/10_NESTED_CV',
        f'./results/result_GIN_0521_GIN_degree_REDDIT-BINARY/GIN_REDDIT-BINARY_assessment/10_NESTED_CV',
        None, # GCN with attr,
        f'./results/result_0525_GCN_degree_REDDIT-BINARY/GCN_REDDIT-BINARY_assessment/10_NESTED_CV'
    ),
    'syn_degree': (
        None,
        None,
        # result_0529_Baseline_mlp_degree_syn_degree_0.x_classy/, x,y are parameters
        f'./results/result_0530_Baseline_mlp_degree_syn_degree/MolecularGraphMLP_syn_degree_assessment/10_NESTED_CV',
        f'./results/result_0530_GIN_degree_syn_degree/GIN_syn_degree_assessment/10_NESTED_CV',
        None, # GCN with attr,
        None # running
    ),
    'syn_cc': (
        None,
        None,
        # result_0529_Baseline_mlp_degree_syn_degree_0.x_classy/, x,y are parameters
        # f'./results/result_0604_Baseline_mlp_cc_syn_cc/MolecularGraphMLP_syn_cc_assessment/10_NESTED_CV',
        f'./results/result_0604_Baseline_mlp_degree_syn_cc/MolecularGraphMLP_syn_cc_assessment/10_NESTED_CV',
        f'./results/result_0604_GIN_degree_syn_cc/GIN_syn_cc_assessment/10_NESTED_CV',
        None, # GCN with attr,
        f'./results/result_0604_GCN_degree_syn_cc/GCN_syn_cc_assessment/10_NESTED_CV' # running
    ),
}

def generate_syn_cc(as_whole=False, return_E=False, dim_y=None, fold=10, 
                    roc_auc=False, class_num=5, use_gcn=False, return_acc=False):
    
    
    def reconstruct_path(path, corr, class_num, pre_name=""):
        splits = path.split('/')
        splits[2] = splits[2]+'_'+str(corr)+'_class'+str(class_num)+pre_name
        return os.path.join(*splits)
    
    generate_res = []
    
    for i in range(1, 10):
        _, _, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
            'syn_cc')
        # reconstruct the path by corr and class_num
        if use_gcn:
            GNN_log_path_degree = GCN_log_path_degree

        corr = round(i/10.0, 1)
        MLP_log_path_degree = reconstruct_path(MLP_log_path_degree, corr, class_num, "_final")
        GNN_log_path_degree = reconstruct_path(GNN_log_path_degree, corr, class_num, "_final")
        
        
        if not return_E:
            configs = get_new_config()
            corrs = round(i/10.0, 1)
            configs['dataset_para'] = f'{corrs}_class{class_num}_final'
            dataset = DATASETS['syn_cc'](config=configs)
        else:
            dataset = None

        generate_res.append(generate_save_regression_dataset('syn_cc', None, None,
                                                MLP_log_path_degree, GNN_log_path_degree,  name_pre=f'{corrs}_class{class_num}_final',
                                                as_whole=as_whole,
                                                return_E=return_E, dim_y=dim_y, fold=fold,
                                                roc_auc=roc_auc, class_num=class_num, dataset=dataset))
        
    print('done')
    
    return generate_res
        
        
def generate_syn_degree(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False, class_num=2,
                        use_gcn=False, return_acc=False):
    
    def reconstruct_path(path, corr, class_num):
        splits = path.split('/')
        splits[2] = splits[2]+'_'+str(corr)+'_class'+str(class_num)
        return os.path.join(*splits)
    
    generate_res = []
    
    for i in range(1, 10):
        _, _, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
            'syn_degree')
        # reconstruct the path by corr and class_num
        if use_gcn:
            GNN_log_path_degree = GCN_log_path_degree
            
        corr = round(i/10.0, 1)
        MLP_log_path_degree = reconstruct_path(MLP_log_path_degree, corr, class_num)
        GNN_log_path_degree = reconstruct_path(GNN_log_path_degree, corr, class_num)
        
        generate_res.append(generate_save_regression_dataset('syn_degree', None, None,
                                                MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                                return_E=return_E, dim_y=dim_y, fold=fold, 
                                                roc_auc=roc_auc, class_num=class_num, return_acc=return_acc))
        
    return generate_res
        

def get_path_by_name(name):
    return data_log_path_dict[name]

def generate_mutag(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'MUTAG')
    
    return generate_save_regression_dataset('MUTAG', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_DD(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'DD')
    return generate_save_regression_dataset('DD', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_PROTEINS(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'PROTEINS')
    return generate_save_regression_dataset('PROTEINS', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_ENZYMES(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'ENZYMES')
    return generate_save_regression_dataset('ENZYMES', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_CIFAR10(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'CIFAR10')
    return generate_save_regression_dataset('CIFAR10', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_MNIST(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'MNIST')
    return generate_save_regression_dataset('MNIST', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_AIDS(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'AIDS')
    return generate_save_regression_dataset('AIDS', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_NCI1(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'NCI1')
    return generate_save_regression_dataset('NCI1', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_IMDB_B(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'IMDB-BINARY')
    return generate_save_regression_dataset('IMDB-BINARY', MLP_log_path_attr=None, GNN_log_path_attr=None,
                                            MLP_log_path_degree=MLP_log_path_degree,
                                            GNN_log_path_degree=GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_IMDB_M(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'IMDB-MULTI')
    return generate_save_regression_dataset('IMDB-MULTI', MLP_log_path_attr=None, GNN_log_path_attr=None,
                                            MLP_log_path_degree=MLP_log_path_degree,
                                            GNN_log_path_degree=GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_COLLAB(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'COLLAB')
    return generate_save_regression_dataset('COLLAB', MLP_log_path_attr=None, GNN_log_path_attr=None,
                                            MLP_log_path_degree=MLP_log_path_degree,
                                            GNN_log_path_degree=GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_REDDITB(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'REDDIT-BINARY')
    return generate_save_regression_dataset('REDDIT-BINARY', MLP_log_path_attr=None, GNN_log_path_attr=None,
                                            MLP_log_path_degree=MLP_log_path_degree,
                                            GNN_log_path_degree=GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_HIV(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'ogbg_molhiv')
    return generate_save_regression_dataset('ogbg_molhiv', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_tox21(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'ogbg_moltox21')
    as_whole = True
    return generate_save_regression_dataset('ogbg_moltox21', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_bace(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'ogbg-molbace')
    return generate_save_regression_dataset('ogbg-molbace', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


def generate_ppa(as_whole=False, return_E=False, dim_y=None, fold=10, roc_auc=False):
    MLP_log_path_attr, GNN_log_path_attr, MLP_log_path_degree, GNN_log_path_degree, GCN_log_path_attr, GCN_log_path_degree  = get_path_by_name(
        'ogbg_ppa')
    as_whole = True
    return generate_save_regression_dataset('ogbg_ppa', MLP_log_path_attr, GNN_log_path_attr,
                                            MLP_log_path_degree, GNN_log_path_degree, as_whole=as_whole,
                                            return_E=return_E, dim_y=dim_y, fold=fold, roc_auc=roc_auc)


_OUTER_RESULTS_FILENAME = 'outer_results.json'

def get_outer_final(acc_log_path):
    # assessment_results.json
    if acc_log_path is None:
        return [(np.nan, np.nan), (np.nan, np.nan)]

    assess_path = os.path.join(acc_log_path, 'assessment_results.json')
    # load file to json, and return avg_TS_score and std_TS_score from json
    with open(assess_path, 'r') as fp:
        assess_results = json.load(fp)
        # "avg_TE_ROCAUC": 0.6298793940035297, "std_TE_ROCAUC":
        if 'avg_TE_ROCAUC' in assess_results:
            te_roc = float(assess_results['avg_TE_ROCAUC'])
            te_roc_std = float(assess_results['std_TE_ROCAUC'])
            return [(float(assess_results['avg_TS_score']), float(assess_results['std_TS_score'])),
                    (100 * te_roc, 100 * te_roc_std)]
        else:
            return [(float(assess_results['avg_TS_score']), float(assess_results['std_TS_score'])),
                    (-100.0, -100.0)]


def load_log_to_log_results(log_results, MLP_log_path_attr=None, GNN_log_path_attr=None,
                            MLP_log_path_degree=None, GNN_log_path_degree=None, 
                            GCN_log_path_attr=None, GCN_log_path_degree=None,
                            is_syn=False):

    name = MLP_log_path_degree.split('/')[2].split('_')[-1]
    print('data name:', name)

    res = [get_outer_final(MLP_log_path_attr), get_outer_final(GNN_log_path_attr),
           get_outer_final(MLP_log_path_degree), get_outer_final(GNN_log_path_degree),
           get_outer_final(GCN_log_path_attr), get_outer_final(GCN_log_path_degree)]

    log_results[name] = res


# %%
as_whole = False



if __name__ == '__main__':
    
    generate_syn_cc(as_whole=as_whole, class_num=2)
    
    # generate_mutag(as_whole)
    # generate_NCI1(as_whole)
    # generate_AIDS(as_whole)
    # generate_bace(as_whole)
    
    # generate_DD(as_whole)
    # generate_ENZYMES(as_whole)
    # generate_PROTEINS(as_whole)
    # generate_HIV(as_whole)
    # generate_tox21(as_whole)
    
    # generate_ppa(as_whole)


    # generate_IMDB_M(as_whole)
    # generate_IMDB_B(as_whole)
    # generate_REDDITB(as_whole)
    # generate_COLLAB(as_whole)

    # generate_MNIST(as_whole)
    # generate_CIFAR10(as_whole)