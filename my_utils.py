import argparse
import pickle
from collections import Counter, defaultdict
import random


import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
    
from torch import  nn
import torch.nn.functional as F


class DaoArgs(object):
    def __init__(self, args=None) -> None:
        if args is None:
            args = get_common_args().parse_args({})
        self.base_args = args
        for k, v in vars(self.base_args).items():
            self.set_attr(k, v)

    def set_attr(self, attr_name, attr_value):
        setattr(self, attr_name, attr_value)
        


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=98, help='seed')
    parser.add_argument('--server_tag', type=str, default='seizure', help='server_tag')
    parser.add_argument('--out_middle_features', action='store_true', help='out_middle_features')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--class_num', type=int, default=-1, help='class_num')
    parser.add_argument('--model_name', type=str, default='gnn', help='model_name')
    
    
    # position encoding
    parser.add_argument('--pos_en', type=str, default='none', help='pos_en')
    parser.add_argument('--pe_init', type=str, default='none', help='pe_init')
    parser.add_argument('--pos_en_dim', type=int, default=8, help='position encoding dim')
    

    # dataset:
    parser.add_argument('--task', type=str, default='seizure', help='eeg task type')
    parser.add_argument('--dataset', type=str, default='SEED', help='SEED, SEED_IV')
    parser.add_argument('--data_path', type=str, default='./data/METR-LA', help='data path')
    parser.add_argument('--adj_file', type=str, default='./data/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    parser.add_argument('--adj_type', type=str, default='scalap', help='adj type', choices=ADJ_CHOICES)

    # EEG specified:
    parser.add_argument('--testing', action='store_true', help='testing')
    parser.add_argument('--arg_file', type=str, default='None', help='chose saved arg file')
    parser.add_argument('--independent', action='store_true', help='subject independent')
    parser.add_argument('--using_fc', action='store_true', help='using_fc')

    parser.add_argument('--unit_test', action='store_true')
    parser.add_argument('--multi_train', action='store_true')

    parser.add_argument('--focalloss', action='store_true', help='focalloss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal_gamma')
    parser.add_argument('--weighted_ce', type=str, help='weighted cross entropy opt')
    parser.add_argument('--dev', action='store_true', help='dev')
    parser.add_argument('--dev_size', type=int, default=1000, help='dev_sample_size')
    parser.add_argument('--best_model_save_path', type=str, default='.best_model', help='best_model')
    parser.add_argument('--pre_model_path', type=str, default='./best_models/seed_pretrain_08021405', help='pre_model_path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.92, help='lr_decay_rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--clip', type=int, default=3, help='clip')
    parser.add_argument('--seq_length', type=int, default=3, help='seq_length')
    parser.add_argument('--predict_len', type=int, default=12, help='predict_len')
    parser.add_argument('--scheduler', action='store_true', help='scheduler')
    parser.add_argument('--mo', type=float, default=0.1, help='momentum')


    # running params
    parser.add_argument('--cuda', action='store_true', help='cuda')
    parser.add_argument('--transpose', action='store_true', help='transpose sequence and feature?')
    parser.add_argument('--runs', type=int, default=1, help='runs')
    parser.add_argument('--fig_filename', type=str, default='./mae', help='fig_filename')

    # choosing gnn model:
    parser.add_argument('--not_using_gnn', action='store_true', help='not_using_gnn')
    parser.add_argument('--gnn_name', type=str, default='gwn', help='gnn_name: gcn or gwn for now.')

    # GNN common params:
    parser.add_argument('--gnn_pooling', type=str, default='gate', help='gnn pooling')
    parser.add_argument('--agg_type', type=str, default='gate', help='gnn pooling')
    parser.add_argument('--gnn_layer_num', type=int, default=3, help='gnn_layer_num')
    parser.add_argument('--gnn_hid_dim', type=int, default=64, help='gnn_hid_dim')
    parser.add_argument('--gnn_out_dim', type=int, default=64, help='gnn_out_dim')
    parser.add_argument('--gnn_fin_fout', type=str, default='1100,550;550,128;128,128',
                        help='gnn_fin_fout for each layer')
    parser.add_argument('--gnn_res', action='store_true', help='gnn_res')
    parser.add_argument('--gnn_adj_type',  type=str, default='None', help='gnn_adj_type')
    parser.add_argument('--gnn_downsample_dim', type=int, default=0, help='gnn_downsample_dim')


    # gwn model params
    parser.add_argument('--coarsen_switch', type=int, default=3,
                        help='coarsen_switch: 0: sum, 1: gated, 2: avg, 3: concat.')
    parser.add_argument('--using_cnn', action='store_true', help='using_cnn')
    parser.add_argument('--gate_t', action='store_true', help='gate_t')
    parser.add_argument('--att', action='store_true', help='attention')
    parser.add_argument('--recur', action='store_true', help='recur')
    parser.add_argument('--fusion', action='store_true', help='fusion')
    parser.add_argument('--pretrain', action='store_true', help='pretrain')
    parser.add_argument('--feature_len', type=int, default=3, help='input feature_len')

    parser.add_argument('--gwn_out_features', type=int, default=32, help='gwn_out_features')
    parser.add_argument('--wavelets_num', type=int, default=20, help='wavelets_num')
    parser.add_argument('--rnn_layer_num', type=int, default=2, help='rnn_layer_num')
    parser.add_argument('--rnn_in_channel', type=int, default=32, help='rnn_in_channel')

    parser.add_argument('--rnn', action='store_true', help='attention')
    parser.add_argument('--bidirect', action='store_true', help='bidirect')


    # gcn params 
    parser.add_argument('--gcn_out_features', type=int, default=32, help='gcn_out_features')
    parser.add_argument('--rnn_hidden_len', type=int, default=32, help='rnn_hidden_len')
    parser.add_argument('--max_diffusion_step', type=int, default=2, help='max_diffusion_step')

    # eeg params
    parser.add_argument('--eeg_seq_len', type=int, default=250, help='eeg_seq_len')
    parser.add_argument('--predict_class_num', type=int, default=4, help='predict_class_num')

    # NOTE: encoder param:gnn_res
    parser.add_argument('--encoder', type=str, default='gnn', help='encoder')
    parser.add_argument('--encoder_hid_dim', type=int, default=256, help='encoder_out_dim')

    # NOTE: decoder param:
    parser.add_argument('--cut_encoder_dim', type=int, default=-1, help='cut_encoder_dim')
    parser.add_argument('--decoder', type=str, default='gnn', help='decoder')
    parser.add_argument('--decoder_type', type=str, default='conv2d', help='decoder_type')
    parser.add_argument('--decoder_downsample', type=int, default=-1, help='decoder_downsample')
    parser.add_argument('--decoder_hid_dim', type=int, default=512, help='decoder_hid_dim')
    parser.add_argument('--decoder_out_dim', type=int, default=32, help='decoder_out_dim')
    
    parser.add_argument('--predictor_num', type=int, default=3, help='predictor_num')
    parser.add_argument('--predictor_hid_dim', type=int, default=512, help='predictor_hid_dim')

    parser.add_argument('--rep_fea_dim', type=int, default=64, help='representation dimension')


    #NOTE: LatentGraphGenerator:
    parser.add_argument('--em_train', action='store_true', help='em_train, alternatively update grad')
    parser.add_argument('--lgg', action='store_true', help='lgg')
    parser.add_argument('--lgg_time', action='store_true', help='lgg time step')
    parser.add_argument('--lgg_warmup', type=int, default=10, help='lgg_warmup')
    parser.add_argument('--lgg_tau', type=float, default=0.01, help='gumbel softmax tau')
    parser.add_argument('--lgg_hid_dim', type=int, default=3, help='lgg_hid_dim')
    parser.add_argument('--lgg_k', type=int, default=3, help='lgg k component')


    # NOTE: DCRNN baseline:
    parser.add_argument('--dcgru_activation', type=str, default='tanh', help='dcgru_activation')

    return parser

def set_grad(m, requires_grad):
    for p in m.parameters():
        p.requires_grad = requires_grad

def freeze_module(m):
    set_grad(m, False)

def unfreeze_module(m):
    set_grad(m, True)


def fill_nan_inf(a:np.ndarray):
    a[np.isnan(a)] = 0
    a[np.isinf(a)] = 0
    return a
    
def matrix_power(m:coo_matrix, pow=1):
    if pow==1:
        return m
    ori_A = m
    for _ in range(pow-1):
        m = m @ ori_A
    m[m>1]=1
    return m



def numpy_to_csr(m) -> csr_matrix:
    row, col = np.nonzero(m)
    values = m[row, col]
    csr_m = csr_matrix((values, (row, col)), shape=m.shape)
    return csr_m


class DaoLogger:
    def __init__(self) -> None:
        self.debug_mode = False

    def init(self, args):
        self.args = args
        self.debug_mode = args.debug
    
    def log(self, *paras):
        print('[DLOG] ', *paras)
    
    def debug(self, *paras):
        if self.debug_mode: print('[DEBUG] ', *paras)

DLog = DaoLogger()



# Plot Correlation:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from minepy import pstats, cstats


def get_corrs(normed_vars, cate="all"):
    
    def make_adj(array):
        y = len(array)
        n = int((1+np.sqrt(1+8*y))/2)
        adj = np.ones((n,n))
        arr_index = 0
        for i in range(n):
            for j in range(i, n):
                if i==j:
                    continue
                adj[i, j] = array[j + arr_index - i -1]
                adj[j, i] = adj[i, j]
            arr_index += n - i - 1
        return adj

    corrs = {}
    # fill nan with zeros:
    pd_na = pd.DataFrame(normed_vars)
    
    pd_na.fillna(0.01)
    
    # Pearson:
    corrs_p = pd_na.corr(method='pearson')
    corrs['pearson'] = corrs_p.values
    
    # Spearman:
    corrs_s = pd_na.corr(method='spearman')
    corrs['spearman'] = corrs_s.values
    
        # MIC:
    if cate in ['all', 'MIC']:
        if not isinstance(normed_vars, np.ndarray):
            pd_na = pd.DataFrame(normed_vars)
            pd_na.fillna(0)
            normed_vars = pd_na.values
            
        normed_vars = normed_vars.transpose()
        mic_p, tic_p =  pstats(normed_vars, alpha=0.6, c=5, est="mic_e")
        cr = np.array(make_adj(tic_p))
        corrs['MIC'] = cr
    
    if cate == "all":
        return corrs

    return corrs[cate]


        
def normalize(data, along_axis=None, ignore_norm=[], same_data_shape=True):
    '''
        only norm numpy type data with last dimension.
    '''
    if isinstance(data, list):
        if same_data_shape:
            # hear also along each axis:
            cur_data = np.concatenate(np.array(data).reshape(-1, 1), axis=0)
            return normalize(cur_data, along_axis=along_axis, ignore_norm=ignore_norm)
        else:
        # NOTE: data shape: [(N, C), (N1, C),...], so cannot concatenate
            normed_res = []
            for each in data:
                each_norm = normalize(each, along_axis=along_axis, ignore_norm=ignore_norm)
                normed_res.append(each_norm)
            return normed_res
        
    if not isinstance(data, np.ndarray):
        data = data.cpu().numpy()
        
    if along_axis is not None:
        if along_axis == -1:
            # along all axis separately. data shape:(NxC) along each C_i
            for ax in range(data.shape[-1]):
                if ax in ignore_norm or (ax - data.shape[-1]) in ignore_norm:
                    continue
                mean = np.mean(data[:, ax])
                std = np.std(data[:, ax])
                scaler = StandardScaler(mean=mean, std=std)
                data[:, ax] = scaler.transform(data[:, ax])
            return data
        else:
            print('norm along ', along_axis)
            pass
            
    mean = np.mean(data)
    std = np.std(data)
    scaler = StandardScaler(mean=mean, std=std)
    normed_data = scaler.transform(data)
                
    return normed_data, scaler

from torch.utils.data import Dataset


def flatten_list(nest_list:list):
    return [j for i in nest_list for j in flatten_list(i)] if isinstance(nest_list, list) else [nest_list]

def append_tag(ori_tag, tag, sep='_'):
    return f'{ori_tag}{sep}{tag}'

def random_split_dataset(torch_dataset:Dataset, ratios: list, classification=True):
    """input: ratios: [prop_train, prop_val, prop_test] or [prop_train, prop_teset], sum is 1.
    """
    p = np.array(ratios)
    assert np.sum(p) == 1.0 and len(ratios) > 1
    
    if classification:
        # NOTE: for each class, the ratio should be kept.
        class_datasets = defaultdict(list)
        for x, y in torch_dataset:
            class_datasets[y.item()].append(x)
            
        if len(ratios) == 2:
            train_dataset_x = []
            train_dataset_y = []
            val_dataset_x = []
            val_dataset_y = []
            for k, v in class_datasets.items():
                random.shuffle(v)
                train_num = int(len(v)*ratios[0])
                train_dataset_x.append(v[:train_num])
                train_dataset_y.append(torch.from_numpy(np.repeat(np.array([k]), train_num)).long())
                val_dataset_x.append(v[train_num:])
                val_dataset_y.append(torch.from_numpy(np.repeat(np.array([k]), len(v)-train_num)).long())
                
                
            train_x = flatten_list(train_dataset_x)
            train_y = torch.cat(train_dataset_y, dim=0).squeeze()
            val_x = flatten_list(val_dataset_x)
            val_y = torch.cat(val_dataset_y, dim=0).squeeze()
            return train_x, train_y, val_x, val_y
        
        else:
            train_dataset_x = []
            train_dataset_y = []
            val_dataset_x = []
            val_dataset_y = []
            test_dataset_x = []
            test_dataset_y = []
            for k, v in class_datasets.items():
                random.shuffle(v)
                train_num = int(len(v)*ratios[0])
                train_dataset_x.append(v[:train_num])
                print(torch.LongTensor(k).repeat(train_num).unsqueeze(dim=1).shape)
                train_dataset_y.append(torch.from_numpy(np.repeat(np.array([k]), train_num)).long())
                
                val_num = int(v) * ratios[1]
                val_dataset_x.append(v[train_num:train_num+val_num])
                val_dataset_y.append(torch.from_numpy(np.repeat(np.array([k]), val_num)).long())
                
                test_x = v[train_num+val_num:]
                test_dataset_x.append(test_x)
                test_dataset_y.append(torch.from_numpy(np.repeat(np.array([k]), len(test_x))).long())
                
                
            train_x = flatten_list(train_dataset_x)
            train_y = torch.cat(train_dataset_y, dim=0).squeeze()
            val_x = flatten_list(val_dataset_x)
            val_y = torch.cat(val_dataset_y, dim=0).squeeze()
            test_x = flatten_list(test_dataset_x)
            test_y = torch.cat(test_dataset_y, dim=0).squeeze()
            return train_x, train_y, val_x, val_y, test_x, test_y
    else:
        # just split the x.
        pass
    
    
        

class SeqDataLoader(object):
    def __init__(self, xs, ys, batch_size, cuda=False, pad_with_last_sample=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            # batch
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size) + 1
        if (self.num_batch - 1) * self.batch_size == self.size:
            self.num_batch -= 1

        print('num_batch ', self.num_batch)
        xs = torch.Tensor(xs)
        ys = torch.LongTensor(ys)
        if cuda:
            xs, ys = xs.cuda(), ys.cuda()
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            start_ind = 0
            end_ind = 0
            while self.current_ind < self.num_batch and start_ind <= end_ind and start_ind <= self.size:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()

def norm(tensor_data, dim=0):
    mu = tensor_data.mean(axis=dim, keepdim=True)
    std = tensor_data.std(axis=dim, keepdim=True)
    return (tensor_data - mu) / (std + 0.00005)


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    print('origin adj:', adj)
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def sym_norm_lap(adj):
    N = adj.shape[0]
    adj_norm = sym_adj(adj)
    L = np.eye(N) - adj_norm
    return L


def conv_L(in_len, kernel, stride, padding=0):
    ''' get the convolution output len
    '''
    return int((in_len - kernel + 2 * padding) / stride) + 1

def torch_dense_to_coo_sparse(dense_m):
    idx = torch.nonzero(dense_m).T
    data = dense_m[idx[0],idx[1]]
    coo_m = torch.sparse_coo_tensor(idx, data, dense_m.shape)
    return coo_m
    
            
def get_conv_out_len(in_len, modules):
    from torch import nn
    out_len = 0
    for i in range(len(modules)):
        m = modules[i]
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.AvgPool1d):
            out_len = conv_L(in_len, m.kernel_size[0], m.stride[0])
            in_len = out_len
            print('conv1d: in_len', in_len)
        elif isinstance(m, nn.MaxPool1d):
            out_len = conv_L(in_len, m.kernel_size, m.stride)
            in_len = out_len
            print('Pooling: in_len', in_len)
        elif isinstance(m, nn.Sequential):
            out_len = get_conv_out_len(in_len, m)
            in_len = out_len
        else:
            print("not counting!", m)

    return out_len


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt).toarray()
    normalized_laplacian = sp.eye(adj.shape[0]) - np.matmul(np.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


class StandardScaler():

    def __init__(self, mean, std, fill_zeroes=False):
        self.mean = mean
        self.std = std
        self.fill_zeroes = fill_zeroes

    def transform(self, data):
        if self.fill_zeroes:
            mask = (data == 0)
            data[mask] = self.mean
            
        if isinstance(data, list):
            if self.std == 0:
                return [0.01 for d in data]
            
            return [(d - self.mean)/self.std for d in data]
        
        if self.std == 0:
            return np.zeros_like(data) + 0.01
        
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


ADJ_CHOICES = ['laplacian', 'origin', 'scalap', 'normlap', 'symnadj', 'transition', 'identity','er']


def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)

    adj_mx[adj_mx > 0] = 1
    adj_mx[adj_mx < 0] = 0

    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32)]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "sym_norm_lap":
        adj = [sym_norm_lap(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj

def calc_eeg_accuracy(preds, labels):
    """
    ACC, R, F1, MPR.
    """
    # return whole acc and each acc:
    num = preds.size(0)
    preds_b = preds.argmax(dim=1).squeeze()
    labels = labels.squeeze()
    ones = torch.zeros(num)
#     print(preds.shape, labels.shape)
    ones[preds_b == labels] = 1
    acc = torch.sum(ones) / num

    preds_dict = dict(Counter(preds_b))
    labels_dict = dict(Counter(labels))
    acc_each_dict = {}
    for k, v in labels_dict.items():
        acc_each_dict[k] = preds_dict[k]/v if k in preds_dict else 0
    

    return acc



def calc_metrics(preds, labels, null_val=0.):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    # handle all zeros.
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    mse = (preds - labels) ** 2
    mae = torch.abs(preds - labels)
    mape = mae / labels
    mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
    rmse = torch.sqrt(mse)
    return mae, mape, rmse


def calc_metrics_eeg(preds, labels, criterion):
    labels = labels.squeeze()
    b = preds.shape[0]
    loss = criterion(preds, labels)
    return loss


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def correlation_map(data):
    data = np.array(data)
    d_shape = data.shape
    print('input data shape:', d_shape)  # (15, 45, 62, 300, 5)
    # 1. take subject - trial - channel, so the data will be: (N x seq)
    for sub in data:
        # take out one trial:
        t = sub[0]
        chan_len = d_shape[-1]
        print('total chan_len: ', chan_len)
        for i in range(d_shape[-1]):
            chan = t[..., i].transpose()

            # plot correaltion map
            pd_chan = pd.DataFrame(chan)
            print(pd_chan.corr())


def load_eeg_adj(adj_filename, adjtype=None):
    if 'npy' in adj_filename:
        adj = np.load(adj_filename)
    else:
        adj = np.genfromtxt(adj_filename, delimiter=',')
    adj_mx = np.asarray(adj)
    if adjtype in ["scalap", 'laplacian']:
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "sym_norm_lap":
        adj = [sym_norm_lap(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adjtype == "origin":
        return adj_mx

    return adj[0]



class Trainer:
    def __init__(self, model, optimizer=None, loss_cal=None, sched=None, scaler:StandardScaler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_cal = loss_cal
        self.scheduler = sched
        self.scaler = scaler

    def lr_schedule(self):
        self.scheduler.step()

    def train(self, input_data, target):
        self.model.train()
        self.optimizer.zero_grad()

        # train
        output = self.model(input_data)

        output = output.squeeze()
        if self.scaler is not None:
            output = self.scaler.inverse_transform(output)
            
        loss = self.loss_cal(output, target)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item(), output.detach()

    def eval(self, input_data, target):
        self.model.eval()

        output = self.model(input_data)  # [batch_size,seq_length,num_nodes]
        output = output.squeeze()
        loss = self.loss_cal(output, target)
        return loss.item(), output.detach()


class FocalLoss(nn.Module):
    def __init__(self, celoss=None, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.celoss = celoss
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        if self.celoss is None:
            loss = F.cross_entropy(inputs,  targets, reduce=False)
        else:
            loss = self.celoss(inputs,  targets)
        p = torch.exp(-loss)
        flloss = torch.mean(self.alpha * torch.pow((1-p), self.gamma) * loss)
        return flloss

from sklearn.metrics import f1_score

def cal_f1(preds, labels):
    mi_f1 = f1_score(labels, preds, average='micro')
    ma_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')

    return mi_f1, ma_f1, weighted_f1

def conv_L(in_len, kernel, stride, padding=0):
    return int((in_len - kernel + 2 * padding) / stride + 1)


def cal_cnn_outlen(modules, in_len, height=True):
    conv_l = in_len
    pos = 0 if height else 1
    for m in modules:
        if isinstance(m, nn.Conv2d):
            conv_l = conv_L(in_len, m.kernel_size[pos], m.stride[0], m.padding[pos])
            in_len = conv_l
        if not height:
            if isinstance(m, nn.Conv1d):
                conv_l = conv_L(in_len, m.kernel_size[0], m.stride[0], m.padding[0])
                in_len = conv_l
                
        if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
            conv_l = conv_L(in_len, m.kernel_size[pos], m.stride, m.padding)
            in_len = conv_l                
    return conv_l


# def plot_confused_cal_f1(preds, labels, fig_dir):
#     preds = preds.cpu()
#     labels = labels.cpu()
    
#     ori_preds = preds
#     sns.set()
#     fig = plt.figure(figsize=(5, 4), dpi=100)
#     ax = fig.gca()
#     gts = [number_label_dict[int(l)][:-2] for l in labels]
#     preds = [number_label_dict[int(l)][:-2] for l in preds]
    
#     label_names = [v[:-2] for v in number_label_dict.values()]
#     print(label_names)
#     C2= np.around(confusion_matrix(gts, preds, labels=label_names, normalize='true'), decimals=2)

#     # from confusion to ACC, micro-F1, macro-F1, weighted-f1.
#     print('Confusion:', C2)
#     mi_f1, ma_f1, w_f1 = cal_f1(ori_preds, labels)
#     print(f'micro f1: {mi_f1}, macro f1: {ma_f1}, weighted f1: {w_f1}')

#     sns.heatmap(C2, cbar=True, annot=True, ax=ax, cmap="YlGnBu", square=True,annot_kws={"size":9},
#         yticklabels=label_names,xticklabels=label_names)

#     ax.figure.savefig(fig_dir, transparent=False, bbox_inches='tight')
    
    
# def resource_allocation(adj_matrix, link_list, batch_size=32768, cate_index=None, cate_degree=None):
#     '''
#     0:cn neighbor
#     1:aa
#     2:ra
#     '''
#     A = adj_matrix
#     w = 1 / A.sum(axis=0)
#     w[np.isinf(w)] = 0
#     w1 = A.sum(axis=0) / A.sum(axis=0)
#     print('w1 shape', w1.shape)
#     print('w1:', w1[0])
#     temp = np.log(A.sum(axis=0))
#     temp = 1 / temp
#     temp[np.isinf(temp)] = 1
#     D_log = A.multiply(temp).tocsr()
#     D = A.multiply(w).tocsr()
#     D_common = A.multiply(w1).tocsr()
#     print('D_common shape:', D_common.shape)
#     print('D_common', D_common[0])
#     link_index = link_list.t()  # (2,:)
#     link_loader = DataLoader(range(link_index.size(1)), batch_size)
#     ra = []
#     cn = []
#     aa = []

#     for idx in tqdm(link_loader):
#         src, dst = link_index[0, idx], link_index[1, idx]
#         ra.append(np.array(np.sum(A[src].multiply(D[dst]), 1)).flatten())
#         aa.append(np.array(np.sum(A[src].multiply(D_log[dst]), 1)).flatten())
#         cn.append(np.array(np.sum(A[src].multiply(D_common[dst]), 1)).flatten())
#         # break

#     cn = np.concatenate(cn, 0)
#     ra = np.concatenate(ra, 0)
#     aa = np.concatenate(aa, 0)
#     return torch.FloatTensor(cn), torch.FloatTensor(ra), torch.FloatTensor(aa)
