import numpy as np
import networkx as nx
from utils import utils
import os

DATASETS_NAMES = {
    'REDDIT-BINARY',
    'REDDIT-MULTI-5K',
    'COLLAB',
    'IMDB-BINARY',
    'IMDB-MULTI',
    'NCI1',
    'ENZYMES',
    'PROTEINS',
    'DD',
    "MUTAG",
    'CSL'
}

def xargs(f):
    def wrap(**xargs):
        return f(**xargs)
    return wrap

@xargs
def graph_cycle_feature(adj, k:str="4"):
    ks = [int(i) for i  in k.split('-')]
    nx_g = nx.from_numpy_array(adj)
    cycles = nx.cycle_basis(nx_g)
    cycle_count = {}
    for i in ks:
        cycle_count[i] = 0
    
    for c in cycles:
        if len(c) in ks:
            cycle_count[len(c)] += 1
    # NOTE: cycle_count to array list
    graph_features = np.zeros(len(ks))
    for i, k in enumerate(ks):
        graph_features[i] = cycle_count[k]
        
    return graph_features.astype(np.float32).reshape(len(ks))
       
@xargs
def node_cycle_feature(adj, k=4):
    
    nx_g = nx.from_numpy_array(adj)
    cycles = nx.cycle_basis(nx_g)

    # collect all len 4 sets.
    # 
    node_fea = np.zeros((adj.shape[0], 1))
    for c in cycles:
        if len(c) == k:
            for id in c:
                node_fea[id] += 1
        
    return node_fea.astype(np.float32)



    
@xargs
def node_tri_cycles_feature(adj, k=2):
    """ A^k as node features. so the dim of feature equals to the number of nodes.
    """

    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    adj = np.multiply(adj, np.matmul(adj, adj))
    adj = np.sum(adj, axis=1).reshape(-1, 1)
    return adj.astype(np.float32)

@xargs
def node_k_adj_feature(adj, k=2):
        
    """ A^k as node features. so the dim of feature equals to the number of nodes.
    """
    if not isinstance(k, int):
        k = int(k)
        
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    ori_adj = adj
    for _ in range(k-1):
        adj = np.matmul(adj, ori_adj)
    return adj.astype(np.float32)


# name='imdb_degree_dist_shuffled.npy'
name='dd_degree_dist_shuffled.npy'
# name='proteins_degree_dist_shuffled.npy'
# name='imdb_degree_dist.npy'
# name='proteins_degree_dist.npy'
# name='dd_degree_dist.npy'

"""
COLLAB_degree_dist.npy
COLLAB_degree_dist_shuffled.npy
ENZYMES_degree_dist.npy
ENZYMES_degree_dist_shuffled.npy
IMDB-MULTI_degree_dist.npy
IMDB-MULTI_degree_dist_shuffled.npy
NCI1_degree_dist.npy
NCI1_degree_dist_shuffled.npy
dd_degree_dist.npy
dd_degree_dist_shuffled.npy
imdb_degree_dist.npy
imdb_degree_dist_shuffled.npy
mutag_degree_dist.npy
mutag_degree_dist_shuffled.npy
proteins_degree_dist.npy
proteins_degree_dist_shuffled.npy
"""



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
    
csd_dict = {}
# for d_name in DATASETS_NAMES:
#     f_name = f'{d_name}_degree_dist_shuffled.npy'
#     if os.path.exists(f_name):
#         csd_dict[d_name] = MyIter(np.load(f_name))
#         print(f'load node feauture: {f_name}\n')



@xargs
def node_degree_feature(adj, name=None, checkpoint=False):
    """ node (weighted, if its weighted adjacency matrix) degree as the node feature.
    """
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    N = adj.shape[0]
    if checkpoint:
        degrees = np.array([csd_dict[name].__next__().item() for _ in range(N)]).reshape(adj.shape[0], 1)
    else:
        degrees = np.sum(adj, axis=1).reshape(adj.shape[0], 1)
    
    return degrees.astype(np.float32)

@xargs
def node_random_id_feature(adj, total=None, ratio=1.0, dist=None):
        
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()

    N = adj.shape[0]
    if dist is not None:
        if int(dist) == 2:
            id_features = np.random.choice(cds_mutag, size=N).reshape(N, 1).astype(np.float32)
            print('dist 2')
        elif int(dist) == 3:
            id_features = np.array([cds_mutag_iter.__next__() for _ in range(N)]).reshape(N, 1).astype(np.float32)
            print('dist 3')
        elif int(dist) == 4:
            print('dist 4')
            # np.random.shuffle(shuf_idx)
            sample_ids = [s for s in np.random.choice(shuf_idx, size=len(shuf_idx), replace=True)].__iter__()
            new_x = []
            for _ in range(N):
                new_x.append(copy_degree_sequence[sample_ids.__next__().item()])                
            id_features = np.array(new_x).reshape(N, 1).astype(np.float32)
        else:
            dist = [3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,1,1,1,1]
            id_features = np.random.choice(dist, size=N).reshape(N, 1).astype(np.float32)
        
        return id_features
    else:
        total = N if total is None else total
        total = int(total)
        id_features = np.random.randint(1, int(total*ratio), size=N).reshape(N, 1).astype(np.float32)
        return id_features

@xargs
def node_allone_feature(adj):
    """return (N, 1) all one node feature
    """
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
        
    N = adj.shape[0]
    return 0.1 * np.ones(N).reshape(N, 1).astype(np.float32)


@xargs
def node_gaussian_feature(adj, mean_v=0.1, std_v=1.0, dim=1):
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
        
    N = adj.shape[0]
    
    return np.random.normal(loc=mean_v, scale=std_v, size=(N, dim)).astype(np.float32)


@xargs
def node_index_feature(adj):
    """return (N, 1) node feature, feature equals to the index+1
    """
    N = adj.shape[0]
    return np.arange(1, N+1).reshape(N, 1).astype(np.float32)

@xargs
def node_deviated_feature(adj):
    N = adj.shape[0]
    block_N = int(N/2)
    fea1 = np.arange(1, block_N+1).reshape(block_N, 1).astype(np.float32)
    fea2 = 3 * np.arange(block_N+1, N+1).reshape(block_N, 1).astype(np.float32)
    return np.concatenate([fea1, fea2], axis=0)
    

# node clustering coefficient

@xargs
def node_cc_avg_feature(adj):
    g_cur = nx.from_numpy_array(adj)
    feats = nx.average_clustering(g_cur)
    return np.array(feats).astype(np.float32).reshape(1)

@xargs
def node_cc_feature(adj):
    N = adj.shape[0]
    g_cur = nx.from_numpy_array(adj)
    feas_dict = nx.clustering(g_cur)
    feats = []
    for i in range(N):
        feats.append(feas_dict[i])
    feats = np.array(feats).reshape(N, 1).astype(np.float32)
    return feats

@xargs
def graph_stats_degree(adj):
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    degrees = np.sum(adj, axis=1).reshape(adj.shape[0], 1)
    mean_D = np.mean(degrees).astype(np.float32)
    std_D = np.std(degrees).astype(np.float32)
    sum_D = mean_D * adj.shape[0]
    return np.stack([mean_D,std_D,sum_D]).reshape(3)


def downsampling(s:np.ndarray, sample_len=64, pad=True):
    s_len = s.shape[0]
    assert s_len > sample_len
            
    indice = np.linspace(0, s_len-1, sample_len)
    int_idc = [int(i) for i in indice]
    downsampled = np.array([s[i] for i in int_idc])
    
    return int_idc, downsampled
    
    
@xargs
def graph_degree_dist(adj, sample_len=128):
    """ pad the node degree set into the same dimension.
        how? sort and sample 128, if less than 128 then pad.
    """
    if not isinstance(sample_len, int):
        sample_len = int(sample_len)
        
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
        
    degrees = np.sum(adj, axis=1).reshape(adj.shape[0], 1)
    # sort
    degree_sorted = np.sort(degrees)[::-1]
    s_len = degree_sorted.shape[0]
    
    if s_len <= sample_len:
        # pad last value:
        downsampled = np.pad(degree_sorted, pad_width=((0, sample_len-s_len), (0, 0)), mode='minimum')
    else:
        _, downsampled = downsampling(degree_sorted, sample_len=sample_len)
        
    downsampled = downsampled.squeeze().reshape(sample_len).astype(np.float32)
    
    # normlize:
    mean = np.mean(downsampled)
    std = np.std(downsampled)
    
    if std == 0:
        downsampled = downsampled - mean
    else:
        downsampled = (downsampled - mean)/std
    
    return downsampled


@xargs
def graph_cycles_degree(adj):
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    degrees = np.sum(adj, axis=1).reshape(adj.shape[0], 1)
    mean_D = np.mean(degrees).astype(np.float32)
    std_D = np.std(degrees).astype(np.float32)
    sum_D = mean_D * adj.shape[0]
    return np.stack([mean_D,std_D,sum_D]).reshape(3)


@xargs
def graph_invariant(adj):
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    N = adj.shape[0]

    ratio=0.5
    degrees = np.sum(adj, axis=1).reshape(adj.shape[0], 1)
    E = np.sum(degrees).astype(np.float32).reshape(1)
    #N+2E
    return (N+ratio*E).reshape(1)


@xargs
def graph_avg_degree(adj):
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    degrees = np.sum(adj, axis=1).reshape(adj.shape[0], 1)
    mean_D = np.mean(degrees).astype(np.float32).reshape(1)
    return mean_D

@xargs
def graph_avgDN_feature(adj):
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
        
    N = adj.shape[0]
    mean_avg = np.mean(node_degree_feature(adj=adj)).item()
    # degrees = np.sum(adj, axis=1).astype(np.float32).reshape(N, 1)
    # mean_avg = np.mean(degrees).astype(np.float32)
    avgD = mean_avg / N
    return np.array(avgD).astype(np.float32).reshape(1)


def add_graph_features(graph_features, cons_fea_func, c_dim=0):
    """input:
            by default, the graph feature at dim=0 of graph_features (numpy) is the original adj matrix.
            graph_features shape: (B, N, N, C), where C is the graph feature number.
            cons_fea_func is the function to construct new graph features with NxN and append to the C-th dimension.
            
            by default, c_dim is 0, use the first adjacency matrix to construct new features. 
       return:
            graph_features, shape will be (B, N, N, C+1). 
    """
    if graph_features.ndim == 3:
        graph_features = np.expand_dims(graph_features, axis=-1)
        
    new_graph_features = []
    for ori_feature in graph_features[..., c_dim]:
        new_graph_features.append(cons_fea_func(ori_feature))
    
    new_graph_features = np.expand_dims(np.stack(new_graph_features, axis=0), axis=-1)
    
    graph_features = np.concatenate([graph_features, new_graph_features], axis=-1)
    
    return graph_features


def composite_graph_feature_list(graph_features:list):
    """graph_features: list of list, e.g., [fea1, fea2, ...], fea1:NxC1, fea2; NxC2
    """
    return np.concatenate(graph_features, axis=-1)

def composite_node_feature_list(node_features:list, padding=False, padding_len=128, pad_value=0):
    """node_features: list of list, e.g., [fea1, fea2, ...], fea1:[node1, node2,...]
    """
    feas = []
    for i in range(len(node_features[0])):
        each_node = []
        for fea in node_features:
            each_node.append(fea[i])
        each_fea = np.concatenate(each_node, axis=-1)
        if padding:
            each_fea = np.pad(each_fea, ((0,0),(0, padding_len-each_fea.shape[-1])), mode='constant', constant_values=pad_value)
        feas.append(each_fea)
        
    return feas


def composite_node_features(*node_features, padding=False, padding_len=128, pad_value=0):
    """ just concatenate the new_node_features with the cur_node_features (N, C1)
        output new node features: (N, C1+C2)
    """
    if padding is None:
        padding=False
        
    if isinstance(node_features[0], list):
        res = []
        for i in range(len(node_features[0])):
            fea = np.concatenate((node_features[0][i],node_features[1][i]), axis=-1)
            if padding:
                fea = np.pad(fea, ((0,0),(0, padding_len-fea.shape[-1])), mode='constant', constant_values=pad_value)
            res.append(fea)
        return res
    
    fea = np.concatenate(node_features, axis=-1)
    if padding:
        fea = np.pad(fea, ((0,padding_len-fea.shape[-1])), mode='constant', constant_values=pad_value)
        
    return fea

def get_features_by_ids(*indices, cur_features, pad=None):
    if len(indices) < 2:
        return (cur_features[indices[0]][0], cur_features[indices[0]][1])
    
    train_fea = composite_node_features(*tuple([cur_features[i][0] for i in indices]), padding=pad)
    test_fea = composite_node_features(*tuple([cur_features[i][1] for i in indices]), padding=pad)
    return (train_fea, test_fea)



def gen_features(adjs, sparse, cons_func, **xargs):
    if sparse:
        # NOTE: the numbers of Node are different, so need sparse.
        # print('cons_func2:', graph_degree_dist.__name__)
        features = [cons_func(adj=adj, **xargs) for adj in adjs]
        # print('adjs:', adjs[0].shape)
        # a = cons_func(adj=adjs[0], **xargs)
        # b = graph_degree_dist(adj=adjs[0], **xargs)
        # print('a: :', a.shape)
        # print('b: :', b.shape)
        for i in range(len(features)):
            features[i] = utils.fill_nan_inf(features[i])
    else:
        features = np.stack([cons_func(adj=adj, **xargs) for adj in adjs], axis=0)
        features = utils.fill_nan_inf(features)
    
    return features

    
def generate_node_feature(all_data, sparse, node_cons_func, **xargs) -> tuple:
    train_adj, _, test_adj, _ = all_data
    if sparse:
        train_node_feas = [node_cons_func(adj=adj, **xargs) for adj in train_adj]
        test_node_feas = [node_cons_func(adj=adj, **xargs) for adj in test_adj]
    else:
        train_node_feas = np.stack([node_cons_func(adj=adj, **xargs) for adj in train_adj], axis=0)
        test_node_feas = np.stack([node_cons_func(adj=adj, **xargs) for adj in test_adj], axis=0)
    
    return (train_node_feas, test_node_feas) 

 # Shuffle, and then split.
# cc_train_adjs, cc_train_y, cc_test_adjs, cc_test_y

def to_dict(var_str:str):
    d = {}
    for i in var_str.split(';'):
        kv = i.split(':')
        d[kv[0]] = kv[1]
    return d



class GraphFeaRegister(object):
    def __init__(self, file_path=None):
        self.id = id(self)
        self.file_path = file_path
        if file_path is not None:
            self.funcs = {} # TODO: load from file.
            print('no funcs found!')
            pass
        else:
            self.funcs = {
                'stats_degree': graph_stats_degree,
                'degree_dist': graph_degree_dist,
                'avg_degree': graph_avg_degree,
                'avg_cc': node_cc_avg_feature,
                'cycle': graph_cycle_feature,
                'avgd': graph_avgDN_feature,
                'invariant':graph_invariant
                }
        self.registered = []

    def register_by_str(self, arg_str:str=None):
        # arg_str format: name@key:value;key:value....
        print('argstr:', arg_str)
        args = arg_str.split("@")
        print('args:', args)
        
        if len(args)>1:
            self.register(args[0], **to_dict(args[1]))
        else:
            self.register(args[0])
        
    def contains(self, name:str) -> bool:
        for i in self.registered:
            if i[0] == name:
                return True
        return False
    
    def remove(self, re_name):
        del_id = None
        for i, ts in enumerate(self.registered):
            if re_name == ts[0]:
                del_id = i
                break
        if del_id is not None:
            self.registered.pop(del_id)
            print('remove func:', re_name)
        else:
            print('func name not found', re_name)
                
        
    def register(self, func_name, **xargs):
        if func_name not in self.funcs:
            print('func_name:', func_name)
            raise NotImplementedError
        
        self.registered.append((func_name, self.funcs[func_name], xargs))
    
    def get_registered(self):
        return self.registered
    
    def list_registered(self):
        for i, (name, _, arg) in enumerate(self.registered):
            print('index:', i, name, ' args: ',arg)




class NodeFeaRegister(object):
    def __init__(self, file_path=None):
        self.id = id(self)
        self.file_path = file_path
        if file_path is not None:
            self.funcs = {} # TODO: load from file.
            pass
        else:
            self.funcs = {
                "degree":node_degree_feature,
                "allone":node_allone_feature,
                "index_id":node_index_feature,
                "guassian":node_gaussian_feature,
                "tri_cycle":node_tri_cycles_feature,
                "cycle":node_cycle_feature,
                "kadj": node_k_adj_feature,
                "rand_id":node_random_id_feature,
                "graph_stats_degree": graph_stats_degree
                }
        self.registered = []

    def register_by_str(self, arg_str:str=None):
        # arg_str format: func_name@key:value;key:value....
        print('argstr:', arg_str)
        args = arg_str.split("@")
        print('args:', args)
        
        if len(args)>1:
            self.register(args[0], **to_dict(args[1]))
        else:
            self.register(args[0])
        
    def contains(self, name:str) -> bool:
        for i in self.registered:
            if i[0] == name:
                return True
        return False
    
    def remove(self, re_name):
        del_id = None
        for i, ts in enumerate(self.registered):
            if re_name == ts[0]:
                del_id = i
                break
        if del_id is not None:
            self.registered.pop(del_id)
            print('remove func:', re_name)
        else:
            print('func name not found', re_name)
                
        
    def register(self, func_name, **xargs):
        if func_name not in self.funcs:
            print('func_name:', func_name)
            raise NotImplementedError
        
        self.registered.append((func_name, self.funcs[func_name], xargs))
    
    def get_registered(self):
        return self.registered
    
    def list_registered(self):
        for i, (name, _, arg) in enumerate(self.registered):
            print('index:', i, name, ' args: ',arg)

def register_features(adjs, fea_register):
    feature_list = []
    for fea_reg in fea_register.get_registered():
        features = gen_features(adjs, sparse=True, cons_func=fea_reg[1], **fea_reg[2])
        print('features: ', features[0].shape)
        feature_list.append(features)
    return feature_list
    
def construct_node_features(alldata, fea_register:NodeFeaRegister):
    node_feature_list = []
    for fea_reg in fea_register.get_registered():
        node_feature_list.append(generate_node_feature(alldata, sparse=True, node_cons_func=fea_reg[1], **fea_reg[2]))
    return node_feature_list
    