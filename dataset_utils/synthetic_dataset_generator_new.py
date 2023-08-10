# %% [markdown]
# # Given correlation coefficient r of X and Y, and given X, now generate Y.

# %%
# plot results 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap
import networkx as nx

# here..
cmaps = {}

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

x_y_label_font = 20
x_y_legend_font = 20

plt.rc('font', family='Times New Roman')
fig_dpi = 220
fig_shape_squre = (6, 5)

def plot_color_gradients(category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh), dpi=100)
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list
    plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn import preprocessing


def z_norm(x_data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(x_data)

def mean_norm(x_data):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(x_data)


def generate_Y_from_X(X1, X2, X3, r1, r2, r3):
    """
    Generates values for variable Y from {0, 1, 2} based on the given independent variables X1, X2, X3 and their
    corresponding correlation coefficients r1, r2, r3.
    
    Args:
        X1 (ndarray): Sample of variable X1 from normal distribution.
        X2 (ndarray): Sample of variable X2 from normal distribution.
        X3 (ndarray): Sample of variable X3 from normal distribution.
        r1 (float): Correlation coefficient between X1 and Y.
        r2 (float): Correlation coefficient between X2 and Y.
        r3 (float): Correlation coefficient between X3 and Y.
        
    Returns:
        ndarray: Generated values for variable Y.
    """
    # Calculate the mean and standard deviation of X1, X2, X3
    X1= X1.squeeze()
    X2= X2.squeeze()
    X3= X3.squeeze()
    
    mean_X1 = np.mean(X1)
    mean_X2 = np.mean(X2)
    mean_X3 = np.mean(X3)
    
    print('mean_X1: ', mean_X1)
    print('mean_X2: ', mean_X2)
    print('mean_X3: ', mean_X3)
    
    std_X1 = np.std(X1)
    std_X2 = np.std(X2)
    std_X3 = np.std(X3)
    
    # Calculate the standard deviation of Y
    std_Y = np.sqrt((std_X1**2 * r1**2 + std_X2**2 * r2**2 + std_X3**2 * r3**2) / (r1**2 + r2**2 + r3**2 + 2*r1*r2*r3))
    
    # Generate values for Y from normal distribution with mean 0 and calculated standard deviation
    # Y = np.random.normal(loc=0, scale=std_Y, size=len(X1))
    
    # Add the means of X1, X2, X3 and the calculated standard deviation of Y to the generated values to get the final values for Y
    # Y = mean_X1 + mean_X2 + mean_X3 + r1 * ((X1 - mean_X1) * std_Y / std_X1) + r2 * ((X2 - mean_X2) * std_Y / std_X2) + r3 * ((X3 - mean_X3) * std_Y / std_X3)
    Y = r1 * ((X1 - mean_X1) * std_Y / std_X1) \
        + r2 * ((X2 - mean_X2) * std_Y / std_X2) \
        + r3 * ((X3 - mean_X3) * std_Y / std_X3)
    
    # Round the generated values of Y to the nearest integer and clip them to be within {0, 1, 2}
    # Y = np.round(Y * 49 + 50).clip(0, 2).astype(int)
    Y = z_norm(Y.reshape(-1, 1))
    Y = Y.squeeze()
    Y = np.round(Y * 10).astype(int)
    
    return Y

# Generate samples of X1, X2, X3 with sample size 100 from normal distribution

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn import preprocessing
from functools import reduce

def z_norm(x_data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(x_data), min_max_scaler

def mean_norm(x_data):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(x_data)


sigmas = [1, 2, 3]

def get_XY(uniform=True, mix=False, class_num=10):
        
    sample_size =  5000
    
    if not uniform:
        ns = [np.random.normal(loc=0, scale=sigmas[i], size=sample_size) for i in range(len(sigmas))]
    else:
        ns = [np.random.uniform(0, 1, size=sample_size) for _ in range(len(sigmas))]

    if mix:
        ns[-1] =  np.random.normal(loc=0, scale=1, size=sample_size) if uniform else np.random.uniform(-np.sqrt(3), np.sqrt(3), size=sample_size)
    # Specify the correlation coefficients between X1, X2, X3, and Y

    rs = [-0.4, 0.1, 0.7]
    
    sum_rs = np.sum([r**2 for r in rs])
    print('sum_rs:', sum_rs)

    scale = np.sqrt(12) if uniform else 1
    scale = 1

    sigma_y = 1

    r5 = np.sqrt(1 - np.sum([r**2 for r in rs]))
    rs.extend([r5])

    Xs = [1*ns[i] for i in range(len(sigmas))]
    # Xs = [sigmas[i]*ns[i] for i in range(len(sigmas))]
    
    Y = scale * sigma_y * (reduce(lambda x, y: x+y, map(lambda x: x[0]*x[1], zip(rs, ns))))

    Y, _ = z_norm(Y.reshape(-1, 1))
    
    Y = Y.squeeze()
    Y = np.round(Y * class_num).astype(int)
    
    return Xs, Y


# %%

# # ROUND:


# %%

# # ROUND:
# Y = np.round(Y).astype(int)
from collections import Counter

# %% [markdown]
# # NOW, generate graphs with - $\color{red}{given\ properties}$
# - $\color{red}{NOTE}$ that, the class is imbalanced.
# - $\color{red}{LIST}$ all properties with fixed N: average degree, CC, density, triangles, 4-cycles, 6-cycles, 8-cycles.

# %%
from functools import reduce
from scipy.sparse import csr_matrix
import random

def rewire_given_cc(desired_cc, G, node_num):
    # Calculate the number of triangles needed to achieve the desired clustering coefficient
    adj = nx.adjacency_matrix(G)
    degrees = [i-1 for i in np.sum(adj, axis=1) if i > 1]
    total_triads = reduce(lambda x, y:x+y, map(lambda x: x*(x-1)/2, degrees))
    desired_triads = int(desired_cc * total_triads)
    actual_triads = sum(nx.triangles(G).values()) // 3
    triads_to_add = desired_triads - actual_triads
    # Add triangles to the graph
    while triads_to_add > 0:
        u, v = random.sample(G.nodes(), k=2)
        if not G.has_edge(u, v) and not G.has_edge(v, u) and not u == v:
            neighbors_u = set(G.neighbors(u))
            neighbors_v = set(G.neighbors(v))
            common_neighbors = neighbors_u.intersection(neighbors_v)
            if len(common_neighbors) > 0:
                w = random.choice(list(common_neighbors))
                G.add_edge(u, v)
                G.add_edge(u, w)
                G.add_edge(v, w)
                triads_to_add -= 1
            
    return G


def add_triangles(G, target_cc, tris_num):
    
    er_nodes = list(G.nodes)
    node_num = len(er_nodes)
    
    tris = [nx.complete_graph(3) for _ in range(tris_num)]
    label_id = node_num
    for i, tr in enumerate(tris):
        tris[i] = nx.relabel_nodes(tr, {0: label_id, 1: label_id+1, 2: label_id+2})
        label_id+=3

    for tr in tris:
        tr_nodes = list(tr.nodes)
        G.add_edge(random.choice(er_nodes), random.choice(tr_nodes))

    for tr in tris:
        G = nx.compose(G, tr)
    
    return G


def get_Y(ns,  rs = None, is_uniform=True):
    
    sum_rs = np.sum([r**2 for r in rs])
    print('sum_rs:', sum_rs)

    scale = np.sqrt(12) if is_uniform else 1
    sigma_y = 1

    r_y = np.sqrt(1 - np.sum([r**2 for r in rs]))
    rs.extend([r_y])

    Y = scale * sigma_y * (reduce(lambda x, y: x+y, map(lambda x: x[0]*x[1], zip(rs, ns))))

    Y, _ = z_norm(Y.reshape(-1, 1))
    Y = Y.squeeze()
    Y = np.round(Y * 10).astype(int)
    
    return Y



# %%
import random
from functools import reduce
import pickle as pk
import os

from sklearn import preprocessing
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
from scipy.stats import pearsonr


# fixed node num, fixed average degree, CC ~ U(0.1, 0.5)
def z_norm(x_data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(x_data), min_max_scaler

def mean_norm(x_data):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(x_data)


def connect_graphs(g1, g2):
    n1 = list(g1.nodes)
    n2 = list(g2.nodes)
    e1 = random.choices(n1, k=1)[0]
    e2 = random.choices(n2, k=1)[0]
    g_cur = nx.compose(g1, g2)
    g_cur.add_edge(e1, e2)
    return g_cur


def random_connect_graph(graph_list:list):
    # NOTE: relabeling the nodes.
    
    new_graphs = []
    np.random.shuffle(graph_list)
    node_idx = 0
    for g in graph_list:
        len_nodes = len(list(g.nodes))
        mapping = {}
        for i in range(len_nodes):
            mapping[i] = i+node_idx
        new_g = nx.relabel_nodes(g, mapping)
        new_graphs.append(new_g)
        node_idx += len_nodes
        
    g_all = reduce(connect_graphs, new_graphs)
    return g_all


def add_square(G, sq_num):
     
    er_nodes = list(G.nodes)
    node_num = len(er_nodes)
    
    added = [nx.cycle_graph(4) for _ in range(sq_num)]
    
    # line graphs:
    
    label_id = node_num
    for i, tr in enumerate(added):
        added[i] = nx.relabel_nodes(tr, {0: label_id, 1: label_id+1, 2: label_id+2, 3:label_id+3})
        label_id+=4

    for tr in added:
        tr_nodes = list(tr.nodes)
        G.add_edge(random.choice(er_nodes), random.choice(tr_nodes))
        er_nodes = list(G.nodes)

    for tr in added:
        G = nx.compose(G, tr)
    
    return G


def add_triangles(G, tris_num, tri=None):
    
    er_nodes = list(G.nodes)
    node_num = len(er_nodes)
    
    if tri is None:
        tris = [nx.complete_graph(3) for _ in range(tris_num)]
    else:
        # copy tri:
        tris = [tri.copy() for _ in range(tris_num)]
        
    label_id = node_num
    for i, tr in enumerate(tris):
        tris[i] = nx.relabel_nodes(tr, {0: label_id, 1: label_id+1, 2: label_id+2})
        label_id+=3

    for tr in tris:
        tr_nodes = list(tr.nodes)
        G.add_edge(random.choice(er_nodes), random.choice(tr_nodes))

    for tr in tris:
        G = nx.compose(G, tr)
    
    return G


def get_Y(ns, class_num, rs = None, is_uniform=True):
    
    sum_rs = np.sum([r**2 for r in rs])
    print('sum_rs:', sum_rs)

    scale = np.sqrt(12) if is_uniform else 1
    sigma_y = 1

    r_y = np.sqrt(1 - np.sum([r**2 for r in rs]))
    rs.extend([r_y])
        
    Y = scale * sigma_y * (reduce(lambda x, y: x+y, map(lambda x: x[0]*x[1], zip(rs, ns))))

    Y, _ = z_norm(Y.reshape(-1, 1))
    Y = Y.squeeze()
    Y = np.round(Y * (class_num-1)).astype(int)
    
    return Y


def convert_to_torch_geometric_data(graphs, Y):
    data_list = []
    for i, graph in enumerate(graphs):
        
        nx.set_node_attributes(graph, torch.ones(1), 'x')
        # nx.set_edge_attributes(graph, torch.randn(graph.number_of_edges(), 1), 'edge_attr')
        data = from_networkx(graph)
        data.y = torch.tensor([Y[i]], dtype=torch.long)
        data_list.append(data)
    return data_list


class SynDataset(InMemoryDataset):
    def __init__(self, data=None, name=None, root=None, transform=None, pre_transform=None):
        super(SynDataset, self).__init__(root, transform, pre_transform)
        if data is None:
            data_path = os.path.join(root, f'syn_{name}.pkl')
            with open(data_path, 'rb') as f:
                data = pk.load(f)
                
        self.num_tasks = len({int(i.y.item()) for i in data})
        self.data, self.slices = self.collate(data)
        self.name = name
        self.root = root

    def _download(self):
        pass

    def _process(self):
        pass

def graph_avg_degree(adj):
    if not isinstance(adj, np.ndarray):
        adj = adj.todense()
    degrees = np.sum(adj, axis=1).reshape(adj.shape[0], 1)
    mean_D = np.mean(degrees).astype(np.float32).reshape(1)
    return mean_D


def dump_ER_graphs_by_Degree(sample_num, class_num, rs, er_nodes=100, min_D=5, max_D=50, name=None):
    """
    return labels and graphs.
    """
    
    ori_dd = np.random.uniform(min_D, max_D, size=(sample_num,1))
    normed_dd, scaler = z_norm(ori_dd)
    
    n_dd = np.random.uniform(0, 1, size=(sample_num, 1))
    # n_cc = np.random.normal(loc=0, scale=1, size=(sample_num, 1))
    
    dd = scaler.inverse_transform(n_dd)
 
    ps = dd / er_nodes
    
    samples = []
    
    rand_p = np.random.normal(loc=0, scale=0.01, size=(sample_num,))
    rand_node = np.random.normal(loc=0, scale=3, size=(sample_num,))
    
    cc_cur = []
    for i in range(sample_num):
        G = nx.erdos_renyi_graph(er_nodes + abs(int(rand_node[i])), abs(rand_p[i]) + ps[i])
        cc_cur.append(nx.average_clustering(G))
        samples.append(G)
  
    n_y = np.random.uniform(0, 1, size=(sample_num, 1))
    Y = get_Y([n_dd, n_y], class_num, rs=rs, is_uniform=True)
    print(Y.shape)
    print('len samples:', len(samples))
    
    # if abs(corr_degree) > 0.1, regenerate the samples.
    
    avg_degree = np.array([graph_avg_degree(nx.to_numpy_matrix(g)) for g in samples])
    
    corr_degree, _ = pearsonr(avg_degree.reshape(-1, 1).squeeze(), Y.squeeze())
    corr_cc, _ = pearsonr(np.array(cc_cur), Y.squeeze())
    # corr_degree = np.corrcoef(Y, avg_degree)[0, 1]
    print('corr_cc', corr_cc, ' corr_degree:', corr_degree)
   
    
    # calculate the pearson correlation coefficient of Y and avg_degree:
    # Calculate the average degree using the average_degree_connectivity() function
    # avg_degree_fix = 2 * np.ones((100,)) + np.random.normal(loc=0, scale=0.1, size=(100,))
    # corr_degree_fix = np.corrcoef(Y, avg_degree_fix)[0, 1]
    # print('corr_degree: ', corr_degree, corr_degree_fix)
    # print('corr new_cc:', np.corrcoef(Y, new_cc)[0, 1])
    # plt.figure()
    # plt.plot(avg_degree[sort_idx])
    # plt.plot(Y[sort_idx])
    # plt.title('avg_degree')
    
    pyg_data = convert_to_torch_geometric_data(samples, Y)
    
    root = 'DATA'
    name = f'degree_{rs[0]}' if name is None else name
    if not os.path.exists(root):
        os.mkdir(root)
    data_path = os.path.join(root, f'syn_{name}.pkl')
    with open(data_path, 'wb') as f:
        pk.dump(pyg_data, f)
    return True


def dump_ER_graphs_by_CC(sample_num, class_num, rs, er_nodes=100, min_CC=0.1, max_CC=0.5, name=None):
    """
    return labels and graphs.
    """
    ori_cc = np.random.uniform(min_CC, max_CC, size=(sample_num,1))
    normed_cc, scaler = z_norm(ori_cc)
    
    n_cc = np.random.uniform(0, 1, size=(sample_num, 1))
    # n_cc = np.random.normal(loc=0, scale=1, size=(sample_num, 1))
    
    cc = scaler.inverse_transform(n_cc)
    
    # # NOTE: plot the distribution of CC:
    # sort_idx = np.argsort(cc.squeeze())
    # plt.figure()
    # plt.plot(ori_cc[sort_idx])
    # plt.figure()
    # plt.plot(cc[sort_idx])
    
    tris = cc * 30
    
    samples = []
    rand_p = np.random.normal(loc=0, scale=0.05, size=(sample_num,))
    rand_node = np.random.normal(loc=0, scale=3, size=(sample_num,))
    
    cc_cur = []
    for i in range(sample_num):
        tr_num = int(tris[i])
        G = nx.erdos_renyi_graph(er_nodes - tr_num * 3 + abs(int(rand_node[i])), abs(rand_p[i]) - tr_num * 0.01)
        new_G = add_triangles(G, tr_num)
        cc_cur.append(nx.average_clustering(new_G))
        samples.append(new_G)
        
    # NOTE: check labels:
    # new_cc = np.array([nx.average_clustering(g) for g in samples])
    # plt.figure()
    # plt.plot(new_cc[sort_idx])
    
    n_y = np.random.uniform(0, 1, size=(sample_num, 1))
    Y = get_Y([n_cc, n_y], class_num, rs=rs, is_uniform=True)
    print(Y.shape)
    print('len samples:', len(samples))
    
    # if abs(corr_degree) > 0.1, regenerate the samples.
    
    avg_degree = np.array([graph_avg_degree(nx.to_numpy_matrix(g)) for g in samples])
    
    corr_degree, _ = pearsonr(avg_degree.reshape(-1, 1).squeeze(), Y.squeeze())
    corr_cc, _ = pearsonr(np.array(cc_cur), Y.squeeze())
    # corr_degree = np.corrcoef(Y, avg_degree)[0, 1]
    print('corr_cc', corr_cc)
    
    if abs(corr_degree) >= 0.1:
        print('no ok:', corr_degree)
        return False
    else:
        print('corr:', corr_degree)
    
    
    pyg_data = convert_to_torch_geometric_data(samples, Y)
    
    root = 'DATA'
    name = f'cc_{rs[0]}' if name is None else name
    if not os.path.exists(root):
        os.mkdir(root)
    data_path = os.path.join(root, f'syn_{name}.pkl')
    with open(data_path, 'wb') as f:
        pk.dump(pyg_data, f)
    return True

# save datasets
import pickle as pk

def save_datasets(datasets, file_name):
    with open(file_name, 'wb') as f:
        pk.dump(datasets, f)

def load_datasets(file_name):
    with open(file_name, 'rb') as f:
        datasets = pk.load(f)
    return datasets

def dump_Degree_syn(sample_num, class_num):
    for i in range(1, 10):
        correlation = i/10
        print(correlation)
        ok = dump_ER_graphs_by_Degree(sample_num, class_num, rs=[correlation], name=f'degree_{correlation}_{round(correlation, 1)}_class{class_num}')


sample_num = 2048
class_num = 2
dump_Degree_syn(sample_num, class_num)