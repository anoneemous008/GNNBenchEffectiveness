from typing import Callable, Union, Optional


import numpy as np
import networkx as nx

import scipy.sparse as sp

import torch
from torch import nn, sparse_coo
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn import Linear
from typing import Union, Tuple
from torch_sparse import SparseTensor
import torch_sparse

from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.data import Data as pyg_Data
from torch_geometric.data import Batch as pyg_Batch
from torch_geometric.utils import add_self_loops, degree

import math

from typing import Sequence

import matplotlib.pyplot as plt



def get_spd_matrix(G, S, max_spd=5):
    spd_matrix = np.zeros((G.number_of_nodes(), len(S)), dtype=np.float32)
    for i, node_S in enumerate(S):
        for node, length in nx.shortest_path_length(G, source=node_S).items():
            spd_matrix[node, i] = min(length, max_spd)
    return spd_matrix


class BaseGraph(object):
    """
        all properties should be torch tensor type.
    """
    def __init__(self, graph_type='default', adj_type='dense', batch=False, batch_num=None, pyg_graph:pyg_Data=None):
        self.graph_type = graph_type
        self.adj_type = adj_type
        self.pyg_graph = pyg_graph
        
        self.ndata = {}  # store node related features.
        self.edata = {}
        self.batch = batch
        self.edge_index = None
        self.batch_num = batch_num
        
        self.A = None
        self.label = None
        self.coo = None
        self.node_num = None
        
    def cuda(self, device='cuda:0'):
        self.device = device
        self.is_cuda = True
        if self.adj_type == 'dense':
            self.A = self.A.cuda()
        
        for k, v in self.ndata.items():
            if v is not None:
                self.ndata[k] = v.cuda()
            
        for k, v in self.edata.items():
            if v is not None:
                self.edata[k] = v.cuda()
        
        
        if self.edge_index is not None:
            self.edge_index = self.edge_index.cuda()
        
        self.pyg_graph.to(device)
            
        return self
    
    def _set_node_num(self, N):
        self.node_num = N
    
    def _set_A(self, A):
        self.A = A
        
    def _set_graph_type(self, g_type):
        self.graph_type = g_type
        
    def _set_coo(self, coo_m):
        self.coo = coo_m
            
    def _set_edge_index(self, edge_index):
        self.edge_index = edge_index
        
    def set_label(self, label:torch.Tensor):
        """for graph level task, classification or regression.
        """
        self.label = label
    
    def get_node_features(self):
        if 'nfeat' in self.ndata:
            return self.ndata['nfeat']
        return None
    
    def get_edge_features(self):
        if 'efeat' in self.edata:
            return self.edata['efeat']
        else:
            return None
        
    def get_edge_index(self):
        return self.edge_index
    
    def set_node_feat(self, node_feat):
        node_feat = torch.from_numpy(node_feat) if isinstance(node_feat, np.ndarray) else node_feat
        if node_feat.dim() == 2:
            assert node_feat.shape[0] == self.node_num
        
        self.ndata['nfeat'] = node_feat.float()
    
    def set_edge_feat(self, edge_feat):
        if edge_feat is None:
            self.edata['efeat'] = None
        else:
            edge_feat = torch.from_numpy(edge_feat) if isinstance(edge_feat, np.ndarray) else edge_feat
            self.edata['efeat'] = edge_feat.float()
        
        
class BaseGraphUtils:
    
    def __init__(self) -> None:
        pass
    
    def from_dense_A(A:torch.Tensor):
        g = BaseGraph(adj_type='dense')
        g._set_node_num(A.shape[0])
        g._set_A(A.float())
        return g
    
    def from_scipy_coo(A:sp.coo.coo_matrix):
        data = A.data
        
        idx_t = torch.LongTensor(np.vstack((A.row, A.col)))
        data_t = torch.FloatTensor(data)
        coo_a = torch.sparse_coo_tensor(idx_t, data_t, A.shape)
        g = BaseGraph(adj_type='coo')
        g._set_node_num(A.shape[0])
        g._set_coo(coo_a)
        return g

        
    def from_numpy(A:np.ndarray) -> BaseGraph:
        """
            A is an adjacency matrix.
            Stored as torch tensor.
        """
        g = BaseGraph(adj_type='dense')
        g._set_node_num(A.shape[0])
        g._set_A(torch.from_numpy(A).float())
        return g
    
    def edge_index_to_coo(edge_index:torch.Tensor, N:int, device='cuda:0'):
        v = torch.ones(edge_index.size(1)).to(device)
        edge_index = edge_index.to(device)
        s = torch.sparse_coo_tensor(edge_index, v, (N, N))
        return s
    
    def dense_to_coo(a:torch.Tensor, device='cuda:0'):
        idx = torch.nonzero(a).to(device).T
        data = a[idx[0],idx[1]]
        coo = torch.sparse_coo_tensor(idx, data, a.shape)
        return coo
    
    def from_pyg_graph(graph:pyg_Data, sparse=False):
        N = graph.num_nodes
        
        g = BaseGraph(graph_type='pyg', adj_type='coo' if sparse else 'both', pyg_graph=graph)
        
        if sparse:
            s = BaseGraphUtils.edge_index_to_coo(graph.edge_index, N)
            g._set_coo(s)
            g._set_edge_index(graph.edge_index.long())
        else:
            A = s.to_dense()
            g.A = A
            
        g._set_node_num(N)
        g.set_node_feat(graph.x)
        g.set_edge_feat(graph.edge_attr)
        if hasattr(graph, 'y'):
            g.set_label(graph.y)
        return g
    
    
    def init_batch_graph(graphs : Sequence[BaseGraph], sparse=False):
        assert len(graphs) > 0
            
        g1 = graphs[0]
        
        if g1.graph_type == 'pyg':
            pyg_b_graph = pyg_Batch.from_data_list([g.pyg_graph for g in graphs])
            batched_g = BaseGraphUtils.from_pyg_graph(pyg_b_graph, sparse)
            batched_g.set_node_feat(pyg_b_graph.x)
            
            batched_g.batch_num = len(graphs)
            batched_g.adj_type = 'coo'
            if g1.is_cuda:
                batched_g.cuda()
                
            return batched_g
        else:
            batched_g = BaseGraph(batch=True, batch_num=len(graphs))
            batched_g._set_graph_type(g1.graph_type)
            batched_g.adj_type=g1.adj_type
            
            if batched_g.adj_type in ['dense','both']:
                
                batch_A = []
                batch_node_feas = []
                batch_edge_feas = []
                batch_pos_encs = []
                batch_labels = []
                
                for g in graphs:
                    batch_A.append(g.A)
                    batch_node_feas.append(g.get_node_features())
                    
                    if g.get_edge_features() is not None:
                        batch_edge_feas.append(g.get_edge_features())
                        
                    if 'pos_enc' in g.ndata and g.ndata['pos_enc'] is not None:
                        batch_pos_encs.append(g.ndata['pos_enc'])
                        
                    if g.label is not None:
                        batch_labels.append(g.label)
                    
                batch_A = torch.stack(batch_A, dim=0)
                batched_g._set_A(batch_A)
                
                batch_node_feas = torch.stack(batch_node_feas, dim=0)
                batched_g.set_node_feat(batch_node_feas)
                
                if len(batch_pos_encs) > 0:
                    batch_pos_encs = torch.stack(batch_pos_encs, dim=0)
                    batched_g.ndata['pos_enc'] = batch_pos_encs
                    
                if len(batch_edge_feas) > 0:
                    batch_edge_feas = torch.stack(batch_edge_feas, dim=0)
                    batched_g.set_edge_feat(batch_edge_feas)
                    
                if len(batch_labels) > 0:
                    batch_labels = torch.stack(batch_labels, dim=0)
                    batched_g.set_label(batch_labels)
                
            elif batched_g.adj_type == 'coo':
                pyg_graphs = [pyg_Data(x=g.get_node_features(), edge_index=g.coo.coalesce().indices()) for g in graphs]
                pyg_b_graph = pyg_Batch.from_data_list(pyg_graphs)
                
                batched_g = BaseGraphUtils.from_pyg_graph(pyg_b_graph, sparse=True)
                batched_g.batch_num = len(graphs)
                
                return batched_g
            else:
                raise NotImplementedError
            
            return batched_g
    
    def lap_positional_encoding(g: BaseGraph, pos_enc_dim):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """

        # Laplacian
        normed_L = utils.calculate_normalized_laplacian(g.A.numpy())

        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(normed_L)
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() # take the first k eigen vectors.
        g.ndata['eigvec'] = g.ndata['pos_enc'] # shape: NxK

        return g



class GraphDataset(Dataset):
    def __init__(self, x, y):
        """ input: x is list of BaseGraph, y is list of LongTensor of pytorch.
        """
        assert len(x) == y.shape[0]
        self.x = x
        self.y = y
        
    # def _transform_dgl(self):
    #     self.graph_list = []
    #     self.graph_label = []
        
    #     for adj, node_fea in self.x:
    #         sp_A = sparse.csc_matrix(adj)
    #         g = dgl.from_scipy(sp_A)
    #         g.ndata['node_feat'] = torch.from_numpy(node_fea) if isinstance(node_fea, np.ndarray) else node_fea
    #         self.graph_list.append(g)
    #     self.graph_label = self.y
    
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.stack(labels, dim=0)
        # NOTE: batched_graph is a tensor
        batched_graph = BaseGraphUtils.init_batch_graph(graphs)
        
        # batched_graphs = [g.A for g in graphs]
        # batched_graphs = torch.stack(batched_graphs, dim=0)
        
        return batched_graph, labels
        
    def cuda(self):
        for i in range(len(self.x)):
            self.x[i].cuda()
        
        self.y = self.y.cuda()
        


    def _add_lap_positional_encodings(self, pos_enc_dim):
            # Graph positional encoding v/ Laplacian eigenvectors
        [BaseGraphUtils.lap_positional_encoding(g, pos_enc_dim) for g in self.x]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Valid: {result[:, 0].max():.2f}')
            print(f'   Final Test: {result[argmax, 1]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))
            best_result = torch.tensor(best_results)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.4f} +- {r.std():.4f}')
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.4f} +- {r.std():.4f}')


class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu(x_j + edge_attr)


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)





class LSDGINConv(MessagePassing):
    def __init__(self, pre_nn: Callable, eps: float = 0., train_eps: bool = False, device='cuda:0',
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.pre_nn = pre_nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.eps = torch.Tensor([eps])
        self.eps = self.eps.to(device)
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_index_opt: Adj=None,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        if edge_index.dim() > 2:
            # same node batch input.
            out = torch.matmul(edge_index_opt, x[0])
            if edge_index_opt is not None:
                out_opt = torch.matmul(edge_index_opt, x[1])
                out = torch.cat([out, out_opt], dim=-1)
            
        else:
            out = self.propagate(edge_index, x=x, size=size)
            
            if edge_index_opt is not None:
                out_opt = self.propagate(edge_index_opt, x=x, size=size)
                out = torch.cat([out, out_opt], dim=-1)
                
        if edge_index_opt is None:
            out += (1 + self.eps) * x[0]
        else:
            out += (1 + self.eps) * torch.cat([x[0], x[1]], dim=-1)
        out = self.pre_nn(out)
        
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return torch_sparse.matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'



class MyDiGINConv(nn.Module):
    def __init__(self, ln):
        super(MyDiGINConv, self).__init__()
        self.ln = ln
        self.eps = torch.nn.Parameter(torch.Tensor([0.001]))
        
    def forward(self, x, adj1, adj2):
        # x and adj are batch-wise.
        x_r = x
        out1 = torch.matmul(adj1, x)
        out2 = torch.matmul(adj2, x)
        
        out = torch.cat([out1, out2], dim=-1)
        
        out += (1 + self.eps) * torch.cat([x_r, x_r], dim=-1)
        
        out = self.ln(out)
        
        return out
        
        
        
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        # propagate: ['message', 'aggregate', 'message_and_aggregate', 'update']

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j  # (Nx1) * (NxC)
    
    
class MyGINConv(nn.Module):
    def __init__(self, ln):
        super(MyGINConv, self).__init__()
        self.ln = ln
        self.eps = torch.nn.Parameter(torch.Tensor([0.001]))
        
    def forward(self, x, adj, graphs=None):
        # x and adj are batch-wise.
        print('x shape', x.shape)
        print('adj shape', adj.shape)
        x_r = x
        out = torch.matmul(adj, x)
        
        out += (1 + self.eps) * x_r
        
        out = self.ln(out)
        return out
        
      

class LSDGINNet(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim, layer_num, dropout=0.6, last_linear=True,
                 device='cuda:0',  bi=True):
        super(LSDGINNet, self).__init__()
        
        self.args = args
        self.device = device
        self.pe_init = args.pe_init
        self.bi = bi
        if self.pe_init == 'lap_pe':
            self.embedding_p = nn.Linear(args.pos_en_dim, hid_dim)
            in_dim += hid_dim    # cat the pos_fea and node_fea
            
        self.convs = nn.ModuleList()
        
        if self.bi:
            in_dim *= 2
        ln_init = MLP(in_dim, hid_dim, out_dims=hid_dim, layer_nums=2)
            
        self.convs.append(LSDGINConv(ln_init, device=device))
        
        for _ in range(layer_num - 2):
            ln_mid = MLP(hid_dim, hid_dim, hid_dim, layer_nums=2)
            self.convs.append(LSDGINConv(ln_mid, device=device))
            
        self.out_dim = out_dim
        if self.bi:
            ln_last = nn.Linear(2*hid_dim, out_dim)
        else:
            ln_last = nn.Linear(hid_dim, out_dim)
            
        if not last_linear:
            ln_last = nn.Sequential(ln_last, nn.ReLU)
        self.convs.append(LSDGINConv(ln_last, device=device))


    def forward(self, x, edge_index, edge_index_opt=None, graphs:BaseGraph=None):

        # NOTE: pre process edge_index:


        if self.pe_init == 'lap_pe':
            # not batch?
            batch_pos_enc = graphs.ndata['pos_enc']
            # print('input x shape ', x.shape, ' pos_en shape:', batch_pos_enc.shape)
            sign_flip = torch.rand(batch_pos_enc.shape).to(self.device)
            sign_flip[sign_flip>=0.5] = 1.0
            sign_flip[sign_flip<0.5] = -1.0
            
            batch_pos_enc = batch_pos_enc * sign_flip
     
            p = batch_pos_enc
            p = self.embedding_p(p)
            x = torch.cat([x, p], dim=-1)
            p = None

        for conv in self.convs:
            x = conv(x, edge_index, edge_index_opt)
        return x
    
    

class DiGINNet(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim, layer_num, dropout=0.6, last_linear=True):
        super(DiGINNet, self).__init__()
        
        self.args = args
        
        self.pos_en = args.pos_en
        
        if self.pos_en == 'lap_pe':
            self.embedding_p = nn.Linear(args.pos_en_dim, hid_dim)
            in_dim += hid_dim    # cat the pos_fea and node_fea
            
            
        self.convs = nn.ModuleList()
        
        ln_init = MLP(2*in_dim, hid_dim, hid_dim, 2)
        
        self.convs.append(MyDiGINConv(ln_init))
        
        for _ in range(layer_num - 2):
            ln_mid = MLP(2 * hid_dim, hid_dim, hid_dim, 2)
            self.convs.append(MyDiGINConv(ln_mid))
            
        self.out_dim = out_dim
        ln_last = nn.Linear(2*hid_dim, out_dim)
        
        if not last_linear:
            ln_last = nn.Sequential(ln_last, nn.ReLU)
        self.convs.append(MyDiGINConv(ln_last))
        
    def forward(self, x, adj1, adj2, graphs:BaseGraph=None):
        
        if self.pos_en == 'lap_pe':
            # not batch?
            batch_pos_enc = graphs.ndata['pos_enc']
            # print('input x shape ', x.shape, ' pos_en shape:', batch_pos_enc.shape)
            sign_flip = torch.rand(batch_pos_enc.shape).cuda()
            sign_flip[sign_flip>=0.5] = 1.0
            sign_flip[sign_flip<0.5] = -1.0
            
            batch_pos_enc = batch_pos_enc * sign_flip
     
            p = batch_pos_enc
            p = self.embedding_p(p)
            # try cat:
            
            x = torch.cat([x, p], dim=-1)
            p = None
            
        for conv in self.convs:
            x = conv(x, adj1, adj2)
            
        return x
    

class GCNO(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, class_num):
        super(GCNO, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.conv3 = GCNConv(hid_dim, hid_dim)
        self.lin = Linear(hid_dim, class_num)

    def forward(self, graphs:BaseGraph):
        x = graphs.get_node_features()
            
        if graphs.graph_type == 'pyg':
            edge_index = graphs.pyg_graph.edge_index
            
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, graphs.pyg_graph.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


class GINNet(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim, layer_num, dropout, last_linear=True):
        super(GINNet, self).__init__()
        
        self.args = args
        
        self.pos_en = args.pos_en
        
        if self.pos_en == 'lap_pe':
            self.embedding_p = nn.Linear(args.pos_en_dim, hid_dim)
            in_dim += hid_dim    # cat the pos_fea and node_fea
        
        self.convs = nn.ModuleList()

        ln_init = MLP(in_dim, hid_dim, hid_dim, 2)
        
        self.convs.append(MyGINConv(ln_init))
        
        for _ in range(layer_num - 2):
            ln_mid = MLP(hid_dim, hid_dim, hid_dim, 2)
            self.convs.append(MyGINConv(ln_mid))
        self.out_dim = out_dim
        ln_last = nn.Linear(hid_dim, out_dim)
        if not last_linear:
            ln_last = nn.Sequential(ln_last, nn.ReLU)
        self.convs.append(MyGINConv(ln_last))
        
    def forward(self, x, adj, graphs=None):
        # 
        
        if self.pos_en == 'lap_pe':
            # not batch?
            batch_pos_enc = graphs.ndata['pos_enc']
            # print('input x shape ', x.shape, ' pos_en shape:', batch_pos_enc.shape)
            sign_flip = torch.rand(batch_pos_enc.shape).cuda()
            sign_flip[sign_flip>=0.5] = 1.0
            sign_flip[sign_flip<0.5] = -1.0
            
            batch_pos_enc = batch_pos_enc * sign_flip
     
            p = batch_pos_enc
            p = self.embedding_p(p)
            # try cat:
            
            x = torch.cat([x, p], dim=-1)
            p = None
        
        for conv in self.convs:
            x = conv(x, adj)
            
        return x
        

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GraphSAGE,self).__init__()
        self.convs = torch.nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_attr, emb_ea):
        edge_attr = torch.mm(edge_attr, emb_ea)
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, edge_attr)  # no nonlinearity
        return x


def gumbel_sampling(shape, mu=0, beta=1):
    y = torch.rand(shape).cuda() + 1e-20  # ensure all y is positive.
    g = mu - beta * torch.log(-torch.log(y)+1e-20)
    return g


class LinkPredictorMy(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, num_gumbels=1, emperical=False):
        super(LinkPredictorMy, self).__init__()

        self.W = nn.Parameter(torch.randn(dim_in, dim_in))
        self.tau = 0.5
        self.dim_hidden = dim_hidden
        self.linkpred = LinkPredictor(dim_in, dim_in, 1, 2, 0.3)

        self.warm_up = -1
        self.count = 0

        self.num_gumbels = num_gumbels
        self.emperical = emperical


    def expectation_sampling(self, pi):
        U = torch.log(pi)
        gumbs = gumbel_sampling(pi.shape)
        for n in range(self.num_gumbels-1):
            gumbs += gumbel_sampling(pi.shape)
        U += gumbs/self.num_gumbels
        U = U/self.tau
        return U

    def forward(self, zi, zj):
        """
        zi, zj \in (batch_size, dim_in)
        """
        def hasNan(x):
            return torch.isnan(x).any()

        if self.count < self.warm_up:
            self.count += 1
            return self.linkpred(zi, zj)

        P = torch.sigmoid(torch.einsum('nc, nc -> n', zi@self.W, zj))
        # P = F.dropout(P, p=0.2)
        pi = torch.stack((1-P, P),dim=1)
        if self.emperical:
            U = self.expectation_sampling(pi)
        else:
            U = torch.log(pi)
            U = U/self.tau
        p_exp = torch.exp(U)
        p_sum = torch.sum(p_exp, dim=1)

        p1 = p_exp[:,1]/p_sum
        # edge = F.softmax(U, dim=1)
        if hasNan(p1):
            print('hasNan:p1p1p1p1p:', p1)
        return p1

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.linkpred.reset_parameters()


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)




class GCN(MessagePassing):
    def __init__(self, in_channel, out_channel, aggr="add", flow: str = "source_to_target", node_dim: int = -2):
        super().__init__(aggr=aggr, flow=flow, node_dim=node_dim)
        self.lin = nn.Linear(in_channel, out_channel)
    
    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)
        if isinstance(x, Tensor):
            xx: OptPairTensor = (x, x)
        
        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=xx, edge_attr=edge_attr, norm=norm)
        out = out + xx[1]
        # Step 2: Linearly transform node feature matrix.
        # out = F.relu(out)
        
        return out

    def message(self, x_j, norm, edge_attr=None):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        if edge_attr is not None:
            return norm.view(-1, 1) * (x_j + edge_attr)
        return norm.view(-1, 1) * x_j
        
class MultiGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers):
        super(MultiGCN,self).__init__()
        self.convs = nn.ModuleList()
        self.out_dim = out_dim
        self.convs.append(GCN(in_dim, hid_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCN(hid_dim, hid_dim))
        self.convs.append(GCN(hid_dim, out_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_index_opt=None, graphs:BaseGraph=None,
                edge_attr=None, emb_ea=None) -> Tensor:
        
        if emb_ea is not None and edge_attr is not None:
            edge_attr = torch.mm(edge_attr, emb_ea)
            
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)  # no nonlinearity
        return x     


class QNN(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(QNN,self).__init__()

        # self.mean = GCN(dim_in, dim_hidden)
        self.std = GCN(dim_in, dim_hidden)

    def forward(self, x, edge_index, edge_attr):
        return self.mean(x,edge_index, edge_attr), self.std(x,edge_index, edge_attr)


    
class GenerativeGNN(nn.Module):
    def __init__(self, A_0, tau, dim_in, dim_hidden, dim_out, num_layers):
        super(GenerativeGNN,self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCN(dim_in, dim_hidden))
        for _ in range(num_layers - 2):
            self.convs.append(GCN(dim_hidden, dim_hidden))
        self.convs.append(GCN(dim_hidden, dim_out))

        self.W = nn.Parameter(torch.rand(dim_in, dim_in))
        self.tau = tau
        self.A_0 = A_0
        self.N = A_0.size(0) # num of nodes.
        self.dim_hidden = dim_hidden


    def update_A(self, z, edge_index, edge_attr):
        """
        update A via node representation Z (N x N)
        """
        P = F.sigmoid(torch.einsum('nc, nc -> n', z@self.W, z))
        pi = torch.tensor(torch.stack((1-P, P),dim=2))
        print('pi shape: ', pi.shape)
        U = torch.log(pi) + gumbel_sampling(pi.shape)
        U = U/self.tau
        A = F.softmax(U, dim=2)
        self.A_opt = A


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_attr, emb_ea):
        edge_attr = torch.mm(edge_attr, emb_ea)
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, adj_t, edge_attr)  # no nonlinearity
        return x


class VGAE(nn.Module):
	def __init__(self, adj, args):
		super(VGAE, self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = torch.randn(X.size(0), self.args.hidden2_dim)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred


class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)


class GAE(nn.Module):
	def __init__(self,adj,args):
		super(GAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		z = self.mean = self.gcn_mean(hidden)
		return z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

def update_A(z, W=None):
    """
    update A via node representation Z (N x N)
    """
    P = F.sigmoid(torch.einsum('nc, nc -> n', z@W, z))
    print('P shape:', P.shape)
    pi = torch.tensor(torch.stack((1-P, P),dim=1))
    print('pi shape: ', pi.shape)
    U = torch.log(pi) + gumbel_sampling(pi.shape)
    # plot:


    A = F.softmax(U, dim=1)
    print('A : ', A[0, :])


import os

import torch
import torch.nn.functional as F
from torch import  nn
from torch_scatter import scatter_mean, scatter, scatter_add, scatter_max
from torch_geometric.nn.conv import MessagePassing

import my_utils as utils
from my_utils import DLog

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



class MLP(nn.Module):
    def __init__(self, in_dims, hid_dims, out_dims, layer_nums,
                 dropout=0.6, act=nn.ReLU):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(in_dims, hid_dims))
        self.lins.append(act())
        self.lins.append(nn.Dropout(dropout))
        
        for _ in range(layer_nums - 2):
            self.lins.append(nn.Linear(hid_dims, hid_dims))
            self.lins.append(act())
            self.lins.append(nn.Dropout(dropout))
        self.lins.append(nn.Linear(hid_dims, out_dims))
        
    def forward(self, x):
        for lin in self.lins[:-1]:
            DLog.debug('MLP input shape', x.shape)
            x = lin(x)
            DLog.debug('MLP lin 1 out shape', x.shape)
        x = self.lins[-1](x)
        return x



        
class ClassPredictor(nn.Module):
    def __init__(self, in_dims, hid_dims, class_num, layer_nums,
                 dropout=0.5):
        super(ClassPredictor, self).__init__()
        self.mlp = MLP(in_dims, hid_dims, class_num, layer_nums, dropout=dropout)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            adj = x[1]
            _ = x[0]
        x = torch.flatten(adj, start_dim=1)
        DLog.debug('input prediction x shape:', x.shape)
        x = self.mlp(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, in_dims, hid_dims, out_dims, dropout=0.6, kernelsize=(3, 3)):
        super(SimpleCNN, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_dims, hid_dims, kernel_size=(1,1), stride=2),
            nn.ReLU(),
            nn.Conv2d(hid_dims, hid_dims, kernel_size=kernelsize, stride=2),
            nn.ReLU(),
            nn.Conv2d(hid_dims, hid_dims, kernel_size=(1,1), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(hid_dims),
            nn.Dropout(dropout))
                
        height = width = utils.cal_cnn_outlen(self.block, 40)
        self.ln = nn.Linear(height * width * hid_dims, out_dims)
        
    def forward(self, x):
        """x is BaseGraph type, batched BaseGraph
        """
        x = x.A
        x = x.unsqueeze(dim=1)
            
        x = self.block(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.ln(x)
        
        return x
        
        

class WalkPooling(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 1,\
                 walk_len: int = 6, cuda=True):
        super(WalkPooling, self).__init__()

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.walk_len = walk_len
        self.device = torch.device("cuda:0" if cuda else "cpu")  
        # the linear layers in the attention encoder
        self.lin_key1 = nn.Linear(in_channels, hidden_channels)
        self.lin_query1 = nn.Linear(in_channels, hidden_channels)
        self.lin_key2 = nn.Linear(hidden_channels, heads * hidden_channels)
        self.lin_query2 = nn.Linear(hidden_channels, heads * hidden_channels)
    def attention_mlp(self, x, edge_index):
    
        query = self.lin_key1(x).reshape(-1,self.hidden_channels)
        key = self.lin_query1(x).reshape(-1,self.hidden_channels)

        query = F.leaky_relu(query,0.2)
        key = F.leaky_relu(key,0.2)

        query = F.dropout(query, p=0.5, training=self.training)
        key = F.dropout(key, p=0.5, training=self.training)

        query = self.lin_key2(query).view(-1, self.heads, self.hidden_channels)
        key = self.lin_query2(key).view(-1, self.heads, self.hidden_channels)

        row, col = edge_index
        weights = (query[row] * key[col]).sum(dim=-1) / np.sqrt(self.hidden_channels)
        
        return weights

    def weight_encoder(self, x, edge_index, edge_mask):        
     
        weights = self.attention_mlp(x, edge_index)
    
        omega = torch.sigmoid(weights[torch.logical_not(edge_mask)])
        
        row, col = edge_index
        num_nodes = torch.max(edge_index)+1

        # edge weights of the plus graph
        weights_p = F.softmax(weights,edge_index[1])

        # edge weights of the minus graph
        weights_m = weights - scatter_max(weights, col, dim=0, dim_size=num_nodes)[0][col]
        weights_m = torch.exp(weights_m)
        weights_m = weights_m * edge_mask.view(-1,1)
        norm = scatter_add(weights_m, col, dim=0, dim_size=num_nodes)[col] + 1e-16
        weights_m = weights_m / norm

        return weights_p, weights_m, omega

    def forward(self, x, edge_index, edge_mask, batch):
        device = self.device
        #encode the node representation into edge weights via attention mechanism
        weights_p, weights_m, omega = self.weight_encoder(x, edge_index, edge_mask)

        # number of graphs in the batch
        batch_size = torch.max(batch)+1

        # for node i in the batched graph, index[i] is i's id in the graph before batch 
        index = torch.zeros(batch.size(0),1,dtype=torch.long)
        
        # numer of nodes in each graph
        _, counts = torch.unique(batch, sorted=True, return_counts=True)
        
        # maximum number of nodes for all graphs in the batch
        max_nodes = torch.max(counts)

        # set the values in index
        id_start = 0
        for i in range(batch_size):
            index[id_start:id_start+counts[i]] = torch.arange(0,counts[i],dtype=torch.long).view(-1,1)
            id_start = id_start+counts[i]

        index = index.to(device)
        
        #the output graph features of walk pooling
        nodelevel_p = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        nodelevel_m = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        linklevel_p = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        linklevel_m = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        graphlevel = torch.zeros(batch_size,(self.walk_len*self.heads)).to(device)
        # a link (i,j) has two directions i->j and j->i, and
        # when extract the features of the link, we usually average over
        # the two directions. indices_odd and indices_even records the
        # indices for a link in two directions
        indices_odd = torch.arange(0,omega.size(0),2).to(device)
        indices_even = torch.arange(1,omega.size(0),2).to(device)

        omega = torch.index_select(omega, 0 ,indices_even)\
        + torch.index_select(omega,0,indices_odd)
        
        #node id of the candidate (or perturbation) link
        link_ij, link_ji = edge_index[:,torch.logical_not(edge_mask)]
        node_i = link_ij[indices_odd]
        node_j = link_ij[indices_even]

        # compute the powers of stochastic matrix
        for head in range(self.heads):

            # x on the plus graph and minus graph
            x_p = torch.zeros(batch.size(0),max_nodes,dtype=x.dtype).to(device)
            x_p = x_p.scatter_(1,index,1)
            x_m = torch.zeros(batch.size(0),max_nodes,dtype=x.dtype).to(device)
            x_m = x_m.scatter_(1,index,1)

            # propagage once
            x_p = self.propagate(edge_index, x= x_p, norm = weights_p[:,head])
            x_m = self.propagate(edge_index, x= x_m, norm = weights_m[:,head])
        
            # start from tau = 2
            for i in range(self.walk_len):
                x_p = self.propagate(edge_index, x= x_p, norm = weights_p[:,head])
                x_m = self.propagate(edge_index, x= x_m, norm = weights_m[:,head])
                
                # returning probabilities around i + j
                nodelevel_p_w = x_p[node_i,index[node_i].view(-1)] + x_p[node_j,index[node_j].view(-1)]
                nodelevel_m_w = x_m[node_i,index[node_i].view(-1)] + x_m[node_j,index[node_j].view(-1)]
                nodelevel_p[:,head*self.walk_len+i] = nodelevel_p_w.view(-1)
                nodelevel_m[:,head*self.walk_len+i] = nodelevel_m_w.view(-1)
  
                # transition probabilities between i and j
                linklevel_p_w = x_p[node_i,index[node_j].view(-1)] + x_p[node_j,index[node_i].view(-1)]
                linklevel_m_w = x_m[node_i,index[node_j].view(-1)] + x_m[node_j,index[node_i].view(-1)]
                linklevel_p[:,head*self.walk_len+i] = linklevel_p_w.view(-1)
                linklevel_m[:,head*self.walk_len+i] = linklevel_m_w.view(-1)

                # graph average of returning probabilities
                diag_ele_p = torch.gather(x_p,1,index)
                diag_ele_m = torch.gather(x_m,1,index)

                graphlevel_p = scatter_add(diag_ele_p, batch, dim = 0)
                graphlevel_m = scatter_add(diag_ele_m, batch, dim = 0)

                graphlevel[:,head*self.walk_len+i] = (graphlevel_p-graphlevel_m).view(-1)
         
        feature_list = graphlevel 
        feature_list = torch.cat((feature_list,omega),dim=1)
        feature_list = torch.cat((feature_list,nodelevel_p),dim=1)
        feature_list = torch.cat((feature_list,nodelevel_m),dim=1)
        feature_list = torch.cat((feature_list,linklevel_p),dim=1)
        feature_list = torch.cat((feature_list,linklevel_m),dim=1)


        return feature_list

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j  
    
class GCNM(torch.nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(GCNM, self).__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.conv3 = GCNConv(hid_dim, hid_dim)
        self.out_dim = hid_dim

    def forward(self, x, edge_index, graphs:BaseGraph=None):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        return x
    
    
class GraphConv(MessagePassing):
    def __init__(self, in_dim, out_dim, dropout, act=nn.ReLU, norm=False):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.ln = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = norm
        self.act = act()
        
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_index_opt=None, edge_attr: OptTensor=None, 
                graphs:BaseGraph=None, size=None):
        # x shape is B*NC
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.ln(out)
        
        if self.norm:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_attr is not None:
            return self.act(x_j + edge_attr)
        else:
            return self.act(x_j)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_dim,
                                   self.out_dim)

class MultilayerGNN(nn.Module):
    def __init__(self, layer_num, in_dim, hid_dim, out_dim, dropout=0.5):
        super(MultilayerGNN, self).__init__()
        self.gnns = nn.ModuleList()

        self.gnns.append(GraphConv(in_dim, hid_dim, dropout))
        for _ in range(layer_num-2):
            self.gnns.append(GraphConv(hid_dim, hid_dim, dropout))
        self.gnns.append(GraphConv(hid_dim, out_dim, dropout))
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_index_opt=None, graphs:BaseGraph=None):
        """
        input x shape: B*NC
        """
        for gnn in self.gnns:
            x = gnn(x, edge_index)
        return x


class CompactPooling(nn.Module):
    def __init__(self, args, K, N):
        super(CompactPooling, self).__init__()
        self.CompM = nn.Parameter(torch.Tensor(K, N).cuda())
        nn.init.normal_(self.CompM, mean=0.01, std=0.01)

    def forward(self, x):
        DLog.debug('CompactPooling in shape:', x.shape)
        x = torch.matmul(self.CompM, x)
        DLog.debug('matmul CompactPooling shape:', x.shape)
        x = torch.sum(x, dim=-2).squeeze()
        DLog.debug('out CompactPooling shape:', x.shape)
        return x


class LinearGraphPooling(nn.Module):
    def __init__(self, in_dim, hid_dim=32):
        super(LinearGraphPooling, self).__init__()
        self.ln = nn.Linear(in_dim, hid_dim)
        
    def forward(self, x:torch.Tensor):
        # input x shape: BNC, or BTNC
        # out shape: BK
        x = self.ln(x) # BNC -> BNK -> BK
        x = torch.sum(x, dim=-2)
        return x
    

    
class MeanPooling(nn.Module):
    def __init__(self, args=None):
        super(MeanPooling, self).__init__()
        self.args = args
        
    def forward(self, x):
        """ignore the following dimensions after the 3rd one.
        Args:
            x (tensor): shape: B, ..., N,C
        Returns:
            x shape: B,C
        """
        x = torch.mean(x, dim=-2)
        return x
    
    
    
class GateGraphPooling(nn.Module):
    def __init__(self, args=None, N=20):
        super(GateGraphPooling, self).__init__()
        self.args = args
        self.N = N
        self.gate =nn.Parameter(torch.FloatTensor(self.N))
        # Cx\\
        self.reset_parameters()
        
        
    def forward(self, x):
        """ignore the following dimensions after the 3rd one.
        Args:
            x (tensor): shape: B,N,C,...
        Returns:
            x shape: B,C,...
        """
        shape = x.shape
        if len(shape) > 3:
            x = torch.einsum('btnc, n -> btc', x, self.gate)
        else:
            x = torch.einsum('bnc, n -> bc', x, self.gate)
        return x
    
    def reset_parameters(self):
        nn.init.normal_(self.gate, mean=0.01, std=0.01)
    

class CatPooling(nn.Module):
    def __init__(self):
        super(CatPooling, self).__init__()
        pass

    def forward(self, x):
        # input B*NC -> B*C
        return torch.flatten(x, start_dim=-2)
    

class AttGraphPooling(nn.Module):
    def __init__(self, args, N, in_dim, hid_dim):
        super(AttGraphPooling, self).__init__()
        self.args = args
        self.N = N
        self.Q = nn.Linear(in_dim, hid_dim)
        self.K = nn.Linear(in_dim, hid_dim)
        self.V = nn.Linear(in_dim, hid_dim)
        if args.agg_type == 'gate':
            self.gate =nn.Parameter(torch.FloatTensor(self.N))
        self.reset_parameters()
        
    def forward(self, x):
        """ignore the following dimensions after the 3rd one.
        Args:
            x (tensor): shape: B,N,C
        Returns:
            x shape: B,C
        """
        x = x.transpose(2, -1)  # make last dimension is channel.
        Q = self.Q(x) # BNC BNC.
        K = self.K(x)
        V = self.V(x)
        
        att = torch.bmm(Q, K.transpose(1,2))/self.N**0.5
        att = torch.softmax(att, dim=1)
        DLog.debug('att shape', att.shape)
        x = torch.bmm(att, V)  # bnc.
        if self.args.agg_type == 'gate':
            x = torch.einsum('bnc, n -> bc', x, self.gate)
        elif self.args.agg_type == 'cat':
            x = torch.flatten(x, start_dim=1)
        elif self.args.agg_type == 'sum':
            x = torch.sum(x, dim=1)
        
        DLog.debug('after att:', x.shape)
        return x
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.Q.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.K.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.V.weight, mode='fan_in')
        if self.args.agg_type == 'gate':
            nn.init.normal_(self.gate, mean=0.01, std=0.01)
        
        
class LatentGraphGenerator(nn.Module):
    def __init__(self, args, A_0, tau, in_dim, hid_dim, K=10):
        super(LatentGraphGenerator,self).__init__()
        self.N = A_0.shape[0] # num of nodes.
        self.tau = tau
        self.args = args
        self.A_0 = A_0
        self.args = args

        if args.gnn_pooling == 'att':
            pooling = AttGraphPooling(args, self.N, in_dim, 64)
        elif args.gnn_pooling == 'cpool':
            pooling = CompactPooling(args, 3, self.N)
        elif args.gnn_pooling.upper() == 'NONE':
            pooling = None
        else:
            pooling = GateGraphPooling(args, self.N)
            
        self.gumbel_tau = 0.1
        self.mu_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        self.sig_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        self.pi_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        
        self.adj_fix = nn.Parameter(self.A_0)

        print('adj_fix', self.adj_fix.shape)

        self.init_norm()


    def init_norm(self):
        self.Norm = torch.randn(size=(1000, self.args.batch_size, self.N)).cuda()
        self.norm_index = 0

    def get_norm_noise(self, size):
        if self.norm_index >= 999:
            self.init_norm()

        if size == self.args.batch_size:
            self.norm_index += 1
            return self.Norm[self.norm_index].squeeze()
        else:
            return torch.randn((size, self.N)).cuda()
        
    def update_A(self, mu, sig, pi):
        """ mu, sig, pi, shape: (B, N, K)
        update A, 
        """
        # cal prob of pi:
        DLog.debug('pi Has Nan:', torch.isnan(pi).any())
        logits = torch.log(torch.softmax(pi, dim=-1))
        DLog.debug('logits Has Nan:', torch.isnan(logits).any())

        pi_onehot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)

        # select one component of mu, sig via pi for each node:

        mu_k = torch.sum(mu * pi_onehot, dim=-1) # BN
        sig_k = torch.sum(sig * pi_onehot, dim=-1) #BN

        n = self.get_norm_noise(mu_k.shape[0]) # BN
        DLog.debug('mu shape:', mu_k.shape)
        DLog.debug('sig_k shape:', sig_k.shape)
        DLog.debug('n shape:', n.shape)

        S = mu_k + n*sig_k
        S = S.unsqueeze(dim=-1)
        # change to gumbel softmax, discrete sampling.
        # DLog.debug('S Has Nan:', torch.isnan(S).any())
        Sim = torch.einsum('bnc, bcm -> bnm', S, S.transpose(2, 1)) # need to be softmax

        P = torch.sigmoid(Sim)

        pp = torch.stack((P+0.01, 1-P + 0.01), dim=3)
        DLog.debug('min:', torch.min(pp))
        # DLog.debug('max',torch.max(pp))
        pp_logits = torch.log(pp)
        DLog.debug('Has Nan:', torch.isnan(pp_logits).any())
        pp_onehot = F.gumbel_softmax(pp_logits, tau=self.gumbel_tau, hard=False, dim=-1)
        A = pp_onehot[:,:,:,0]
        A = torch.mean(A, dim=0)

        return A

    def forward(self, x, adj_t=None):
        if adj_t is None:
            adj_t = self.adj_fix
            DLog.debug('LGG: adj_t shape', adj_t.shape)
        
        mu = self.mu_nn(x, adj_t)
        sig = self.sig_nn(x, adj_t)
        pi = self.pi_nn(x, adj_t)

        A = self.update_A(mu, sig, pi)

        return A
    
    
    
if __name__ == '__main__':
    
    z = torch.rand((1000, 2))
    
    plt.hist(z[:,0])
    
    w = torch.nn.Parameter(torch.ones((2, 2)))
    # nn.init.kaiming_uniform_(w, a=math.sqrt(5))


    print('z shape:', z.shape)
    print('w shape:', w.shape)
    update_A(z, w)








def global_mean_pool(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    if batch is None:
        return x.mean(dim=-2, keepdim=x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=-2, dim_size=size, reduce='mean')

