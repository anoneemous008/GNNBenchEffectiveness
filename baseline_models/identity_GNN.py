# my implementation based on paper: "Identity-aware Graph Neural Networks"

import numpy as np
import random

from typing import Callable, Union
from typing import List, Optional, Set
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
import torch
import torch.nn as nn
# from torch.functional import F
from torch_geometric.nn import GATConv, GINConv, ChebConv, SAGEConv, HypergraphConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric import utils as pygutils
from torch import Tensor
from torch_sparse import SparseTensor, matmul
# import torch_sparse
import models


class IDGINConv(MessagePassing):
    def __init__(self, pre_nn0: Callable, pre_nn1: Callable, eps: float = 0., 
                train_eps: bool = False, device='cuda:0',
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.pre_nn0 = pre_nn0
        self.pre_nn1 = pre_nn1
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.eps = torch.Tensor([eps])
        self.eps = self.eps.to(device)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_index_opt: Adj=None,
                size: Size = None, idx=None, k_nodes=None) -> Tensor:
        
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out, h0 = self.propagate(edge_index, x=x[0], size=size, idx=idx, k_nodes=k_nodes)
        
        if edge_index_opt is not None:
            out_opt = self.propagate(edge_index_opt, x=x, size=size)
            out = torch.cat([out, out_opt], dim=-1)
                
        if edge_index_opt is None:
            out += (1 + self.eps) * h0
        else:
            out += (1 + self.eps) * torch.cat([h0, h0], dim=-1)

        return out

    # def message(self, x_j: Tensor, idx=None) -> Tensor:
    #     print('calling mess', idx)
    #     return x_j

    
    def message_and_aggregate(self, edge_index: SparseTensor, x=None, idx=None, k_nodes=None ) -> Tensor:
        N = x.shape[0]
        h0 = self.pre_nn0(x)
        h1 = self.pre_nn1(x[idx, ...].unsqueeze(0))
        h_pre = torch.cat([h0, h1], dim=0)
        for j in k_nodes:
            # neighbors = torch.index_select(edge_index, dim=0, index=j) # TODO: pre_calculate
            neighbors = edge_index.index_select(dim=0, idx=j.unsqueeze(-1))
            neighbors = neighbors.to_torch_sparse_coo_tensor()._indices()[1]
            # NOTE: if n == i then xx[n + N] else: xx[n]
            neighbors = neighbors.squeeze()
            neighbors[neighbors==idx] = N
            
            h_pre[j] = torch.sum(h_pre[neighbors, :], dim=0) # NOTE: aggregate or other agg op.
            
        return h_pre[:N, ...], h0


class IDGNN(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim, layer_num, dropout=0.6):
        super(IDGNN, self).__init__()
        
        self.args = args
        self.out_dim = out_dim
        self.K = 2
        self.convs = nn.ModuleList()
        
        ln_0 = models.MLP(in_dim, hid_dim, hid_dim, 2)
        ln_1 = models.MLP(in_dim, hid_dim, hid_dim, 2)
        self.convs.append(IDGINConv(ln_0, ln_1))
        
        for _ in range(layer_num - 2):
            ln_mid0 = models.MLP(hid_dim, hid_dim, hid_dim, 2)
            ln_mid1 = models.MLP(hid_dim, hid_dim, hid_dim, 2)
            self.convs.append(IDGINConv(ln_mid0, ln_mid1))
            
        ln_last0 = models.MLP(hid_dim, hid_dim, out_dim, 2)
        ln_last1 = models.MLP(hid_dim, hid_dim, out_dim, 2)

        self.convs.append(IDGINConv(ln_last0, ln_last1))
        
    def forward(self, x, adj1, adj2=None, graphs:models.BaseGraph=None):
        sp_index = SparseTensor.from_edge_index(adj1)
        sp_index.set_value(None)
        
        k_adj = sp_index
        for _ in range(self.K-1):
            k_adj = matmul(k_adj, sp_index)
        dense_k_adj = k_adj.to_dense().long()
        out = []
        N = x.shape[0]
        for i in range(N):
            hj = x
            for conv in self.convs:
                hj = conv(hj, sp_index, adj2, idx=i, k_nodes=dense_k_adj[i])
            out.append(hj[i])
        out = torch.stack(out, dim=0)
        return out
    

