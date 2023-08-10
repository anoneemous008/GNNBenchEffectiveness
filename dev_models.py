from turtle import forward
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
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GINConv
from torch_geometric.data import Data as pyg_Data
from torch_geometric.data import Batch as pyg_Batch
from torch_geometric.utils import add_self_loops, degree

import math

from typing import Sequence

import matplotlib.pyplot as plt
import models

from torch_sparse import SparseTensor, matmul

class KGIN(nn.Module):
    def __init__(self, args, K, in_dim, hid_dim, out_dim, layer_num, dropout=0.6, last_linear=True,
                device='cuda:0',  bi=True):
        super(KGIN, self).__init__()
        self.args = args
        self.K = K
        
    def forward(self, x, adj1, adj2=None, graphs:models.BaseGraph=None):
        # NOTE: select k neighbors:
        sp_index = SparseTensor.from_edge_index(adj1)
        sp_index.set_value(None)
        # NOTE: if 2th hop neighbors exist in 1st hop neighbors, then circles exist.
        # check from the A^2 ?.
        
        k_adj = sp_index
        for _ in range(self.K-1):
            k_adj = matmul(k_adj, sp_index)
        
        x.index_add_(0, id, x_id)
        

class GINConv(MessagePassing):
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

