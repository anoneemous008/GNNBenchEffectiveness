import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import degree


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, edge_attr_dim, emb_dim):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        if edge_attr_dim is not None:
            self.edge_encoder = torch.nn.Linear(edge_attr_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        
        x = self.linear(x)
        
        if edge_attr is not None:
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr.view(-1,1)
            edge_embedding = self.edge_encoder(edge_attr.float())
        else:
            edge_embedding = 0

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, fea_dim, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False,
                 edge_attr_dim=None, use_edge_attr = False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.use_edge_attr = use_edge_attr
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.node_encoder = nn.Sequential(nn.Linear(fea_dim, emb_dim),
                                          nn.ReLU())
        for layer in range(num_layer):
            self.convs.append(GCNConv(edge_attr_dim, emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, data=None, x=None, edge_index=None, edge_attr=None, batch=None):
        if data is not None:
            x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
            
        edge_attr = edge_attr if self.use_edge_attr else None
        
        ### computing input node embedding
        h_list = [self.node_encoder(x.float())]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

class GCN(torch.nn.Module):

    def __init__(self, dim_features, edge_attr_dim, dim_target, config,
                    num_layer = 3, emb_dim = 300,
                    residual = False,
                    JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(GCN, self).__init__()

        self.config = config
        self.drop_ratio = config['dropout']
        self.embeddings_dim = [
            config['hidden_units'][0]] + config['hidden_units']
        self.no_layers = len(self.embeddings_dim)
        self.edge_attr_dim = edge_attr_dim
        self.num_layer = num_layer
        
        if 'num_layer' in config:
            num_layer = config['num_layer']
            
        if 'use_edge_attr' in config:
            self.use_edge_attr = config['use_edge_attr']
        else:
            self.use_edge_attr = False
            
        self.JK = JK
        self.emb_dim = emb_dim
        self.dim_target = dim_target
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNN_node(dim_features, num_layer, emb_dim, JK = JK, drop_ratio = self.drop_ratio,
                                 residual = residual, edge_attr_dim=edge_attr_dim, use_edge_attr = self.use_edge_attr)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.dim_target)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.dim_target)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)