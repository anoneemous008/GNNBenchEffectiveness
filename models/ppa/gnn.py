import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from models.ppa.conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class OGBGNN(torch.nn.Module):
    def __init__(self, dim_features, edge_attr_dim, dim_target, config, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = False, residual = False, 
                    drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(OGBGNN, self).__init__()
        self.dim_target = dim_target
        if 'gnn_type' in config:
            gnn_type = config['gnn_type']
        
        self.embeddings_dim = [
            config['hidden_units'][0]] + config['hidden_units']
        self.num_layer = len(self.embeddings_dim)
        
        print('num_layer', self.num_layer)
        self.drop_ratio = config['dropout']
        
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = dim_target
        self.graph_pooling = config['aggregation']

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(dim_features, num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(dim_features, num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


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
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    OGBGNN(num_class = 10)