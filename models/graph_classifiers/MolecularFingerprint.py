import torch
from torch.nn import ReLU
from torch import dropout, nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from models.graph_classifiers.GIN import MyAtomEncoder


class MolecularGraphMLP(torch.nn.Module):

    def __init__(self, dim_features, edge_attr_dim, dim_target, config):
        super(MolecularGraphMLP, self).__init__()
        hidden_dim = config['hidden_units']
        dropout = config['dropout'] if 'dropout' in config else 0.4
        print('dim_features: ', dim_features)
        print('hidden_dim: ', hidden_dim)
        print('dim_target: ', dim_target)
        self.dim_target = dim_target
        print('dropout:', dropout)
        self.act_func = nn.ReLU
        if 'activation' in config and config['activation'] == 'sigmoid':
            self.act_func = nn.Sigmoid
        self.bn1 = nn.BatchNorm1d(dim_features)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_features, hidden_dim), self.act_func(),
                                       nn.Dropout(dropout),
                                       torch.nn.Linear(hidden_dim, dim_target),  self.act_func())

    def forward(self, data):
        if 'g_x' in data:
            if data['g_x'].dim() == 1:
                h_g = data['g_x'].unsqueeze(dim=1)
            else:
                h_g = data['g_x']
            
            if h_g.shape[0] > 1:
                h_g = self.bn1(h_g)
            result = self.mlp(h_g)
            
            # print('result: ', result)
            return result
                
        return self.mlp(global_add_pool(data.x, data.batch))
    


class MolecularFingerprint(torch.nn.Module):

    def __init__(self, dim_features, edge_attr_dim, dim_target, config):
        super(MolecularFingerprint, self).__init__()
        hidden_dim = config['hidden_units']
        print('finger dim_features:', dim_features)
        self.dim_target = dim_target
        self.mlp = nn.Sequential(nn.BatchNorm1d(dim_features),
                                nn.Linear(dim_features, hidden_dim), ReLU(),
                                nn.Dropout(config['dropout']),
                                # nn.Linear(hidden_dim, hidden_dim), ReLU(),
                                # nn.BatchNorm1d(hidden_dim),
                                nn.Linear(hidden_dim, dim_target), ReLU())
        
    def forward(self, data):
        # return self.mlp(global_mean_pool(data.x.float(), data.batch))
        h = global_add_pool(data.x.float(), data.batch)
        return self.mlp(h)


class AtomMLP(torch.nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super(AtomMLP, self).__init__()

        self.config = config
        self.dropout = config['dropout']
        self.embeddings_dim = config['hidden_dim']
        hidden_dim = self.embeddings_dim
        
        self.node_encoder = MyAtomEncoder(self.embeddings_dim)
        # self.ln = nn.Linear(dim_features, self.embeddings_dim)
        self.mol = True
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mlp = nn.Sequential(nn.BatchNorm1d(hidden_dim),
                                nn.Linear(hidden_dim, hidden_dim), ReLU(),
                                nn.Dropout(config['dropout']),
                                nn.Linear(hidden_dim, hidden_dim), ReLU())
        
        self.out = nn.Linear(hidden_dim, dim_target)
        self.reset_parameters()

    def reset_parameters(self):
        if self.mol:
            for emb in self.node_encoder.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            nn.init.xavier_uniform_(self.node_encoder.weight.data)


    def forward(self, data=None, x=None, edge_index=None, edge_attr=None, batch=None):
        if data is not None:
            x, _, _, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.node_encoder(x)
        h = global_add_pool(h, batch=batch)
        h = self.out(self.mlp(h))

        return h
    