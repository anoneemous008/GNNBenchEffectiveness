from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool, GCNConv, SAGEConv
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool, BatchNorm

from ogb.graphproppred.mol_encoder import AtomEncoder



class GIN(torch.nn.Module):
    def __init__(self, dim_features, edge_attr_dim, dim_target, config):
        super(GIN, self).__init__()

        self.config = config
        self.dropout = config['dropout']
        self.embeddings_dim = [
            config['hidden_units'][0]] + config['hidden_units']
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.dim_target = dim_target
        
        train_eps = config['train_eps']
        if config['aggregation'] == 'sum':
            self.pooling = global_add_pool
        elif config['aggregation'] == 'mean':
            self.pooling = global_mean_pool
        elif config['aggregation'] == 'max':
            self.pooling = global_max_pool
        # self.batch0 = BatchNorm1d(dim_features)
        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                          Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                           Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(
                    GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2
                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        # has got one more for initial input
        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, data=None, x=None, edge_index=None, batch=None):
        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0
        # x = self.batch0(x)
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x.float())
                out += F.dropout(self.pooling(
                    self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                # NOTE: residual connection
                out += F.dropout(self.linears[layer](self.pooling(x, batch)),
                                 p=self.dropout, training=self.training)

        return out


class EGINConv(MessagePassing):
    def __init__(self, emb_dim, mol):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(EGINConv, self).__init__(aggr="add")

        self.mol = mol

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(
            2 * emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        # if self.mol:
        #     self.edge_encoder = BondEncoder(emb_dim)
        # else:
        #     self.edge_encoder = nn.Linear(7, emb_dim)

    def reset_parameters(self):
        for c in self.mlp.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.eps.data, 0)

        # if self.mol:
        #     for emb in self.edge_encoder.bond_embedding_list:
        #         nn.init.xavier_uniform_(emb.weight.data)
        # else:
        #     self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        edge_embedding = self.edge_encoder(edge_attr)
        # out = self.mlp((1 + self.eps) * x +
        #                self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        out = self.mlp((1 + self.eps) * x +
                       self.propagate(edge_index, x=x, edge_attr=None))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return F.relu(x_j)
        else:
            return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class EGCNConv(MessagePassing):
    def __init__(self, emb_dim, mol):
        super(EGCNConv, self).__init__(aggr='add')

        self.mol = mol

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        if self.mol:
            self.edge_encoder = BondEncoder(emb_dim)
        else:
            self.edge_encoder = nn.Linear(7, emb_dim)

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.root_emb.reset_parameters()

        if self.mol:
            for emb in self.edge_encoder.bond_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# allowable multiple choice node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
    ]))


full_atom_feature_dims = get_atom_feature_dims()


class MyAtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(MyAtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            print('dim:', dim)
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        """

        x shape: torch.Size([898, 9])
        x0: tensor([5, 0, 4, 5, 3, 0, 2, 0, 0], device='cuda:1')

        atom x: torch.Size([1029, 9])
        atom x0: tensor([7, 0, 1, 4, 0, 0, 3, 0, 0], device='cuda:0')

        """
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding
    


class EGNN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super(EGNN, self).__init__()

        self.config = config
        self.dropout = config['dropout']
        self.embeddings_dim = config['hidden_dim']
        self.num_layers = config['layer_num']
        if 'gnn_type' in config:
            self.gnn_type = config['gnn_type']
        else:
            self.gnn_type = 'gin'
            
        self.node_encoder = AtomEncoder(self.embeddings_dim)
        # self.ln = nn.Linear(dim_features, self.embeddings_dim)
        self.mol = True
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.ln_degree = Linear(1, self.embeddings_dim)

        for i in range(self.num_layers):
            if self.gnn_type == 'gin':
                self.convs.append(EGCNConv((self.embeddings_dim, True)))
            elif self.gnn_type == 'gcn':
                self.convs.append(EGCNConv((self.embeddings_dim, True)))
            else:
                raise ValueError('Undefined gnn type called {}'.format(self.gnn_type))
                
            self.bns.append(torch.nn.BatchNorm1d(self.embeddings_dim))

        self.out =  nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.embeddings_dim, self.embeddings_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.embeddings_dim, dim_target)
        )
        
        # self.out = nn.Linear(self.embeddings_dim, dim_target)
        # self.reset_parameters()
        print('EGNN dim_target: ', dim_target)

    # def reset_parameters(self):
    #     if self.mol:
    #         for emb in self.node_encoder.atom_embedding_list:
    #             nn.init.xavier_uniform_(emb.weight.data)
    #     else:
    #         nn.init.xavier_uniform_(self.node_encoder.weight.data)

    #     for i in range(self.num_layers):
    #         self.convs[i].reset_parameters()
    #         self.bns[i].reset_parameters()

    #     self.out.reset_parameters()
    def forward(self, data=None, x=None, edge_index=None, batch=None):
        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch
    # def forward(self, data=None, x=None, edge_index=None, edge_attr=None, batch=None):
    #     if data is not None:
    #         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.node_encoder(x[:, :9])
        
        if x.shape[-1] > 9:
            h = self.ln_degree(x[:, 9:]) + h

        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index, None)

            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.convs[-1](h, edge_index, None)

        h = F.dropout(h, self.dropout, training=self.training)

        h = global_mean_pool(h, batch)

        h = self.out(h)

        return h


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp

from torch_geometric.utils import to_dense_batch

# from layers import SAB, ISAB, PMA
# from layers import GCNConv_for_OGB, GINConv_for_OGB


# from ogb.graphproppred.mol_encoder import AtomEncoder

# from math import ceil

# class GraphRepresentation(torch.nn.Module):

#     def __init__(self, args):

#         super(GraphRepresentation, self).__init__()

#         self.args = args
#         self.num_features = args.num_features
#         self.nhid = args.num_hidden
#         self.num_classes = args.num_classes
#         self.pooling_ratio = args.pooling_ratio
#         self.dropout_ratio = args.dropout

#     def get_convs(self):

#         convs = nn.ModuleList()

#         _input_dim = self.num_features
#         _output_dim = self.nhid

#         for _ in range(self.args.num_convs):

#             if self.args.conv == 'GCN':
            
#                 conv = GCNConv(_input_dim, _output_dim)

#             elif self.args.conv == 'GIN':

#                 conv = GINConv(
#                     nn.Sequential(
#                         nn.Linear(_input_dim, _output_dim),
#                         nn.ReLU(),
#                         nn.Linear(_output_dim, _output_dim),
#                         nn.ReLU(),
#                         nn.BatchNorm1d(_output_dim),
#                 ), train_eps=False)

#             convs.append(conv)

#             _input_dim = _output_dim
#             _output_dim = _output_dim

#         return convs

#     def get_pools(self):

#         pools = nn.ModuleList([gap])

#         return pools

#     def get_classifier(self):

#         return nn.Sequential(
#             nn.Linear(self.nhid, self.nhid),
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_ratio),
#             nn.Linear(self.nhid, self.nhid//2),
#             nn.ReLU(),
#             nn.Dropout(p=self.dropout_ratio),
#             nn.Linear(self.nhid//2, self.num_classes)
#         )

# class GraphMultisetTransformer(GraphRepresentation):

#     def __init__(self, args):

#         super(GraphMultisetTransformer, self).__init__(args)

#         self.ln = args.ln
#         self.num_heads = args.num_heads
#         self.cluster = args.cluster

#         self.model_sequence = args.model_string.split('-')

#         self.convs = self.get_convs()
#         self.pools = self.get_pools()
#         self.classifier = self.get_classifier()

#     def forward(self, data):

#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         # For Graph Convolution Network
#         xs = []

#         for _ in range(self.args.num_convs):

#             x = F.relu(self.convs[_](x, edge_index))
#             xs.append(x)

#         # For jumping knowledge scheme
#         x = torch.cat(xs, dim=1)

#         # For Graph Multiset Transformer
#         for _index, _model_str in enumerate(self.model_sequence):

#             if _index == 0:

#                 batch_x, mask = to_dense_batch(x, batch)

#                 extended_attention_mask = mask.unsqueeze(1)
#                 extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
#                 extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

#             if _model_str == 'GMPool_G':

#                 batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch))

#             else:

#                 batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask)

#             extended_attention_mask = None

#         batch_x = self.pools[len(self.model_sequence)](batch_x)
#         x = batch_x.squeeze(1)

#         # For Classification
#         x = self.classifier(x)

#         return F.log_softmax(x, dim=-1)

#     def get_pools(self, _input_dim=None, reconstruction=False):

#         pools = nn.ModuleList()

#         _input_dim = self.nhid * self.args.num_convs if _input_dim is None else _input_dim
#         _output_dim = self.nhid
#         _num_nodes = ceil(self.pooling_ratio * self.args.avg_num_nodes)

#         for _index, _model_str in enumerate(self.model_sequence):

#             if (_index == len(self.model_sequence) - 1) and (reconstruction == False):
                
#                 _num_nodes = 1

#             if _model_str == 'GMPool_G':

#                 pools.append(
#                     PMA(_input_dim, self.num_heads, _num_nodes, ln=self.ln, cluster=self.cluster, mab_conv=self.args.mab_conv)
#                 )

#                 _num_nodes = ceil(self.pooling_ratio * _num_nodes)

#             elif _model_str == 'GMPool_I':

#                 pools.append(
#                     PMA(_input_dim, self.num_heads, _num_nodes, ln=self.ln, cluster=self.cluster, mab_conv=None)
#                 )

#                 _num_nodes = ceil(self.pooling_ratio * _num_nodes)

#             elif _model_str == 'SelfAtt':

#                 pools.append(
#                     SAB(_input_dim, _output_dim, self.num_heads, ln=self.ln, cluster=self.cluster)
#                 )

#                 _input_dim = _output_dim
#                 _output_dim = _output_dim

#             else:

#                 raise ValueError("Model Name in Model String <{}> is Unknown".format(_model_str))

#         pools.append(nn.Linear(_input_dim, self.nhid))

#         return pools

# class GraphMultisetTransformer_for_OGB(GraphMultisetTransformer):

#     def __init__(self, args):

#         super(GraphMultisetTransformer_for_OGB, self).__init__(args)

#         self.atom_encoder = AtomEncoder(self.nhid)
#         self.convs = self.get_convs()

#     def forward(self, data):

#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

#         x = self.atom_encoder(x)

#         # For Graph Convolution Network
#         xs = []

#         for _ in range(self.args.num_convs):

#             x = F.relu(self.convs[_](x, edge_index, edge_attr))
#             xs.append(x)

#         # For jumping knowledge scheme
#         x = torch.cat(xs, dim=1)

#         # For Graph Multiset Transformer
#         for _index, _model_str in enumerate(self.model_sequence):

#             if _index == 0:

#                 batch_x, mask = to_dense_batch(x, batch)

#                 extended_attention_mask = mask.unsqueeze(1)
#                 extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
#                 extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

#             if _model_str == 'GMPool_G':

#                 batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch))

#             else:

#                 batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask)

#             extended_attention_mask = None

#         batch_x = self.pools[len(self.model_sequence)](batch_x)
#         x = batch_x.squeeze(1)

#         # For Classification
#         x = self.classifier(x)

#         return x

#     def get_convs(self):

#         convs = nn.ModuleList()

#         for _ in range(self.args.num_convs):

#             if self.args.conv == 'GCN':
            
#                 conv = GCNConv_for_OGB(self.nhid)

#             elif self.args.conv == 'GIN':

#                 conv = GINConv_for_OGB(self.nhid)

#             convs.append(conv)

#         return convs