import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU, Softmax, Dropout
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, BatchNorm
from models.graph_classifiers.GIN import GIN, EGNN


class ModelAdapter(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(ModelAdapter, self).__init__()

        hid_out_dim = config['hidden_units'][-1]

        self.gin1 = GIN(1, hid_out_dim, config)
        self.gin2 = GIN(dim_features-1, hid_out_dim, config)
        self.ln = Linear(2*hid_out_dim, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # NOTE: seperate features:
        out1 = self.gin1(
            x=x[..., -1:], edge_index=edge_index, batch=batch)  # degree
        out2 = self.gin2(x=x[..., :-1], edge_index=edge_index,
                         batch=batch)  # attribute

        out = self.ln(torch.cat([out1, out2], dim=-1))
        return out


class ModelMix(torch.nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super(ModelMix, self).__init__()

        hid_out_dim = config['hidden_units'][-1]
        self.dropout = Dropout(config['dropout'])

        self.gin_degree = GIN(1, dim_target, config)
        self.gin_attr = GIN(dim_features-1, dim_target, config)

        # self.upsample1 = Linear(1, hid_out_dim) # the last dimension is degree
        # self.upsample2 = Linear(dim_features-1, hid_out_dim)


        # self.gin_gated = GIN(hid_out_dim, 1, config)
        # self.ln_gated = Linear(hid_out_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        # softmax ???
        # self.sf = Softmax(dim=-1)
        
        # self.ln = Linear(2*hid_out_dim, dim_target)
        

    def get_gatescore(self, upsample, h1, h2, edge_index, batch):
        x_up = upsample(h1)
        out = torch.cat([self.gin_gated(x=x_up, edge_index=edge_index,
                                        batch=batch), h2], dim=-1)
        return F.sigmoid(self.ln_gated(out))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # NOTE: seperate features:
        out1 = self.gin_degree(
            x=x[..., -1:], edge_index=edge_index, batch=batch)  # degree
        out2 = self.gin_attr(
            x=x[..., :-1], edge_index=edge_index, batch=batch)  # attribute
        
        # h1 = self.ln_gated(out1)
        # h2 = self.ln_gated(out2)
        # # h1 = self.gin_gated(x=self.upsample1(x[..., -1:]), edge_index=edge_index, batch=batch)
        # # h2 = self.gin_gated(x=self.upsample2(x[..., :-1]), edge_index=edge_index, batch=batch)
        
        # att = self.sf(torch.cat([h1, h2], dim=-1))
        # NOTE: energy the softmax:
        alpha = torch.sigmoid(self.alpha)
        p1 = torch.log_softmax(out1, dim=-1)
        p2 = torch.log_softmax(out2, dim=-1)

        with torch.no_grad():
            alpha_2 = 1- alpha.item()

        out = alpha * p1 + alpha_2 *p2

        # out = self.dropout(self.ln(torch.cat([out1,out2], dim=-1)))

        return out


class MolMix(torch.nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super(MolMix, self).__init__()

        hid_out_dim = config['hidden_units'][-1]
        self.dropout = Dropout(config['dropout'])

        self.gin_degree = GIN(1, hid_out_dim, config)
        self.gin_attr = EGNN(dim_features-1, hid_out_dim, config)

        # self.alpha = nn.Parameter(torch.tensor(0.01))
        self.ln = Linear(2*hid_out_dim, dim_target)

    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # NOTE: seperate features:
        out1 = self.gin_degree(
            x=x[..., -1:], edge_index=edge_index, batch=batch)  # degree
        out2 = self.gin_attr(
            x=x[..., :-1].long(), edge_index=edge_index, batch=batch)  # attribute
        
        out = self.dropout(self.ln(torch.cat([out1, out2], dim=-1)))

        return out