from torch_geometric import data
import numpy as np
import torch

class Data(data.Data):
    def __init__(self,
                 x=None,
                 edge_index=None,
                 edge_attr=None,
                 y=None,
                 v_outs=None,
                 e_outs=None,
                 g_outs=None,
                 o_outs=None,
                 laplacians=None,
                 v_plus=None,
                 **kwargs):

        additional_fields = {
            'v_outs': v_outs,
            'e_outs': e_outs,
            'g_outs': g_outs,
            'o_outs': o_outs,
            'laplacians': laplacians,
            'v_plus': v_plus

        }
        super().__init__(x, edge_index, edge_attr, y, **additional_fields)
        self.N = self.x.shape[0]
        
    def set_additional_attr(self, attr_name, attr_value):
        setattr(self, attr_name, attr_value)
        
    def to_numpy_array(self):
        self.N = self.x.shape[0]
        m = np.zeros((self.N, self.N))
        m[self.edge_index[0], self.edge_index[1]] = 1
        return m
    
    def from_pyg_data(cur:data):
        # NOTE:to float and long
    
        # cur.x = torch.tensor(cur.x).float()
        cur.y = torch.tensor(cur.y).long().squeeze()

        return Data(x=cur.x,
                 edge_index=cur.edge_index,
                 edge_attr=cur.edge_attr,
                 y=cur.y,
                 laplacians=None,
                 v_plus=None)
        
        
class Batch(data.Batch):
    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        laplacians = None
        v_plus = None

        if 'laplacians' in data_list[0]:
            laplacians = [d.laplacians[:] for d in data_list]
            v_plus = [d.v_plus[:] for d in data_list]

        copy_data = []
        # batch_graph_features:
        batch_graph_features = []
        for d in data_list:
            if len(d.y.shape) < 1:
                d.y = d.y.reshape(1)
                
            if d.y.shape[0] > 1:
                copy_data.append(Data(x=d.x,
                                    y=d.y.unsqueeze(dim=0),
                                    edge_index=d.edge_index,
                                    edge_attr=d.edge_attr,
                                    v_outs=d.v_outs,
                                    g_outs=d.g_outs,
                                    e_outs=d.e_outs,
                                    o_outs=d.o_outs)
                                )
            else:
                copy_data.append(Data(x=d.x,
                                y=d.y,
                                edge_index=d.edge_index,
                                edge_attr=d.edge_attr,
                                v_outs=d.v_outs,
                                g_outs=d.g_outs,
                                e_outs=d.e_outs,
                                o_outs=d.o_outs)
                            )
                
            if hasattr(d, 'g_x'):
                batch_graph_features.append(d.g_x)

        batch = data.Batch.from_data_list(copy_data, follow_batch=follow_batch)
        batch['laplacians'] = laplacians
        batch['v_plus'] = v_plus
        if len(batch_graph_features) > 0:
            N = len(batch_graph_features)
            batch['g_x'] = torch.stack(batch_graph_features, dim=0).reshape(N, -1)
        
        return batch
