from models import *

from torch.utils.data import Dataset, DataLoader
import torch_geometric.datasets as pygdataset



def assemble_dataloader(batch_size, train_dataset: GraphDataset, test_dataset: GraphDataset, cuda=True):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=test_dataset.collate)
    
    return (train_dataloader, test_dataloader)


def load_data(data_name, args):
    
    if data_name == 'AIDS':
        tudataset = pygdataset.tu_dataset.TUDataset(root='/li_zhengdao/github/GenerativeGNN/dataset/', name='MUTAG')

        tu_base_graphs = []
        for a in tudataset:
            tu_base_graphs.append(BaseGraphUtils.from_pyg_graph(a))
            
        print('x shape', tu_base_graphs[0].pyg_graph.x.shape)
        
        tu_y = []
        for g in tu_base_graphs:
            tu_y.append(g.label)
        tu_y = torch.stack(tu_y, dim=0).squeeze()

        train_tu_dataset = GraphDataset(x=tu_base_graphs, y=torch.LongTensor(tu_y))

        train_x, train_y, test_x, test_y = utils.random_split_dataset(train_tu_dataset, [0.8, 0.2])

        from collections import Counter
        tr_y = Counter(train_y.numpy())
        te_y = Counter(test_y.numpy())
        print('Counter training:', tr_y)
        print('Counter testing:', te_y)
        print('len of train x:', len(train_x))
        print('len of test x:', len(test_x))

        tu_train_dataset = GraphDataset(x=train_x, y=train_y)
        tu_test_dataset = GraphDataset(x=test_x, y=test_y)
        
        if args.cuda:
            tu_train_dataset.cuda()
            tu_test_dataset.cuda()

        tu_train_dataloader, tu_test_dataloader = assemble_dataloader(args.batch_size, tu_train_dataset, tu_test_dataset)
        
        return tu_train_dataloader, tu_test_dataloader