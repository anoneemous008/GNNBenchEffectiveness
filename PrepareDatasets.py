import argparse

from datasets import *
from datasets.manager import *

DATASETS_used = {
    'REDDIT-BINARY': RedditBinary,
    'COLLAB': Collab,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-MULTI': IMDBMulti,
    'NCI1': NCI1,
    'AIDS': AIDS,
    'ENZYMES': Enzymes,
    'PROTEINS': Proteins,
    'DD': DD,
    "MUTAG": Mutag,
    'CIFAR10': CIFAR10,
    'MNIST': MNIST,
    'ogbg_molhiv':OGBHIV,
    'ogbg_ppa':OGBPPA,
    'ogbg_moltox21': OGBTox21,
    'ogbg-molbace': OGBBACE
}

DATASETS = {
    'REDDIT-BINARY': RedditBinary,
    'REDDIT-MULTI-5K': Reddit5K,
    'COLLAB': Collab,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-MULTI': IMDBMulti,
    'NCI1': NCI1,
    'AIDS': AIDS,
    'ENZYMES': Enzymes,
    'PROTEINS': Proteins,
    'DD': DD,
    "MUTAG": Mutag,
    'CSL': CSL,
    'CIFAR10': CIFAR10,
    'MNIST': MNIST,
    'PPI': PPI,
    'hiv': HIV,
    'bace':BACE,
    'bbpb':BBPB,
    'ogbg_molhiv':OGBHIV,
    'ogbg_ppa':OGBPPA,
    'PTC': PTC,
    'QM9':QM9,
    'ogbg_moltox21': OGBTox21,
    'ogbg-molbbbp': OGBBBBP,
    'ogbg-molbace': OGBBACE,
    'syn_cc': SynCC,
    'syn_degree': SynDegree,
}

SYN_DATASETS = {
    'CSL': CSL
}


def get_args_dict():
    parser = argparse.ArgumentParser()

    parser.add_argument('DATA_DIR',
                        help='where to save the datasets')
    parser.add_argument('--dataset-name', dest='dataset_name',
                        choices=list(DATASETS.keys())+list(SYN_DATASETS.keys())+['all'], default='all', help='dataset name [Default: \'all\']')
    parser.add_argument('--outer-k', dest='outer_k', type=int,
                        default=10, help='evaluation folds [Default: 10]')
    parser.add_argument('--inner-k', dest='inner_k', type=int,
                        default=None, help='model selection folds [Default: None]')
    parser.add_argument('--use-one', action='store_true',
                        default=False, help='use 1 as feature')
    parser.add_argument('--use-degree', dest='use_node_degree', action='store_true',
                        default=False, help='use degree as feature')
    parser.add_argument('--use-shared', dest='use_shared', action='store_true',
                        default=False, help='use shared vector as feature')
    parser.add_argument('--use-1hot', dest='use_1hot', action='store_true',
                        default=False, help='use 1hot vector as feature')
    parser.add_argument('--use-random-normal', dest='use_random_normal', action='store_true',
                        default=False, help='use randomly initializatied vector as feature')
    parser.add_argument('--use-pagerank', dest='use_pagerank', action='store_true',
                        default=False, help='use pagerank value as feature')
    parser.add_argument('--use-eigen', dest='use_eigen', action='store_true',
                        default=False, help='use eigen vectors as feature')
    parser.add_argument('--use-eigen-norm', dest='use_eigen_norm', action='store_true',
                        default=False, help='use degree-normalized eigen vectors as feature')
    parser.add_argument('--use-deepwalk', dest='use_deepwalk', action='store_true',
                        default=False, help='use deepwalk embeddings as feature')
    parser.add_argument('--no-kron', dest='precompute_kron_indices', action='store_false',
                        default=True, help='don\'t precompute kron reductions')
    
    return vars(parser.parse_args())


def preprocess_dataset(name, args_dict):
    
    dataset_class = DATASETS[name] if name in DATASETS else SYN_DATASETS[name]
    
    if name == 'ENZYMES':
        args_dict.update(use_node_attrs=True)
    
    dataset_class(**args_dict)


if __name__ == "__main__":
    args_dict = get_args_dict()
    print(args_dict)

    dataset_name = args_dict.pop('dataset_name')
    if dataset_name == 'all':
        for name in DATASETS:
            preprocess_dataset(name, args_dict)
    else:
        preprocess_dataset(dataset_name, args_dict)