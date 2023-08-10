import sys,os
sys.path.append(os.getcwd())

import argparse
from EndToEnd_Evaluation import main as endtoend
from PrepareDatasets import DATASETS
from config.base import Grid, Config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--experiment', dest='experiment', default='endtoend')
    parser.add_argument('--result-folder', dest='result_folder', default='RESULTS')
    parser.add_argument('--dataset-name', dest='dataset_name', default='none')
    parser.add_argument('--dataset_para', dest='dataset_para',type=str, default='0.9')
    parser.add_argument('--outer-folds', dest='outer_folds', default=10)
    parser.add_argument('--outer-processes', dest='outer_processes', default=2)
    parser.add_argument('--inner-folds', dest='inner_folds', default=5)
    parser.add_argument('--inner-processes', dest='inner_processes', default=1)
    parser.add_argument('--debug', action="store_true", dest='debug')
    parser.add_argument('--mol_split', type=bool, dest='mol_split', default=False)
    parser.add_argument('--ogb_evl', type=bool, dest='ogb_evl', default=False)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.dataset_name not in ['all', 'none']:
        datasets = [args.dataset_name]
    else:
        datasets = list(DATASETS.keys())
        
        # ['IMDB-MULTI', 'IMDB-BINARY', 'PROTEINS', 'NCI1', 'ENZYMES', 'DD',
                    # 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB', 'REDDIT-MULTI-12K']

    config_file = args.config_file
    experiment = args.experiment
    
    for dataset_name in datasets:
        try:
            model_configurations = Grid(config_file, dataset_name)
            # NOTE: override value from args.
            model_configurations.override_by_dict(args.__dict__)
            
            endtoend(model_configurations,
                     outer_k=int(args.outer_folds), outer_processes=int(args.outer_processes),
                     inner_k=int(args.inner_folds), inner_processes=int(args.inner_processes),
                     result_folder=args.result_folder, debug=args.debug)
        
        except Exception as e:
            raise e  # print(e)