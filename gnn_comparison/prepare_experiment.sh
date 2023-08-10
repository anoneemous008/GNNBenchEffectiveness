
#    CHEMICAL:
#         NCI1
#         DD
#         ENZYMES
#         PROTEINS
#    SOCIAL[_1 | _DEGREE]:
#         IMDB-BINARY
#         IMDB-MULTI
#         REDDIT-BINARY
#         REDDIT-MULTI-5K
#         COLLAB
dat='all'
dat='NCI1'
dat='ENZYMES'
dat='DD'
dat="CSL"
dat="PROTEINS"
dat='IMDB-BINARY'
dat='COLLAB'
dat='REDDIT-BINARY'
dat='MUTAG'

# python3 PrepareDatasets.py DATA/SYNTHETIC --dataset-name ${dat} --outer-k 10 --use-degree
# python3 PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10 --use-random-normal
python3 gnn_comparison/PrepareDatasets.py DATA/ --dataset-name ${dat} --outer-k 10