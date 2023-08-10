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
dat="CSL"
dat='COLLAB'
dat='REDDIT-BINARY'
dat='IMDB-BINARY' # no attribute
dat="PROTEINS"
dat='MUTAG'
dat='DD'
dat='NCI1'
dat='ENZYMES'

dats='PATTERN'
dats='ogbg_molhiv'
dats='CIFAR10'
dats='COLLAB REDDIT-BINARY'
dats='REDDIT-BINARY'
dats='AIDS'
dats='ogbg-molbbbp'
dats='ogbg_moltox21'
dats='ogbg_moltox21 ogbg-molbace ogbg_molhiv'


model_set='GIN_attr GIN_mix GIN_degree Baseline_mlp EGNN_mix'
model_set='EGNN_attr EGNN_mix'
model_set='GCN_degree GIN_degree'


gpu=01
dt=0605
model_set='GIN_degree'
dats='MUTAG NCI1 PROTEINS DD'

for ms in ${model_set};do

conf_file=config_${ms}.yml

for dat in ${dats};do

echo 'running '${conf_file}

tag=${ms}_${dat}

# NOTE: if use ogb dataset, set following parameters:
# --outer-folds 1 \
# --inner-folds 1 \
# --ogb_evl True \
# --mol_split True \

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--dataset-name ${dat} \
--dataset_para ${para} \
--result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'

done
done