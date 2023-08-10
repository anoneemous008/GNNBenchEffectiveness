model_set='GIN_attr GIN_mix GIN_degree Baseline_mlp EGNN_mix'
model_set='EGNN_attr EGNN_mix'
model_set='GCN_degree GIN_degree'


gpu=01
dt=0605
model_set='GIN_degree'
dats='syn_cc'
paras='0.1 0.2 0.3 0.4'
class_num='class2'

for ms in ${model_set};do

conf_file=config_${ms}.yml

for dat in ${dats};do

for para in ${paras};do
echo 'running '${conf_file}

tag=${ms}_${dat}_${para}_${class_num}

nohup python3 -u Launch_Experiments.py --config-file gnn_comparison/${conf_file} \
--dataset-name ${dat} \
--dataset_para ${para}_${class_num} \
--result-folder results/result_${dt}_${tag} --debug > logs/${gpu}_${dt}_${tag}_nohup.log 2>&1 &

echo '    check log:'
echo 'tail -f logs/'${gpu}_${dt}_${tag}'_nohup.log'

done
done
done