data_names='PROTEINS ENZYMES NCI1 DD MUTAG'



data_names='ogbg_moltox21 ogbg-molbace ogbg_molhiv'

for ms in ${data_names};do
    echo 'less logs/kernels_'${ms}_nohup.log
    nohup python3 -u kernel_baselines.py ${ms} > logs/kernels_${ms}_nohup.log 2>&1 &
done


