# Rethinking the Effectiveness of Graph Classification Datasets in Benchmarks for Assessing GNNs
----
**It's widely acknowledged that the quality of a dataset directly affects the performance of a model, much like in the fields of computer vision (CV) and natural language processing (NLP). However, in the domain of graph learning, the standards are not as mature. Determining the quality of a dataset is a crucial and intriguing challenge.**

---
## Abstract

Graph classification benchmarks, vital for assessing and developing graph neural network (GNN) models, have recently been scrutinized, as simple methods like MLPs have demonstrated comparable performance on certain datasets. This leads to an important question: 

* Do these benchmarks effectively distinguish the advancements of GNNs over other methodologies? If so, how do we quantitatively measure this effectiveness?

In response, we propose an empirical protocol based on a fair benchmarking framework to investigate the performance discrepancy between simple methods and GNNs. We further propose a novel metric to quantify the effectiveness of a dataset by utilizing the performance gaps and considering dataset complexity. Through extensive testing across 16 real-world datasets, we found our metric to align with existing studies and intuitive assumptions. Finally, to explore the causes behind the low effectiveness, we investigated the relationship between intrinsic graph properties and task labels and developed a novel technique for generating more synthetic datasets that can precisely control these correlations. Our findings shed light on the current understanding of benchmark datasets, and our new platform backed by an effectiveness validation protocol could fuel the future evolution of graph classification benchmarks.

## Prepare Datasets:

- `mkdir DATA`
- set the dataset name in `run_real_experiment.sh`, then `bash run_real_experiment.sh` will automatically download the dataset in ./DATA and run the benchmark, but if you only need to download, you could run `prepare_experiment.sh`.
- **NOTE**, some bash parameters are required to set in `run_real_experiment.sh`, such as `dats` (dataset names), `model_set` (models to run, corresponding to each running config file *.yml), etc. You could run in parallel by setting `dats` and `model_set` to multiple values, but we suggest to run one by one to avoid memory issues.

## Configuration of benchmark, all config files are in gnn_comparison/*.yml.

- specify parameters including model name, batch size, lr, feature types, etc in  `gnn_comparison/*.yml` files.
- for simplicity, we have seperate each main config into different config files, such as `config_GIN_attr.yml`, `config_GCN_degree.yml`, etc.

## Run benchmark:

- specify the config file name in `run_real_experiment.sh` for real-world datasets, or `run_syn_experiment.sh` for synthetic datasets.
- set config file parameters in `config_Baseline_[xxxx].yml` or `config_GIN_[xxxx].yml`, `config_GCN_[xxxx].yml`, etc. Check details in `gnn_comparison/*.yml`.
- run benchmark: `bash gnn_comparison/run_real_experiment.sh` or `bash gnn_comparison/run_syn_experiment.sh`.
- **NOTE**, all logs and results locations are specified in `run_real_experiment.sh` and `run_syn_experiment.sh` files. The results are saved in `./results/` folder for further performance analysis, the folder name will be used for extracting statistics in `plot_performance_gaps.ipynb` and `plot_statistics.ipynb`.


## Generate performance gap and effectiveness of benchmark results:

- the results in paper were generated in `plot_performance_gaps.ipynb`
- some statistics of datasets are in `plot_statistics.ipynb`

## Grpah kernel baselines:

- all kernels are in kernel_baseline.ipynb
- or `bash run_kernel_baseline.sh` for parallel processing.

## Regression:

- run generate in `generate_regression_datasets.py`
- run regression in `regressor.ipynb`
