# Prism Artifact Evaluation
This is the artifact evaluation repo for PRISM.

## Getting Started Instructions

### Hardware setup

#### Using the Provided Machine

Please spin up a machine with the following characteristics:
- At least one GPU with more than 80GB HBM (**Recommended**: 4 x NVIDIA A100-SXM4-80GB GPUs)
- 200GB disk memory

### Preparation

**Recommended: Use the provided docker image to run the experiments. If you use the docker image, you can skip the steps 1-3 below.**

Please note: if you are using a shared server with other reviewers, please avoid environment conflicts:
- If you use Docker, make sure your container name is unique to avoid collisions with other running containers.
- If you do not use Docker, please create and use an isolated Python environment before installation.

1. Please download the code with 

```bash
git clone https://github.com/Akemiiii/prism-mlsys-ae.git
```

2. Install the Prism implemented on top of SGLang with

```bash
cd prism-mlsys-ae/sglang/
pip install -e "python[all]"
cd ../..
```

3. Install another python environment for some experiments with
```bash
python -m venv sglang054
./sglang054/bin/activate
pip install sglang==0.5.4
deactivate
```

4. Download the models with
```bash
cd evaluation/
bash model_download.sh
```

By default, `model_download.sh` downloads models from ModelScope.
If the download speed is too slow, you can use the backup Hugging Face script instead:

```bash
cd evaluation/
bash model_download_hf.sh
```

## Running the Experiments

Here are step-by-step instructions for reproducing the evaluation results in the paper. 

Please note that, due to the difference machine setups, the evaluation results will be different from what shown in the paper. However, the evaluation results should follow similar trend with what we've shown in the paper and support the same claims.


### Figure 7 (1.5 hours)

Please use the following commands to run the experiments for Figure 7:

```base
cd evaluation/figure7
./run_figure7.sh
```

The results are saved in `evaluation/figure7` including `evaluation/figure7/e2e_llama2_bs_sweep_avg.csv` and `evaluation/figure7/figure7.pdf`. You could compare the generated `figure7.pdf` with Figure 7 in the paper to verify that the reproduced results match the reported performance trends.

If you are interested in experimenting with different batch size settings, you can pass a custom list of values via the `--batch-size` argument when invoking `evaluation/figure7/e2e_llama2_bs_sweep.py` directly. For example:

```bash
python evaluation/figure7/e2e_llama2_bs_sweep.py --batch-size 10 12
```

The default sweep is `2 4 8 16 32`, matching the settings used in the paper. You can also run `python evaluation/figure7/e2e_llama2_bs_sweep.py --help` to see all available options. The result of this python script will be shown in `evaluation/figure7/e2e_llama2_bs_sweep_avg.csv` and `evaluation/figure7/e2e_llama2_bs_sweep_detail.csv`.

### Figure 8 （1 hour)

Please use the following commands to run the experiments for Figure 8:

```base
cd evaluation/figure8
./run_figure8.sh
```

The results are saved in `evaluation/figure8` including `evaluation/figure8/e2e_llama2_tree_verify_sweep_avg.csv` and `evaluation/figure8/figure8.pdf`. You could compare the generated `figure8.pdf` with Figure 8 in the paper to verify that the reproduced results match the reported performance trends.

If you are interested in experimenting with different tree settings, you can pass a custom list of values via the `--sweep-configs` argument when invoking `evaluation/figure8/e2e_llama2_tree_verify_sweep.py` directly. For example:

```bash
python evaluation/figure8/e2e_llama2_tree_verify_sweep.py \
  --sweep-configs 3,2,6 5,8,32 8,10,64
```

The default tree sweep config is  matching the settings used in the paper. You can also run `python evaluation/figure8/e2e_llama2_tree_verify_sweep.py --help` to see all available options. The result of this python script will be shown in `evaluation/figure8/e2e_llama2_tree_verify_sweep_avg.csv` and `evaluation/figure8/e2e_llama2_tree_verify_sweep_detail.csv`.

### Table 4 and Table 5 (2 hours for 2 x NVIDIA A800 GPUs)

On a single machine in this AE setup, you can only reproduce Table 4. To reproduce Table 5, you need another machine with a different GPU type.

Please use the following commands to run the Table 4 experiments.

If you have two or more GPUs, pass two GPU IDs (comma-separated). The script will use the first two IDs and run LLaMA-2 and LLaMA-3 in parallel. For example:

```bash
cd evaluation/table4
./run_table4.sh 0,1
```

If you only have one GPU, pass a single GPU ID. The script will run LLaMA-2 and LLaMA-3 sequentially on the same GPU. For example:

```bash
cd evaluation/table4
./run_table4.sh 0
```

The results are saved in `evaluation/table4`, including `evaluation/table4/e2e_llama2.csv`, `evaluation/table4/e2e_llama3.csv`, and `evaluation/table4/table4.pdf`. You could compare the generated `table4.pdf` with Table 4 in the paper to verify that the reproduced results match the reported performance trends.


