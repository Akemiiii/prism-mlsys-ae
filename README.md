# Prism Artifact Evaluation
This is the artifact evaluation repo for PRISM.

## Getting Started Instructions

### Hardware setup

#### Using the Provided Machine

Please spin up a machine with the following characteristics:
- At least one GPU with more than 80GB HBM (**Recommended**: 4 x NVIDIA A100-SXM4-80GB GPUs)
- 200GB disk memory

### Preparation

**Recommended: Use the provided docker image `akemiiii/prism-mlsys26-ae:latest` to run the experiments. If you use the docker image, you can skip the steps 1-5 below.**

Please note: if you are using a shared server with other reviewers, please avoid environment conflicts:
- If you use Docker, make sure your container name is unique to avoid collisions with other running containers.
- Please pay attention to the possible port conflict with other reviewers.

#### 1. Please download the code with 

```bash
git clone https://github.com/Akemiiii/prism-mlsys-ae.git
```

#### 2. Install the Prism environment implemented on top of SGLang with

```bash
conda create -n prism-sglang python=3.12 -y
conda activate prism-sglang
pip install -r sglang/requirements.txt
pip install -e "python[all]"
cd ../..
```

#### 3. Install another python environment for some experiments with SGLang 0.5.4
```bash
conda activate prism-sglang
python -m venv sglang054
./sglang054/bin/activate
pip install sglang==0.5.4
deactivate
```
If you do not have `conda` installed, you can install it following ths [link](https://www.anaconda.com/docs/getting-started/miniconda/install).

#### 4. Install python enviroment for Prism implemented on top of pytorch
```bash
conda create -n prism python=3.12 -y
conda activate prism
pip install -r PRISM/requirements.txt
```

#### 5. Install python environment for eagle3
```bash
conda create -n eagle3 python=3.12 -y
conda activate eagle3
pip install -r Eagle3/requirements_eagle3.txt
```

#### 6. Download the models with
```bash
cd evaluation/
bash model_download.sh
cd ..
```

By default, `model_download.sh` downloads models from ModelScope.
If the download speed is too slow, you can use the backup Hugging Face script instead:

```bash
cd evaluation/
bash model_download_hf.sh
cd ..
```

Note that the `models` directory will be created in the root directory of the repository.

#### 7. Use the docker image
If you use the docker image to run all the experiments, please follow the steps below:
```bash
docker pull akemiiii/prism-mlsys26-ae:latest
docker run -it -v ./models:/prism/models --gpus all prism-mlsys26-ae:latest
```


## Running the Experiments

Here are step-by-step instructions for reproducing the evaluation results in the paper. 

Please note that, due to the difference machine setups, the evaluation results will be different from what shown in the paper. However, the evaluation results should follow similar trend with what we've shown in the paper and support the same claims.

### Preparations for Figure 1, 4, 5, 6（6 hours for 8 x NVIDIA A800 GPUs）
To reproduce the results in Figure 1, 4, 5, 6. We first evaluate all draft model architectures on all target models, which will generate log files recoding accept length and conditional acceptance ratio. The log files will later be parsed for drawing the diagrams. There are 8 combinations of target and draft models. For each combination, accept length experiments on 6 benchmarks and 2 different temprature settings are conducted. Each combination on all 12 runs takes around 45 minutes. To conduct the experiment, run

```bash
cd evaluation/prep_figure1456
bash eval_acceptance_length.sh
```

This will take a long time to finish. If you prefer less benchmarks to save your time, pass valid benchmark name (mt_bench, humaneval, gsm8k, alpaca, sum, qa) to the script so that experiments for each combination will just run on selected benchmarks. For example:

```bash
bash eval_acceptance_length.sh --benches humaneval,qa
```

### Figure 4 （No extra time）
After all experiements finished, run the following script to draw the plot. Note that the plot will report missing log files if you did not run the corresponding benches. Missing values will be replaced by hard-coded values.

```bash
cd evaluation/figure4
bash draw_figure4.sh
```

### Figure 1 （No extra time）
Draw figure 1 with

```bash
cd evaluation/figure1
bash draw_figure1.sh
```

### Figure 5 （No extra time）
Draw figure 5 with

```bash
cd evaluation/figure5
bash draw_figure5.sh
```

### Figure 6 （No extra time）
Draw figure 6 with

```bash
cd evaluation/figure6
bash draw_figure6.sh
```

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

### Figure 8 （1 hour）

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

The results are saved in `evaluation/table4`, including `evaluation/table4/e2e_llama2.csv`, `evaluation/table4/e2e_llama3.csv`, and `evaluation/table4/table4.pdf`. You could compare the generated `table4.pdf` with Table 4 in the paper to verify that the reproduced results match the reported performance.

### Preparations for Figure 1,4,5,6（6 hour for 8 x NVIDIA A800 GPUs）
To reproduce the results in Figure 1,4,5,6. We first evaluate all draft model architectures on all target models, which will generate log files recoding accept length and conditional acceptance ratio. The log files will later be parsed for drawing the diagrams. There are 8 combinations of target and draft models. For each combination, accept length experiments on 6 benchmarks and 2 different temprature settings are conducted. Each combination on all 12 runs takes around 45 minutes. To conduct the experiment, run

```bash
cd evaluation/prep_figure1456
bash eval_acceptance_length.sh --gpus 0,1
```

Use `--gpus` to assign accelerators.This will take a long time to finish. If you prefer less benchmarks to save your time, pass valid benchmark name (mt_bench, humaneval, gsm8k, alpaca, sum, qa) to the script so that experiments for each combination will just run on selected benchmarks. For example:

```bash
bash eval_acceptance_length.sh --gpus --benches humaneval,qa
```

The results are saved in `evaluation/prep_figure1456/outputs`. The log file for each experiment is saved at `evaluation/prep_figure1456/outputs/<TARGET MODEL NAME>/<DRAFT MODEL NAME>/latest/logs/`, where the acceptance length and conditional acceptance rate are reported.

You could also go into the code files to conduct a single experiment. For example, if you want to test PRISM on Llama-3-8B on MT-bench, run:
```bash
cd PRISM
CUDA_VISIBLE_DEVICES=0,1 PROJECT=Llama-3-8B MODEL=PRISM BENCHES=mt_bench bash eval_all.sh 
```

Note that the PRISM code is only for the PRISM model; Eagle3 code is only for the Eagle3 model； All other models use the PRISM_legacy code for experimenting.

### Figure 1 （No extra time）
Draw figure 1 with

```bash
cd evaluation/figure1
bash draw_figure1.sh
```
The output image should be stored as a pdf file in evaluation/figure1. Compare it with the image in the paper.

### Figure 4 （No extra time）
After all experiements finished, run the following script to draw the plot. Note that the plot will report missing log files if you did not run the corresponding benches. Missing values will be replaced by hard-coded values.

```bash
cd evaluation/figure4
bash draw_figure4.sh
```

The output image should be stored as a pdf file in evaluation/figure4. Compare it with the image in the paper.

### Figure 5 （No extra time）
Draw figure 5 with

```bash
cd evaluation/figure5
bash draw_figure5.sh
```

The output image should be stored as a pdf file in evaluation/figure5. Compare it with the image in the paper.

### Figure 6 （No extra time）
Draw figure 6 with

```bash
cd evaluation/figure6
bash draw_figure6.sh
```

The output image should be stored as a pdf file in evaluation/figure6. Compare it with the image in the paper.