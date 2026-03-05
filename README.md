# Prism Artifact Evaluation
This is the artifact evaluation repo for PRISM.

## Getting Started Instructions

### Hardware setup

#### Using the Provided Machine

Please spin up a machine with the following characteristics:
- GPU(s) with more than 80GB HBM (**Recommended**: NVIDIA A100-SXM4-80GB GPU)
- Docker support with NVIDIA container runtime
- 100GB+ disk memory

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

## Running the Experiments

Here are step-by-step instructions for reproducing the evaluation results in the paper. 

Please note that, due to the difference machine setups mentioned in Due to the difference in machine setups mentioned in the second section, the evaluation results will be different from what shown in the paper. However, the evaluation results should follow similar trend with what we've shown in the paper and support the same claims.

Each script list below launch a sequence of experiments with different configurations. The evaluation of a single configuration takes around 15 minutes.

### Figure 8 and 9

Use the following commands to run the experiments for Figure 8 and 9. To run the evaluation for LLaMA-3.1-70B-Instruct:

```bash
ADASERVE=ON RPS_MIN=2.6 RPS_MAX=4.8 ./exps/fig8,9/run_llama_rps.sh
```

To run the evaluation for Qwen2.5-32B-Instruct:

```bash
ADASERVE=ON RPS_MIN=2.4 RPS_MAX=4.2 ./exps/fig8,9/run_qwen_rps.sh
```

`RPS_MIN` and `RPS_MAX` can be adjusted to cover different RPS ranges. The minimal RPS is 2.6 and the maximal RPS is 4.8 for LLaMA-3.1-70B-Instruct on our evaluation. The minimal RPS is 2.4 and the maximal RPS is 4.2 for Qwen2.5-32B-Instruct on our evaluation. The minimal step size is set to 0.2.


The results are saved in `results/fig8,9/llama/adaserve/` and `results/fig8,9/qwen/adaserve/`.

### Figure 10

Use the following commands to run the experiments for Figure 10. To run the evaluation for LLaMA-3.1-70B-Instruct:

```bash
ADASERVE=ON PROP_MIN=0.2 PROP_MAX=0.9 ./exps/fig10/run_llama_prop.sh
```

To run the evaluation for Qwen2.5-32B-Instruct:

```bash
ADASERVE=ON PROP_MIN=0.2 PROP_MAX=0.9 ./exps/fig10/run_qwen_prop.sh
``` 

`PROP_MIN` and `PROP_MAX` can be adjusted to cover different proportion ranges. The minimal proportion is 0.1 and the maximal proportion is 0.9 for both LLaMA-3.1-70B-Instruct and Qwen2.5-32B-Instruct on our evaluation. The minimal step size is 0.1.

The results are saved in `results/fig10/llama/adaserve/` and `results/fig10/qwen/adaserve/`.

### Figure 11

Use the following commands to run the experiments for Figure 11. To run the evaluation for LLaMA-3.1-70B-Instruct:

```bash
ADASERVE=ON SLO_SCALE_MIN=0.6 SLO_SCALE_MAX=1.6 OUTPUT_LENGTH=256 ./exps/fig11/run_llama_slo.sh
```

To run the evaluation for Qwen2.5-32B-Instruct:

```bash
ADASERVE=ON SLO_SCALE_MIN=0.6 SLO_SCALE_MAX=1.6 OUTPUT_LENGTH=256 ./exps/fig11/run_qwen_slo.sh
```

`SLO_SCALE_MIN` and `SLO_SCALE_MAX` can be adjusted to cover different SLO ranges. The minimal SLO scale is 0.6 and the maximal SLO scale is 1.6 for both LLaMA-3.1-70B-Instruct and Qwen2.5-32B-Instruct on our evaluation. The minimal step size is 0.2.

The results are saved in `results/fig11/llama/adaserve/` and `results/fig11/qwen/adaserve/`.

### Figure 12

The data for Figure 12 is collected during the experiments for Figure 8 and 9. You can find the data in `results/fig8,9/llama/adaserve/` and `results/fig8,9/qwen/adaserve/`. The number is reported in the line starting with `mean_generated_tokens_per_step` at the end of the files.

### Figure 14   

Use the following commands to run the experiments for Figure 14. To run the evaluation for LLaMA-3.1-70B-Instruct:

```bash
ADASERVE=ON ./exps/fig14/run_llama_fluc.sh
```

To run the evaluation for Qwen2.5-32B-Instruct:

```bash
ADASERVE=ON ./exps/fig14/run_qwen_fluc.sh
```

The results are saved in `results/fig14/llama/adaserve/` and `results/fig14/qwen/adaserve/`.

### Figure 15

Use the following commands to run the experiments for Figure 15. To run the evaluation for LLaMA-3.1-70B-Instruct:

```bash
LLAMA_OVERHEAD=ON ./exps/fig15/run_overhead_breakdown.sh
```

To run the evaluation for Qwen2.5-32B-Instruct:

```bash
QWEN_OVERHEAD=ON ./exps/fig15/run_overhead_breakdown.sh
```

The results are saved in `results/fig15/llama/` and `results/fig15/qwen/`.