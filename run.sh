#!/bin/bash
#SBATCH --gres=gpu:80g:1
#SBATCH --mem=100G
#SBATCH --output=run_output.txt
#SBATCH --qos=short
#SBATCH --job-name=ds

module load python/3.11
module load cuda/12.8
source ds_kg_venv/bin/activate

vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --max-model-len 35904