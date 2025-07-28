#!/bin/bash
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=100G
#SBATCH --output=d_kg_output.txt
#SBATCH --qos=short
#SBATCH --job-name=d_kg

source ds_kg_venv/bin/activate

python deduplicate_kg.py