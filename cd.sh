#!/bin/bash
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=100G
#SBATCH --output=cd_output.txt
#SBATCH --qos=short
#SBATCH --job-name=cd

source ds_kg_venv/bin/activate

python community_detection.py