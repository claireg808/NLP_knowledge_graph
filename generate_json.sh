#!/bin/bash
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=100G
#SBATCH --output=json_output.txt
#SBATCH --qos=short
#SBATCH --job-name=json

module load python/3.11
module load cuda/12.4
source ds_kg_venv/bin/activate

python ds_kg_json.py
