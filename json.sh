#!/bin/bash
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=100G
#SBATCH --output=json_output.txt
#SBATCH --qos=short
#SBATCH --job-name=json

source ds_kg_venv/bin/activate

python ds_kg_json.py
