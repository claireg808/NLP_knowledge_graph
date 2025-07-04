#!/bin/bash
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=100G
#SBATCH --output=kg_output.txt
#SBATCH --qos=short
#SBATCH --job-name=kg

module load pydantic

source ds_kg_venv/bin/activate

python generate_kg.py
