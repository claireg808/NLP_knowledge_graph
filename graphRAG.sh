#!/bin/bash
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=100G
#SBATCH --output=graph_rag_output.txt
#SBATCH --qos=short
#SBATCH --job-name=grag

source ds_kg_venv/bin/activate

python graphRAG.py
