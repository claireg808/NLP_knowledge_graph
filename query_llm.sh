#!/bin/bash
#SBATCH --gres=gpu:80g:1
#SBATCH --mem=100G
#SBATCH --output=query_llm_output.txt
#SBATCH --qos=short
#SBATCH --job-name=query_llm

source ds_kg_venv/bin/activate

export HUGGINGFACE_HUB_TOKEN=$(cat ~/NLP_knowledge_graph/.hf_token)
huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"

# start vLLM in the background
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --served-model-name llama3