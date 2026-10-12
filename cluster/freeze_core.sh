#!/bin/bash
# Writes cluster/constraints.txt with the versions currently installed in the
# frmedqa-v2 env, so jobs cannot change them (see common.sh, PIP_CONSTRAINT).
cd "$(dirname "$0")/.."
source "${CONDA_SH:-/mnt/tank/scratch/dbrice/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-frmedqa-v2}"
pip freeze | grep -iE '^(torch|transformers|tokenizers|unsloth|unsloth[_-]zoo|peft|trl|accelerate|bitsandbytes|datasets|huggingface[_-]hub|safetensors|xformers|triton)==' > cluster/constraints.txt
echo "cluster/constraints.txt:"; cat cluster/constraints.txt
