#!/bin/bash
# Sourced by every job: conda env, secrets from .env, and project-wide settings.
: "${PROJECT_DIR:?PROJECT_DIR must be set}"
set -a; source "$PROJECT_DIR/.env"; set +a
source "${CONDA_SH:-/mnt/tank/scratch/dbrice/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-frmedqa-v2}"

export FRMEDQA_PROFILE="${FRMEDQA_PROFILE:-full}"
export FRMEDQA_TMP="${TMPDIR:-/tmp}/frmedqa_${USER}_${SLURM_JOB_ID:-local}"   # node-local scratch
mkdir -p "$FRMEDQA_TMP"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export JUPYTER_RUNTIME_DIR="${FRMEDQA_TMP}/jupyter-runtime"
if [ -z "${WANDB_API_KEY:-}" ]; then export WANDB_DISABLED=true WANDB_MODE=disabled; fi
module load cuda 2>/dev/null || true      # nvcc for the llama.cpp judge build (conda toolkit also works)
