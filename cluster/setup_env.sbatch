#!/bin/bash
#SBATCH --job-name=fm-setup-env
#SBATCH --partition=aihub
#SBATCH --account=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%x_%j.out
# One-time environment build as a batch job (on a GPU node, so the CUDA
# toolkit matches the driver). Submit from the project folder:
#   mkdir -p logs && sbatch --mail-user=<you> --mail-type=END,FAIL cluster/setup_env.sbatch
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
bash cluster/setup_env.sh
