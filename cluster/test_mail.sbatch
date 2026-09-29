#!/bin/bash
#SBATCH --job-name=fm-test-mail
#SBATCH --partition=aihub
#SBATCH --account=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/%x_%j.out
# Checks that email works FROM A COMPUTE NODE (where the real jobs run).
#   sbatch cluster/test_mail.sbatch     then read logs/fm-test-mail_<id>.out and your inbox
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
export PROJECT_DIR="$PWD"
source cluster/common.sh
echo "node: $(hostname)"
python cluster/notify.py test && echo "✓ mail sent" || echo "✗ mail failed (see messages above)"
