#!/bin/bash
# Submit a single notebook as one job (tests, or re-running one stage).
#   bash cluster/submit_one.sh notebooks/NB1_DataEngineering.ipynb shared
#   bash cluster/submit_one.sh notebooks/NB3_SFT_NoPPL.ipynb seed 43
set -euo pipefail
cd "$(dirname "$0")/.."; PROJECT_DIR="$(pwd)"
set -a; source .env; set +a
NBF="$1"; SCOPE="${2:-shared}"; SEED="${3:-}"; MEM="${MEM:-48G}"
mkdir -p logs
MAILOPT=(); [ -n "${MAIL_TO:-}" ] && MAILOPT=(--mail-user="$MAIL_TO" --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80)
name="$(basename "$NBF" .ipynb)${SEED:+-s$SEED}"
sbatch --job-name="fm-$name" --mem="$MEM" "${MAILOPT[@]}" --chdir="$PROJECT_DIR" \
       --output="$PROJECT_DIR/logs/%x_%j.out" --export=ALL,PROJECT_DIR="$PROJECT_DIR" \
       cluster/run_nb.sbatch "$NBF" "$SCOPE" $SEED
