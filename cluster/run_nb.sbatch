#!/bin/bash
#SBATCH --partition=aihub
#SBATCH --account=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@900
# Runs one or more notebooks headless with papermill.
# Usage: sbatch cluster/run_nb.sbatch "<nb1.ipynb,nb2.ipynb,...>" <shared|seed> [SEED]
# 15 min before the time limit the job requeues itself; training / evaluation /
# judging all resume from what is already saved.
NBS="$1"; SCOPE="${2:-shared}"; SEED="${3:-}"
export PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR" || exit 1
source cluster/common.sh

if [ "$SCOPE" = "seed" ]; then
  export FRMEDQA_SEED="$SEED"
  bash cluster/prepare_seed_dir.sh "$SEED"
  export FRMEDQA_BASE="$PROJECT_DIR/runs/seed$SEED"
else
  unset FRMEDQA_SEED
  export FRMEDQA_BASE="$PROJECT_DIR"
fi
export JOB_LOG="$PROJECT_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
EXEC_DIR="$FRMEDQA_BASE/executed"; mkdir -p "$EXEC_DIR"

notify() { python cluster/notify.py send --subject "$1" --tail "${2:-40}" || true; }
PID=""
on_timeout() {
  notify "[frmedqa] $SLURM_JOB_NAME: time limit near, requeued (resumes from saved progress)" 30
  [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap on_timeout USR1

notify "[frmedqa] START $SLURM_JOB_NAME on $(hostname) (restart #${SLURM_RESTART_COUNT:-0})" 5
python cluster/notify.py heartbeat --job "$SLURM_JOB_NAME" &
HB=$!

rc=0
IFS=',' read -ra LIST <<< "$NBS"
for nb in "${LIST[@]}"; do
  name=$(basename "$nb" .ipynb)
  echo "===== $(date '+%F %T') NOTEBOOK $name (base=$FRMEDQA_BASE seed=${FRMEDQA_SEED:-none} profile=$FRMEDQA_PROFILE) ====="
  papermill "$nb" "$EXEC_DIR/${name}_${SLURM_JOB_ID}.ipynb" -k "${KERNEL:-frmedqa-v2}" \
            --log-output --cwd "$PROJECT_DIR/notebooks" &
  PID=$!
  wait "$PID"; rc=$?
  PID=""
  [ $rc -ne 0 ] && break
  notify "[frmedqa] $SLURM_JOB_NAME: finished $name" 15
done
kill "$HB" 2>/dev/null

# after each seed's statistics: refresh the across-seed summary
if [ $rc -eq 0 ] && [[ "$NBS" == *NB10* ]]; then python cluster/aggregate_seeds.py || true; fi

if [ $rc -eq 0 ]; then notify "[frmedqa] DONE $SLURM_JOB_NAME" 30
else notify "[frmedqa] FAILED $SLURM_JOB_NAME (exit $rc) — see log" 80; fi
exit $rc
