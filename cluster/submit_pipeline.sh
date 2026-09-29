#!/bin/bash
# Submit the whole pipeline as SLURM jobs chained by dependencies.
# Each job starts automatically when the jobs it needs have succeeded; if one
# fails, everything that depends on it is cancelled (and you get an email).
#
#   bash cluster/submit_pipeline.sh                # everything
#   SKIP="nb1 nb2-ppl nb2-noppl" bash cluster/submit_pipeline.sh   # reuse finished stages
set -euo pipefail
cd "$(dirname "$0")/.."; PROJECT_DIR="$(pwd)"
set -a; source .env; set +a
SEEDS="${SEEDS:-42 43 44}"; SKIP=" ${SKIP:-} "
mkdir -p logs
MAILOPT=(); [ -n "${MAIL_TO:-}" ] && MAILOPT=(--mail-user="$MAIL_TO" --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80)
RECORD="logs/pipeline_$(date +%Y%m%d_%H%M%S).txt"

join_deps() { local out=""; for d in "$@"; do [ -n "$d" ] && out="${out:+$out:}$d"; done; echo "$out"; }
sub() {  # sub NAME MEM "DEP1 DEP2" NOTEBOOKS SCOPE [SEED]
  local name=$1 mem=$2 deps; deps=$(join_deps $3); shift 3
  if [[ "$SKIP" == *" $name "* ]]; then echo "  skip  $name" >&2; echo ""; return; fi
  local dep=(); [ -n "$deps" ] && dep=(--dependency="afterok:$deps" --kill-on-invalid-dep=yes)
  local id
  id=$(sbatch --parsable --job-name="fm-$name" "${dep[@]}" --mem="$mem" "${MAILOPT[@]}" \
         --chdir="$PROJECT_DIR" --output="$PROJECT_DIR/logs/%x_%j.out" \
         --export=ALL,PROJECT_DIR="$PROJECT_DIR" cluster/run_nb.sbatch "$@")
  echo "  $id  $name  (after: ${deps:-none})" | tee -a "$RECORD" >&2
  echo "$id"
}

NB=notebooks
ANALYSIS="$NB/NB6_GemmaJudge.ipynb,$NB/NB9_MultiJudge.ipynb,$NB/NB7_DeepAnalysis.ipynb,$NB/NB8_DeepStats.ipynb,$NB/NB10_StatAnalysis.ipynb"
echo "Submitting FrMedQA pipeline | seeds: $SEEDS | profile: ${FRMEDQA_PROFILE:-full}" | tee "$RECORD" >&2

J1=$(sub  nb1       48G ""     $NB/NB1_DataEngineering.ipynb   shared)
J2P=$(sub nb2-ppl   64G "$J1"  $NB/NB2_CPT_Perplexity.ipynb    shared)
J2N=$(sub nb2-noppl 64G "$J1"  $NB/NB2_CPT_NoPPL.ipynb         shared)
J5A=$(sub nb5a      48G "$J1"  $NB/NB5a_VanillaBaselines.ipynb shared)

PREV=""
for S in $SEEDS; do
  J3P=$(sub nb3-ppl-s$S     48G "$J2P"            $NB/NB3_SFT_Perplexity.ipynb  seed $S)
  J3N=$(sub nb3-noppl-s$S   48G "$J2N"            $NB/NB3_SFT_NoPPL.ipynb       seed $S)
  J5P=$(sub nb5b-ppl-s$S    48G "$J3P $J5A"       $NB/NB5b_PerplexityEval.ipynb seed $S)
  J5N=$(sub nb5b-noppl-s$S  48G "$J3N $J5A"       $NB/NB5b_NoPPLEval.ipynb      seed $S)
  JA=$(sub  analysis-s$S    48G "$J5P $J5N $PREV" "$ANALYSIS"                   seed $S)
  PREV=$JA
done
echo "Job list saved to $RECORD. Watch with: bash cluster/status.sh" >&2
