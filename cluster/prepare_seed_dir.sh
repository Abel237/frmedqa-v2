#!/bin/bash
# Builds runs/seed<S>: links to shared code / models / caches, copies of the
# small data files. Links come FIRST (they cost no disk), so even with a full
# quota the merged DAPT models are visible to NB3/NB5b.
set -uo pipefail
S="$1"; P="$PROJECT_DIR"; R="$P/runs/seed$S"
err=0
mkdir -p "$R"/{RESULT,EVALS/results_cache,MODEL_TRAINING,analysis_output/figures} "$P/hf_cache" "$P/tools" || err=1

# 1) links (no disk use)
for d in "$P"/MODEL_TRAINING/*-merged-bf16; do
  [ -d "$d" ] && { ln -sfn "$d" "$R/MODEL_TRAINING/$(basename "$d")" || err=1; }
done
for f in enmed_core.py enmed_recipe.py local_judges.py PRECISION.txt reverso_medical_fr.txt; do
  [ -e "$P/$f" ] && { ln -sfn "$P/$f" "$R/$f" || err=1; }
done
ln -sfn "$P/hf_cache" "$R/hf_cache" || err=1
ln -sfn "$P/tools" "$R/tools" || err=1

# 2) small copies
for f in train.jsonl val.jsonl test.jsonl exemplars.jsonl dapt_corpus_sample.jsonl data_stats.json leakage_report.json; do
  if [ -e "$P/RESULT/$f" ] && [ ! -s "$R/RESULT/$f" ]; then cp "$P/RESULT/$f" "$R/RESULT/" || err=1; fi
done
for f in "$P"/EVALS/results_cache/*.json; do
  [ -e "$f" ] || continue
  t="$R/EVALS/results_cache/$(basename "$f")"
  [ -s "$t" ] || cp "$f" "$t" || err=1
done

if [ $err -ne 0 ]; then
  echo "prepare_seed_dir: FAILED for seed $S (disk quota? run: bash cluster/disk_report.sh)" >&2
  exit 1
fi
echo "prepared $R"
