#!/bin/bash
# Create runs/seed<S>: a run folder that REUSES the shared, seed-independent
# artefacts (data splits, merged DAPT models, HF cache, llama.cpp build,
# vanilla evaluations) and keeps everything seed-specific separate.
set -euo pipefail
S="$1"; P="$PROJECT_DIR"; R="$P/runs/seed$S"
mkdir -p "$R"/{RESULT,EVALS/results_cache,MODEL_TRAINING,analysis_output/figures} "$P/hf_cache" "$P/tools"
for f in enmed_core.py enmed_recipe.py local_judges.py PRECISION.txt reverso_medical_fr.txt; do
  [ -e "$P/$f" ] && ln -sfn "$P/$f" "$R/$f"
done
ln -sfn "$P/hf_cache" "$R/hf_cache"; ln -sfn "$P/tools" "$R/tools"
# NB1 splits are COPIED (small), so per-seed outputs can never overwrite shared files
for f in train.jsonl val.jsonl test.jsonl exemplars.jsonl dapt_corpus_sample.jsonl data_stats.json leakage_report.json; do
  if [ -e "$P/RESULT/$f" ] && [ ! -e "$R/RESULT/$f" ]; then cp "$P/RESULT/$f" "$R/RESULT/"; fi
done
# merged DAPT models (read-only, ~28 GB each): linked, not copied
for d in "$P"/MODEL_TRAINING/*-merged-bf16; do
  [ -d "$d" ] && ln -sfn "$d" "$R/MODEL_TRAINING/$(basename "$d")"
done
# vanilla evaluations are seed-independent (run once by NB5a): copied in
for f in "$P"/EVALS/results_cache/Qwen3-14B-vanilla__*.json; do
  [ -e "$f" ] && [ ! -e "$R/EVALS/results_cache/$(basename "$f")" ] && cp "$f" "$R/EVALS/results_cache/"
done
true
echo "prepared $R"
