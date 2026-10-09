#!/bin/bash
# Where is the disk going?  Usage: bash cluster/disk_report.sh
P="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
echo "== quota"; (quota -s 2>/dev/null || true); (lfs quota -h -u "$USER" /mnt/tank 2>/dev/null || true)
echo "== filesystem"; df -h "$P" | tail -1
echo "== project ($P)"; du -sh "$P" 2>/dev/null
echo "== biggest folders"
du -h --max-depth=2 "$P" 2>/dev/null | sort -rh | head -20
echo "== HF cache models"; du -sh "$P"/hf_cache/models--* "$P"/hf_cache/hub/models--* ~/.cache/huggingface/hub/models--* 2>/dev/null | sort -rh
echo "== training checkpoints (safe to delete once the final adapter is saved)"
find "$P/MODEL_TRAINING" "$P"/runs/*/MODEL_TRAINING -maxdepth 3 -type d -name 'checkpoint-*' -exec du -sh {} + 2>/dev/null | sort -rh
echo "== conda / pip caches"
du -sh "$(dirname "$P")"/miniconda3/pkgs 2>/dev/null
echo "== everything under ~/.cache (modelscope, huggingface, pip, unsloth, triton...)"
du -sh ~/.cache/* 2>/dev/null | sort -rh | head -15
echo "== top level of your scratch space"
du -sh "$(dirname "$P")"/* "$(dirname "$P")"/.[!.]* 2>/dev/null | sort -rh | head -15
