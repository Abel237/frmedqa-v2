#!/bin/bash
# Quick overview: queued/running jobs, today's finished jobs, latest log lines.
cd "$(dirname "$0")/.."
echo "── queue ─────────────────────────────────────────────"
squeue -u "$USER" -o "%.9i %.26j %.9T %.11M %.11l %.24R"
echo; echo "── finished since yesterday ──────────────────────────"
sacct -u "$USER" -S "$(date -d yesterday +%F)" -X -o JobID%9,JobName%26,State%12,Elapsed%11,ExitCode 2>/dev/null | grep -E "fm-|JobID" || true
echo; echo "── latest log lines ──────────────────────────────────"
for f in $(ls -t logs/*.out 2>/dev/null | head -2); do
  echo "== $f"; tail -c 4000 "$f" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 6
done
