# Running FrMedQA v2 on the ITMO AI cluster (RTX A6000, 48 GB)

Everything runs as `sbatch` jobs: a job keeps running after you log out, frees the
GPU when it finishes, requeues itself 15 min before the 5-day limit, and resumes
from what it already saved. You get email at every step.

## What runs, in what order
| Job | Notebook(s) | Runs in | Needs |
|---|---|---|---|
| `nb1` | NB1 data splits (full datasets) | project folder | – |
| `nb2-ppl`, `nb2-noppl` | NB2 CPT, full scale, then bf16 merge | project folder | nb1 |
| `nb5a` | NB5a, local `unsloth/Qwen3-14B` | project folder | nb1 |
| `nb3-ppl-sS`, `nb3-noppl-sS` | NB3 SFT, seed S | `runs/seedS/` | nb2 of that pipeline |
| `nb5b-ppl-sS`, `nb5b-noppl-sS` | NB5b evaluation, seed S | `runs/seedS/` | nb3 of that seed + nb5a |
| `analysis-sS` | NB6 → NB9 → NB7 → NB8 → NB10, seed S | `runs/seedS/` | both nb5b of that seed |

Seeds default to 42, 43, 44 (`SEEDS` in `.env`). CPT runs once; each seed repeats SFT,
evaluation and analysis. After each seed's analysis, `runs/all_seeds/` is refreshed
with mean ± SD across the seeds finished so far.

**Full-scale settings (`FRMEDQA_PROFILE=full`):** sequence length 2048, gradient
accumulation 8, CPT budgets 250M / 80M / 40M tokens, and perplexity scoring over the
**whole** corpus. The budget is kept on purpose: both pipelines see the same number
of tokens and differ only in which documents fill it. **Precision:** `PRECISION.txt`
is set to `bf16` (fits the A6000 and avoids re-quantization); write `qlora` to switch.

---

## A. Push the code (on your laptop)
1. Unzip `frmedqa-v2-cluster.zip` and open a terminal in the `frmedqa-v2` folder.
2. (Optional, keeps the repo small) clear notebook outputs:
   `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
3. On GitHub create a **private** repository named `frmedqa-v2` (no README).
4. Push:
   ```bash
   git init
   git add -A
   git commit -m "FrMedQA v2: cluster pipeline"
   git branch -M main
   git remote add origin git@github.com:<your-github-user>/frmedqa-v2.git
   git push -u origin main
   ```
   `.gitignore` already keeps `.env`, outputs, caches and models out of git.

## B. First-time setup on the cluster
1. Log in: `ssh dbrice@77.234.216.102`
2. Give the cluster access to your private repo (once):
   ```bash
   ssh-keygen -t ed25519 -C "cluster"      # press Enter 3 times
   cat ~/.ssh/id_ed25519.pub               # copy → GitHub › Settings › SSH keys › New key
   ```
3. Clone next to your old project (the old `frmedqa-crosslingual` stays untouched):
   ```bash
   cd /mnt/tank/scratch/dbrice
   git clone git@github.com:<your-github-user>/frmedqa-v2.git
   cd frmedqa-v2
   ```
4. Secrets and settings: `cp cluster/env.example .env && nano .env`
   Fill in `HF_TOKEN`, `MAIL_TO` and the SMTP lines (a Gmail "App password" works).
5. Build the environment as a batch job (about 20–40 min, once):
   ```bash
   mkdir -p logs
   sbatch --mail-user=413374@edu.itmo.ru --mail-type=END,FAIL cluster/setup_env.sbatch
   ```
   When it finishes, check the end of `logs/fm-setup-env_<jobid>.out`: it must show
   `CUDA: True` and `ok` for every package.
6. Test email from a compute node (where the real jobs run), also as a batch job:
   ```bash
   sbatch cluster/test_mail.sbatch
   ```
   After a minute: read `logs/fm-test-mail_<jobid>.out` and check your inbox (and spam).

## C. Launch
- Whole pipeline, all seeds:
  ```bash
  cd /mnt/tank/scratch/dbrice/frmedqa-v2
  bash cluster/submit_pipeline.sh
  ```
  It prints one line per job with its ID and dependencies (saved in `logs/pipeline_*.txt`).
- Safer first run: submit only NB1, check the result, then the rest:
  ```bash
  bash cluster/submit_one.sh notebooks/NB1_DataEngineering.ipynb shared
  # when NB1 is done:
  SKIP="nb1" bash cluster/submit_pipeline.sh
  ```

## D. Watch it
- Overview: `bash cluster/status.sh`
- Live log: `tail -f logs/fm-nb2-ppl_<jobid>.out`
- Executed notebooks with all outputs: `executed/` and `runs/seedS/executed/`
- **Emails you receive:** SLURM start / end / fail / requeue / 80%-of-time-limit;
  START and DONE / FAILED with the last log lines; one mail per finished notebook;
  a progress mail every `HEARTBEAT_HOURS` (notebook, cell, GPU use, last lines);
  a warning if a log stops changing for `STALL_HOURS`.
- Cancel one job: `scancel <jobid>` · cancel all: `scancel -u $USER`

## E. Change code later
Edit on your laptop → `git commit -am "..." && git push` → on the cluster `git pull`.
Running jobs are not affected; new jobs use the new code. Re-run one stage:
`bash cluster/submit_one.sh notebooks/NB3_SFT_NoPPL.ipynb seed 43`, or resubmit the
pipeline skipping finished stages, e.g. `SKIP="nb1 nb2-ppl nb2-noppl nb5a" bash cluster/submit_pipeline.sh`.

## F. Results
- Per seed: `runs/seedS/analysis_output/` (CSVs, figures), `runs/seedS/EVALS/`
- Across seeds: `runs/all_seeds/summary_mean_sd.csv`, `judge_summary_mean_sd.csv`,
  `ppl_vs_noppl_by_seed.csv`, and every analysis CSV stacked as `*__all_seeds.csv`
- Download (on your laptop):
  `rsync -av dbrice@77.234.216.102:/mnt/tank/scratch/dbrice/frmedqa-v2/runs/all_seeds/ ./results/`

## G. If something goes wrong
| Symptom | Fix |
|---|---|
| No emails | Check spam; run `python cluster/notify.py test` on a compute node; some clusters block port 587, so try `SMTP_PORT=465` |
| Job stays `PENDING` | `squeue` shows the reason: `Resources`/`Priority` = waiting for a GPU, `Dependency` = waiting for an earlier job |
| `DependencyNeverSatisfied` / cancelled | An earlier job failed: read its FAILED mail, fix, then resubmit with `SKIP=` for finished stages |
| CUDA out of memory in NB2/NB3 | write `qlora` into `PRECISION.txt` (needs about 20 GB), then resubmit that stage |
| Requeue refused by the cluster | Resubmit the stage; training resumes from its checkpoints |
| Judge server fails to build | the FAILED mail shows cmake's error; usually `module load cuda` or the conda `cuda-toolkit` is missing |
| `401` from Hugging Face | `HF_TOKEN` in `.env` is missing or lacks access to the gated model |

**Disk:** expect roughly 150–250 GB in the project folder (base model, two merged
28 GB models, judges, datasets). Check your scratch quota first.

**Every step above uses `sbatch`.** Interactive debugging is the only exception, and optional:
start `srun ... --pty bash`, activate
`frmedqa-v2`, export `FRMEDQA_BASE=/mnt/tank/scratch/dbrice/frmedqa-v2` (and
`FRMEDQA_PROFILE=full`), then start Jupyter and open the notebooks.
