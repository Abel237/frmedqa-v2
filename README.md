# FrMedQA-CrossLingual v2 (precision-correct)

Three-way comparison: Qwen3-14B-vanilla vs Perplexity pipeline vs NoPPL pipeline.
Drive folder: `MyDrive/00_PHD_THESIS/frmedqa-colab-v2/`

## Files
| File | Role |
|---|---|
| `enmed_core.py` | config (precision switch), loading, merging, utilities |
| `enmed_recipe.py` | CPT data recipe (perplexity scoring, DSIR, curriculum) |
| `local_judges.py` | the four Unsloth judge models and their backends |
| `PRECISION.txt` | `qlora` or `bf16`, read by every notebook |
| `notebooks/` | NB1 to NB10 |

The notebooks only use `enmed_core.py` functions that existed before the last
fixes, so replacing `enmed_core.py` never forces you to re-run finished notebooks.

## Precision
| Mode | Frozen base | LoRA | GPU |
|---|---|---|---|
| qlora | `unsloth/Qwen3-14B` quantized on load to 4-bit NF4 | bf16 | L4 22 GB / A100 40 GB |
| bf16 | `unsloth/Qwen3-14B` in bf16 | bf16 | A100 80 GB / H100 |

- The base is always the 16-bit checkpoint, never a pre-quantized one.
- NB2 Stage 16 merges the DAPT LoRA into the bf16 weights on CPU (High-RAM runtime)
  and saves it to `MODEL_TRAINING/qwen3-14b-cpt[-noppl]-v2-<precision>-merged-bf16`.
- NB3 and NB5b load that Drive folder (Hub copy only as fallback). Uploading to the
  Hub is optional: set `PUSH_MERGED_TO_HUB = True` in NB2 Stage 16 (resumable upload).
- Each task adapter is trained on a freshly loaded DAPT base (no leakage between tasks)
  and evaluated unmerged, at the same precision as training.

## Models (all from Unsloth, all local)
- Reference: `unsloth/Qwen3-14B` (NB5a)
- Pointwise judges: `unsloth/gemma-4-12B-it-qat-GGUF`, `unsloth/gpt-oss-20b-GGUF` (F16)
- Pairwise judges: the same two models, `unsloth/gemma-4-12B-it-qat-GGUF` (UD-Q4_K_XL) and `unsloth/gpt-oss-20b-GGUF` (F16).
- GGUF judges run through llama.cpp `llama-server`, built once (~10 min) and cached in `tools/`.

## Run order
1. NB1_DataEngineering
2. NB2_CPT_Perplexity, NB2_CPT_NoPPL (end with the bf16 merge)
3. NB3_SFT_Perplexity, NB3_SFT_NoPPL (check Stage 3b logs "Drive folder")
4. NB5a_VanillaBaselines, NB5b_PerplexityEval, NB5b_NoPPLEval
5. NB6_GemmaJudge
6. NB7_DeepAnalysis (clears figures once), NB8_DeepStats, NB9_MultiJudge, NB10_StatAnalysis

## Safe re-runs
- Output folders end in `-v2-<precision>`, so old checkpoints are never resumed.
- Old eval caches and judge files are deleted once; a marker file in `EVALS/` prevents
  deleting new results on later runs.
- NB2 Stage 16 skips the merge when the merged folder exists, and skips the upload when
  the model is already on the Hub.

## Cluster (SLURM) run
See `CLUSTER_GUIDE.md`: full-scale settings, all seeds, `sbatch` job chain, email updates.
The same notebooks still run on Colab unchanged (cluster behaviour is switched on by
environment variables set in `cluster/`).
