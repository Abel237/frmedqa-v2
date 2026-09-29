#!/bin/bash
# One-time setup of the conda env used by every job.
# Normally submitted as a batch job on a GPU node (so the CUDA toolkit matches the driver):
#   mkdir -p logs && sbatch cluster/setup_env.sbatch
set -eo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; source .env; set +a; }
source "${CONDA_SH:-/mnt/tank/scratch/dbrice/miniconda3/etc/profile.d/conda.sh}"
ENV="${CONDA_ENV:-frmedqa-v2}"
conda env list | grep -qE "^${ENV}[[:space:]]" || conda create -y -n "$ENV" python=3.11
conda activate "$ENV"

# CUDA toolkit matching the driver: only used to compile the llama.cpp judge server
CV=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "12.4")
echo "driver supports CUDA $CV"
conda install -y -c "nvidia/label/cuda-${CV}.0" cuda-toolkit \
  || conda install -y -c nvidia cuda-toolkit \
  || echo "WARN: cuda-toolkit not installed via conda; the judge build will try 'module load cuda'"
conda install -y -c conda-forge cmake git

pip install --upgrade pip
pip install unsloth                       # pulls torch, transformers, peft, trl, bitsandbytes
pip install datasets rouge-score nltk bert-score huggingface-hub tqdm scikit-learn sacrebleu \
            modelscope datasketch papermill ipykernel nbconvert statsmodels seaborn \
            scikit-posthocs matplotlib pandas scipy psutil requests openai
python -m ipykernel install --user --name "${KERNEL:-frmedqa-v2}" --display-name "Python (frmedqa-v2)"
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo "──────── check ────────"
python - << 'PY'
import importlib, torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU visible")
for m in ["unsloth", "transformers", "peft", "trl", "bitsandbytes", "datasets", "papermill", "datasketch", "statsmodels"]:
    try:
        mod = importlib.import_module(m); print(f"  ok    {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  FAIL  {m}: {e}")
PY
(command -v nvcc >/dev/null && nvcc --version | tail -1) || echo "nvcc not found (only needed for the judges): try 'module load cuda'"
cmake --version | head -1
echo "✓ env '$ENV' ready"
