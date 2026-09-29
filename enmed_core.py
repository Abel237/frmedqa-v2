"""
enmed_core.py
=============
Shared utilities for the EnToFrMedicaLLM 6-notebook pipeline.

This module is the single source of truth for:
  - Config dataclass
  - HF/Colab secret bootstrap
  - Logging
  - Persistence helpers (Colab keep-alive, checkpoint resume, atomic JSON)
  - Memory cleanup
  - Qwen3-thinking prompt builders for MCQA / ExtQA / AbsQA
  - Few-shot exemplar selection (random / stratified / similarity)
  - Metrics (ROUGE-L, BLEU-4, BERTScore, EM, token-F1, MCQA Acc/F1/Hamming)
  - Reproducibility seeding

Drop this file in `{base_dir}/MODEL_TRAINING/enmed_core.py` and `import enmed_core as ec`
from every notebook. Every utility is type-annotated and side-effect-free unless documented.

Author: Boods (PhD, French Medical NLP)
"""
from __future__ import annotations

import os
import re
import sys
import json
import gc
import math
import time
import random
import hashlib
import logging
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

QWEN3_CHAT_TEMPLATE = "qwen3"

# Per-task instruction prompts (French; identical across all generative LLMs
# so score differences reflect model capability, not prompt advantage).
#
# Design principles:
#   1. SYSTEM prompt is a hard French-only directive — it forbids the English
#      reasoning leak we saw in the first eval pass ("Okay, let's tackle this…").
#   2. INSTR_*_FR are extremely strict and end with a formatted "Réponse:"
#      anchor so the model emits only the expected payload, nothing else.
#   3. We use the plain `qwen3` chat template (NOT `qwen3-thinking`) so the
#      model emits direct answers without wrapping them in <think>...</think>.
SYSTEM_PROMPT_MEDICAL_FR = (
    "Tu es un assistant médical expert francophone. RÈGLES STRICTES:\n"
    "1. Tu réponds EXCLUSIVEMENT en français. Jamais en anglais.\n"
    "2. Tu suis EXACTEMENT le format de réponse demandé.\n"
    "3. Tu n'ajoutes aucune explication, préambule, ou commentaire non demandé.\n"
    "4. Si on te demande des lettres, tu donnes uniquement des lettres.\n"
    "5. Si on te demande un passage du texte, tu copies ce passage sans le reformuler."
)

INSTR_MCQA_FR = (
    "TÂCHE: Question à choix multiples. Choisis la ou les bonnes réponses.\n"
    "FORMAT DE RÉPONSE (OBLIGATOIRE):\n"
    "- Une seule lettre majuscule (ex: B), OU plusieurs lettres séparées par "
    "des virgules sans espaces (ex: A,C,E).\n"
    "- AUCUN autre caractère, mot, phrase, ou explication.\n"
    "- AUCUN raisonnement écrit. Pense en silence.\n"
    "- Réponds uniquement par les lettres."
)

INSTR_EXTQA_FR = (
    "TÂCHE: Extraction d'un passage. Trouve dans le texte ci-dessous le "
    "passage le plus court qui répond à la question.\n"
    "FORMAT DE RÉPONSE (OBLIGATOIRE):\n"
    "- Recopie EXACTEMENT le passage du texte, sans le reformuler.\n"
    "- Le passage doit être TRÈS COURT et CIBLÉ (idéalement 1 à 10 mots).\n"
    "- AUCUN préambule, AUCUN guillemets, AUCUN explication.\n"
    "- Si le texte ne contient pas la réponse, réponds par un seul mot pertinent du texte."
)

INSTR_ABSQA_FR = (
    "TÂCHE: Question médicale ouverte. Réponds en français.\n"
    "FORMAT DE RÉPONSE:\n"
    "- Réponse claire, concise, médicalement exacte, EN FRANÇAIS UNIQUEMENT.\n"
    "- Maximum 3 à 5 phrases.\n"
    "- Pas de raisonnement étape-par-étape — donne directement la réponse."
)

# Default seeds for variance estimation
DEFAULT_SEEDS = [42, 43, 44]

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnMedConfig:
    """Single config for all 6 notebooks. Override fields per-notebook."""

    # ─── Identity ────────────────────────────────────────────────────────
    # ┌──────────────────────────────────────────────────────────────────┐
    # │ HF REPO NAME — single source of truth.  Every Hub repo in the     │
    # │ pipeline is built as  {hf_user}/{project_name}-<suffix> :          │
    # │   -data  (NB1)   -DAPT (NB2)   -MCQA/-ExtQA/-AbsQA/-Unified (NB3)  │
    # │   -GGUF  (NB4).   Change project_name here and ALL of them rename. │
    # └──────────────────────────────────────────────────────────────────┘
    project_name: str = field(default_factory=lambda: os.environ.get(
        "FRMEDQA_PROJECT_NAME", "FrMedQA-CrossLingual"))  # repo identity (also via .env)
    hf_user: str = field(default_factory=lambda: os.environ.get(
        "FRMEDQA_HF_USER", "boods"))
    seed: int = 42

    # ─── Filesystem ──────────────────────────────────────────────────────
    # Shared dated project folder. All notebooks pin BASE_DIR to this same
    # value; keep them consistent or downstream notebooks won't find artifacts.
    base_dir: str = field(default_factory=lambda: os.environ.get(
        "FRMEDQA_BASE", os.path.expanduser("~/frmedqa_experiments")))
    # Leave empty to auto-derive to Drive; set explicitly to override.
    cache_dir: str = ""

    # Computed sub-folders (filled by `__post_init__`)
    training_dir: str = ""
    evals_dir: str = ""
    results_dir: str = ""

    # ─── HF Hub repo names (filled by __post_init__) ─────────────────────
    hf_repo_dapt: str = ""           # CPT/DAPT result
    hf_repo_unified: str = ""        # SFT multi-task result
    hf_repo_mcqa: str = ""           # SFT MCQA-only adapter
    hf_repo_gguf: str = ""           # quantized release

    # ─── Base model + precision ──────────────────────────────────────────
    # base_model MUST be the original 16-bit (bf16) checkpoint, never a
    # pre-quantized one: adapters are merged into these 16-bit weights.
    #
    # precision:
    #   "qlora" → QLoRA. Base quantized on load to 4-bit NF4 (bitsandbytes,
    #             double quant, bf16 compute), frozen; LoRA weights in bf16.
    #             Fits a 22 GB L4 / 40 GB A100.
    #   "bf16"  → 16-bit LoRA. Base in bf16, frozen; LoRA weights in bf16.
    #             Needs >= 40 GB VRAM (A100 80 GB / H100 recommended).
    # Set via env FRMEDQA_PRECISION before building the config.
    base_model: str = "unsloth/Qwen3-14B"
    precision: str = field(default_factory=lambda: os.environ.get(
        "FRMEDQA_PRECISION", "qlora").strip().lower())
    max_seq_length: int = 2048
    load_in_4bit: bool = True          # derived from `precision` in __post_init__
    hf_repo_dapt_merged: str = ""      # DAPT LoRA merged into the bf16 base

    # ─── Datasets ────────────────────────────────────────────────────────
    parcomed_id: str = "HealthDataHub/PARCOMED"  # primary FR medical CPT corpus
    mcqa_test_id: str = "qanastek/FrenchMedMCQA"
    absqa_test_id: str = "DrBenchmark/MEDIQA-l"  # may need fallback
    extqa_source_ids: List[str] = field(default_factory=lambda: [
        "DrBenchmark/CAS",                  # primary
        "Dr-BERT/CAS",                      # mirror
    ])

    # ─── DAPT (NB2) ──────────────────────────────────────────────────────
    cpt_lr: float = 2e-5
    cpt_epochs: int = 1
    cpt_batch_size: int = 2
    cpt_grad_accum: int = 8
    cpt_warmup_ratio: float = 0.02
    cpt_lora_r: int = 32
    cpt_lora_alpha: int = 64

    # ─── SFT (NB3) ───────────────────────────────────────────────────────
    sft_lr: float = 2e-4
    sft_lora_r: int = 32
    sft_lora_alpha: int = 64
    sft_lora_dropout: float = 0.05
    sft_lora_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    sft_epochs_mcqa: int = 3
    sft_epochs_extqa: int = 3
    sft_epochs_absqa: int = 2
    sft_epochs_unified: int = 2
    sft_batch_size: int = 2
    sft_grad_accum: int = 8
    sft_seeds: List[int] = field(default_factory=lambda: [42])

    # ─── Leakage shield (NB1) ────────────────────────────────────────────
    minhash_threshold: float = 0.8     # Jaccard above this = duplicate
    minhash_num_perm: int = 128
    exemplar_pool_size_per_task: int = 200   # held out for 3-shot/5-shot

    # ─── Eval (NB5) ──────────────────────────────────────────────────────
    eval_n_shots: List[int] = field(default_factory=lambda: [0, 3, 5])
    eval_max_new_tokens_mcqa: int = 16
    eval_max_new_tokens_extqa: int = 256
    eval_max_new_tokens_absqa: int = 512
    eval_temperature: float = 0.0          # deterministic eval
    eval_subset_size: Optional[int] = None  # None = all, int = cap for debug

    # ─── Misc ────────────────────────────────────────────────────────────
    auto_resume: bool = True
    save_every_n_steps_frac: float = 0.10
    keep_n_checkpoints: int = 3
    use_wandb: bool = False
    wandb_project: str = "EnToFrMedicaLLM"

    def __post_init__(self):
        # Derive sub-folders
        self.training_dir = os.path.join(self.base_dir, "MODEL_TRAINING")
        self.evals_dir    = os.path.join(self.base_dir, "EVALS")
        self.results_dir  = os.path.join(self.base_dir, "RESULT")
        # Pin cache to Drive so weights survive Colab disconnects.
        # Paths under /content are ephemeral — wiped on every Colab restart.
        if not self.cache_dir:
            self.cache_dir = os.path.join(self.base_dir, "hf_cache")

        # Precision → quantization flag. One switch drives training AND eval,
        # so every model in the comparison runs at the same precision.
        if self.precision not in ("qlora", "bf16"):
            raise ValueError(f"precision must be 'qlora' or 'bf16', got {self.precision!r}")
        self.load_in_4bit = (self.precision == "qlora")
        # The precision is always part of the project name, so QLoRA and bf16
        # artefacts live in separate HF repos and never overwrite each other:
        #   boods/FrMedQA-CrossLingual-v2-PPL-qlora-DAPT, ...-v2-NoPPL-bf16-MCQA, etc.
        _suffix = f"-{self.precision}"
        if not self.project_name.endswith(_suffix):
            self.project_name = f"{self.project_name}{_suffix}"

        # Derive HF repo names if not set
        if not self.hf_repo_dapt_merged:
            self.hf_repo_dapt_merged = f"{self.hf_user}/{self.project_name}-DAPT-merged-bf16"
        if not self.hf_repo_dapt:
            self.hf_repo_dapt = f"{self.hf_user}/{self.project_name}-DAPT"
        if not self.hf_repo_unified:
            self.hf_repo_unified = f"{self.hf_user}/{self.project_name}-Unified"
        if not self.hf_repo_mcqa:
            self.hf_repo_mcqa = f"{self.hf_user}/{self.project_name}-MCQA"
        if not self.hf_repo_gguf:
            self.hf_repo_gguf = f"{self.hf_user}/{self.project_name}-GGUF"

    def make_dirs(self):
        """Create all output directories."""
        for d in [self.base_dir, self.training_dir, self.evals_dir,
                  self.results_dir, self.cache_dir]:
            os.makedirs(d, exist_ok=True)

    def save(self, path: Optional[str] = None):
        """Persist config as JSON for reproducibility."""
        path = path or os.path.join(self.training_dir, "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # asdict is enough since all fields are JSON-serialisable
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(name: str = "enmed", level: int = logging.INFO,
                 log_file: Optional[str] = None) -> logging.Logger:
    """Configure a single logger with optional file sink."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:  # already configured
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname).1s | %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# 3. SECRETS (Colab + .env fallback)
# ─────────────────────────────────────────────────────────────────────────────

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Try Colab userdata, then env vars."""
    # Colab path
    try:
        from google.colab import userdata  # noqa: WPS433
        try:
            v = userdata.get(key)
            if v:
                return v
        except Exception:
            pass
    except ImportError:
        pass
    # Env fallback
    return os.environ.get(key, default)


def bootstrap_hf(logger: Optional[logging.Logger] = None) -> Optional[str]:
    """Load HF token, login, return token (or None)."""
    tok = get_secret("HF_TOKEN") or get_secret("HUGGINGFACE_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGINGFACE_TOKEN"] = tok
        try:
            from huggingface_hub import login
            login(token=tok, add_to_git_credential=False)
            if logger:
                logger.info("✓ HF authenticated")
        except Exception as e:
            if logger:
                logger.warning(f"HF login failed (non-fatal): {e}")
    elif logger:
        logger.warning("⚠ HF_TOKEN not found — uploads will fail")
    return tok


def bootstrap_wandb(project: str = "EnToFrMedicaLLM",
                    logger: Optional[logging.Logger] = None) -> bool:
    """Login to W&B if WANDB_API_KEY is available. Returns True on success."""
    tok = get_secret("WANDB_API_KEY")
    if not tok:
        if logger:
            logger.info("• WANDB_API_KEY absent — skipping W&B")
        return False
    try:
        import wandb
        wandb.login(key=tok, relogin=False)
        os.environ["WANDB_PROJECT"] = project
        if logger:
            logger.info(f"✓ W&B authenticated (project={project})")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"W&B login failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. PERSISTENCE — KEEP-ALIVE, RESUME, ATOMIC JSON
# ─────────────────────────────────────────────────────────────────────────────

def colab_keep_alive():
    """Inject JS that clicks reconnect every 60s. No-op outside Colab."""
    import sys as _sys
    if "google.colab" not in _sys.modules:
        return False  # SLURM/cluster: job runs to walltime, no keep-alive needed
    try:
        from IPython.display import display, Javascript
        display(Javascript("""
            function ColabReconnect(){
                const btn = document.querySelector('colab-connect-button');
                if (btn) { btn.click(); }
            }
            if (window._kaInterval) { clearInterval(window._kaInterval); }
            window._kaInterval = setInterval(ColabReconnect, 60000);
            console.log('[keep-alive] activated');
        """))
        return True
    except Exception:
        return False


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the path to the most recent `checkpoint-*` subdir, or None."""
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-", 1)[1])
                candidates.append((step, os.path.join(output_dir, name)))
            except ValueError:
                continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def resumable_train(trainer, output_dir: str, auto_resume: bool = True,
                    logger: Optional[logging.Logger] = None):
    """Wrap trainer.train() with resume-from-checkpoint."""
    ckpt = find_latest_checkpoint(output_dir) if auto_resume else None
    if ckpt:
        if logger:
            logger.info(f"▶ Resuming from {ckpt}")
        return trainer.train(resume_from_checkpoint=ckpt)
    if logger:
        logger.info("▶ Training from scratch")
    return trainer.train()


def atomic_write_json(obj: Any, path: str) -> str:
    """Write JSON atomically (write to .tmp, fsync, rename) — survives Colab kills."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)
    return path


def atomic_write_jsonl(rows: List[Dict], path: str) -> str:
    """JSONL variant — one record per line, atomic."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)
    return path


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def drive_flush():
    """Force OS sync so Drive picks up partial writes."""
    try:
        os.sync()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 5. MEMORY / CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def cleanup(*objs):
    """Aggressive memory cleanup. Pass any objects you want to del first."""
    import ctypes
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def seed_all(seed: int = 42):
    """Set every seed we can find for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 6. PROMPT BUILDERS — Qwen3-thinking, identical across models
# ─────────────────────────────────────────────────────────────────────────────

def _format_options(options: Union[List[str], Dict[str, str]]) -> str:
    """Format MCQA options as 'A) ...\nB) ...' regardless of input shape."""
    if isinstance(options, dict):
        # e.g. {"A": "...", "B": "..."}
        return "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
    # list — letter them A, B, C...
    return "\n".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options))


def build_mcqa_messages(question: str,
                        options: Union[List[str], Dict[str, str]],
                        demos: Optional[List[Dict]] = None) -> List[Dict]:
    """Build conversation for MCQA. demos = list of {question, options, answer}.

    The last user turn ends with 'Réponse:' so the model is anchored to emit
    only the answer payload (letters), nothing else.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT_MEDICAL_FR}]
    if demos:
        for d in demos:
            opt_str = _format_options(d["options"])
            messages.append({
                "role": "user",
                "content": (f"{INSTR_MCQA_FR}\n\nQuestion: {d['question']}\n\n"
                            f"{opt_str}\n\nRéponse:"),
            })
            # Demo answer is *just* the letters — teaches the format
            messages.append({"role": "assistant", "content": d["answer"]})
    opt_str = _format_options(options)
    messages.append({
        "role": "user",
        "content": (f"{INSTR_MCQA_FR}\n\nQuestion: {question}\n\n"
                    f"{opt_str}\n\nRéponse:"),
    })
    return messages


def build_extqa_messages(context: str, question: str,
                         demos: Optional[List[Dict]] = None) -> List[Dict]:
    """Build conversation for extractive QA. Same 'Réponse:' anchor."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT_MEDICAL_FR}]
    if demos:
        for d in demos:
            messages.append({
                "role": "user",
                "content": (f"{INSTR_EXTQA_FR}\n\nTexte:\n{d['context']}\n\n"
                            f"Question: {d['question']}\n\nRéponse (passage exact, court):"),
            })
            messages.append({"role": "assistant", "content": d["answer"]})
    messages.append({
        "role": "user",
        "content": (f"{INSTR_EXTQA_FR}\n\nTexte:\n{context}\n\n"
                    f"Question: {question}\n\nRéponse (passage exact, court):"),
    })
    return messages


def build_absqa_messages(question: str, context: Optional[str] = None,
                         demos: Optional[List[Dict]] = None) -> List[Dict]:
    """Build conversation for abstractive QA. context optional (open-book)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT_MEDICAL_FR}]
    if demos:
        for d in demos:
            ctx = f"\n\nContexte:\n{d['context']}" if d.get("context") else ""
            messages.append({
                "role": "user",
                "content": (f"{INSTR_ABSQA_FR}{ctx}\n\nQuestion: {d['question']}\n\n"
                            f"Réponse en français:"),
            })
            messages.append({"role": "assistant", "content": d["answer"]})
    ctx = f"\n\nContexte:\n{context}" if context else ""
    messages.append({
        "role": "user",
        "content": (f"{INSTR_ABSQA_FR}{ctx}\n\nQuestion: {question}\n\n"
                    f"Réponse en français:"),
    })
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEW-SHOT EXEMPLAR SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_demos_random(pool: List[Dict], k: int, seed: int = 42) -> List[Dict]:
    """Random k examples from pool."""
    if k <= 0 or not pool:
        return []
    rng = random.Random(seed)
    return rng.sample(pool, min(k, len(pool)))


def select_demos_stratified(pool: List[Dict], k: int,
                            stratify_key: str = "specialty",
                            seed: int = 42) -> List[Dict]:
    """Stratified by stratify_key — round-robin through groups."""
    if k <= 0 or not pool:
        return []
    rng = random.Random(seed)
    groups: Dict[str, List[Dict]] = {}
    for ex in pool:
        key = ex.get(stratify_key, "_unknown")
        groups.setdefault(key, []).append(ex)
    keys = list(groups.keys())
    rng.shuffle(keys)
    chosen: List[Dict] = []
    i = 0
    while len(chosen) < k and any(groups[g] for g in keys):
        g = keys[i % len(keys)]
        if groups[g]:
            chosen.append(groups[g].pop(rng.randrange(len(groups[g]))))
        i += 1
        if i > 10 * k:  # safety
            break
    return chosen[:k]


def select_demos_similarity(pool: List[Dict], query_text: str, k: int,
                            embedder=None, pool_embeds=None) -> List[Dict]:
    """
    Similarity-based: pick the k examples whose `question` is closest to query_text.
    Caller must provide either an `embedder` (with .encode) or precomputed
    `pool_embeds` (np.ndarray of shape [len(pool), d]).
    """
    if k <= 0 or not pool:
        return []
    try:
        import numpy as np
    except ImportError:
        return select_demos_random(pool, k)

    if pool_embeds is None:
        if embedder is None:
            return select_demos_random(pool, k)
        texts = [ex["question"] for ex in pool]
        pool_embeds = embedder.encode(texts, convert_to_numpy=True,
                                      show_progress_bar=False)

    if embedder is None:
        # Can't encode the query — fall back to random
        return select_demos_random(pool, k)
    q_emb = embedder.encode([query_text], convert_to_numpy=True,
                            show_progress_bar=False)[0]

    # Cosine similarity
    a = pool_embeds / (np.linalg.norm(pool_embeds, axis=1, keepdims=True) + 1e-9)
    b = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    sims = a @ b
    top_idx = np.argsort(-sims)[:k]
    return [pool[i] for i in top_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 8. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Standard QA normalisation: lowercase, strip articles + punct + space."""
    s = (s or "").lower().strip()
    s = re.sub(r"\b(le|la|les|un|une|des|du|de|l')\s+", " ", s)
    s = re.sub(r"[^\w\sàâäéèêëïîôöùûüÿñç]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1, French-aware via normalize_answer."""
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    n_match = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            n_match += 1
            common[t] -= 1
    if n_match == 0:
        return 0.0
    prec = n_match / len(p_toks)
    rec = n_match / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def parse_mcqa_letters(text: str, n_choices: int = 5) -> List[str]:
    """Extract chosen letter(s) from a free-text answer.

    Handles single-letter ("A"), comma lists ("A,C"), and prose
    ("La bonne réponse est B"). Returns a sorted list of unique letters
    drawn from {A, B, ..., chr(64+n_choices)}.
    """
    valid = {chr(65 + i) for i in range(n_choices)}
    if not text:
        return []
    t = text.upper()
    # First try comma-separated patterns (e.g. "A, C, E")
    comma_match = re.findall(r"\b([A-Z])\b", t)
    found = [c for c in comma_match if c in valid]
    return sorted(set(found))


def hamming_accuracy(pred_letters: List[str], gold_letters: List[str],
                     n_choices: int = 5) -> float:
    """Hamming accuracy for multi-label MCQA.

    Score = (n_choices - mismatches) / n_choices.
    """
    p_set = set(pred_letters)
    g_set = set(gold_letters)
    all_letters = {chr(65 + i) for i in range(n_choices)}
    mismatches = 0
    for c in all_letters:
        if (c in p_set) != (c in g_set):
            mismatches += 1
    return (n_choices - mismatches) / n_choices


def compute_rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1. Lazy-imports rouge_score."""
    from rouge_score import rouge_scorer
    if not hasattr(compute_rouge_l, "_scorer"):
        compute_rouge_l._scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=False)
    return compute_rouge_l._scorer.score(gold or "", pred or "")["rougeL"].fmeasure


def compute_bleu4(pred: str, gold: str) -> float:
    """Sentence BLEU-4. Returns 0..1."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    sm = SmoothingFunction().method1
    g = (gold or "").split()
    p = (pred or "").split()
    if not g or not p:
        return 0.0
    return sentence_bleu([g], p,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=sm)


def compute_bertscore_batch(preds: List[str], golds: List[str],
                            model_name: str = "almanach/camembert-bio-base"
                            ) -> Tuple[List[float], List[float], List[float]]:
    """BERTScore (P, R, F1) using a French model. Returns three lists."""
    from bert_score import score as bs_score
    P, R, F1 = bs_score(preds, golds, model_type=model_name,
                        lang="fr", verbose=False, rescale_with_baseline=False)
    return P.tolist(), R.tolist(), F1.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 9. MINHASH DEDUP — for leakage shield in NB1
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_for_hash(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def minhash_signature(text: str, num_perm: int = 128, n_gram: int = 5):
    """Compute a MinHash signature for `text`. Returns datasketch.MinHash."""
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    norm = _normalize_for_hash(text)
    tokens = norm.split()
    if len(tokens) < n_gram:
        # short text: hash whole tokens
        for t in tokens:
            m.update(t.encode("utf-8"))
        return m
    for i in range(len(tokens) - n_gram + 1):
        shingle = " ".join(tokens[i:i + n_gram])
        m.update(shingle.encode("utf-8"))
    return m


def build_lsh_index(test_texts: List[str], threshold: float = 0.8,
                    num_perm: int = 128) -> Tuple[Any, List[Any]]:
    """Index a list of test texts in MinHashLSH for fast leakage queries."""
    from datasketch import MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sigs = []
    for i, t in enumerate(test_texts):
        sig = minhash_signature(t, num_perm=num_perm)
        sigs.append(sig)
        lsh.insert(f"test_{i}", sig)
    return lsh, sigs


def is_leaked(text: str, lsh, num_perm: int = 128) -> bool:
    """Check whether `text` has any near-duplicate in the LSH test index."""
    sig = minhash_signature(text, num_perm=num_perm)
    return len(lsh.query(sig)) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 10. TIME / FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def hash_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# 11. SELF-TEST — run `python enmed_core.py` to sanity-check the module
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = EnMedConfig()
    print(f"Config OK | base_dir={cfg.base_dir}")
    print(f"  hf_repo_dapt    = {cfg.hf_repo_dapt}")
    print(f"  hf_repo_unified = {cfg.hf_repo_unified}")
    print(f"  hf_repo_mcqa    = {cfg.hf_repo_mcqa}")
    print(f"  hf_repo_gguf    = {cfg.hf_repo_gguf}")

    msgs = build_mcqa_messages(
        question="Quelle est la cause la plus fréquente d'IDM ?",
        options=["Embolie", "Athérosclérose", "Dissection", "Spasme", "Anévrisme"],
        demos=[{
            "question": "Quel est le ttt de 1ère intention dans l'HTA ?",
            "options": ["IEC", "Bétabloquants", "Diurétiques", "ICA", "ARA II"],
            "answer": "A,C",
        }],
    )
    assert msgs[0]["role"] == "system"
    assert "A) Embolie" in msgs[-1]["content"]
    print("✓ Prompt builders OK")

    em = exact_match("Le patient", "patient")
    f1 = token_f1("le patient présente une dyspnée",
                  "le patient a une dyspnée")
    assert em == 1, f"EM normalisation broken: {em}"
    assert 0.5 < f1 < 1.0, f"F1 broken: {f1}"
    print(f"✓ Metrics OK | EM={em} F1={f1:.3f}")

    letters = parse_mcqa_letters("La bonne réponse est A et C", n_choices=5)
    assert letters == ["A", "C"], letters
    ham = hamming_accuracy(["A", "C"], ["A", "B"], n_choices=5)
    assert abs(ham - 0.6) < 1e-6, ham
    print(f"✓ MCQA parsing OK | letters={letters} hamming={ham}")

    print("\n✓ enmed_core.py self-test passed.")


# ═══════════════════════════════════════════════════════════════════════════
# Precision-aware loading and LoRA merging
# ═══════════════════════════════════════════════════════════════════════════
def load_model(cfg, model_name, max_seq_length=None, logger=None):
    """Load a Qwen3 checkpoint for training or inference at cfg.precision.

    qlora → weights quantized on load to 4-bit NF4, bf16 compute (QLoRA).
    bf16  → weights kept in bf16.
    The checkpoint itself must be a 16-bit one (never pre-quantized).
    """
    import torch
    from unsloth import FastLanguageModel
    token = os.environ.get("HF_TOKEN") or None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length or cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=torch.bfloat16,
        token=token, cache_dir=cfg.cache_dir,
        device_map={"": 0},
    )
    describe_precision(model, cfg, logger=logger)
    return model, tokenizer


def describe_precision(model, cfg, logger=None):
    """Log what the loaded weights actually are, and fail loudly on mismatch."""
    n4 = n16 = 0
    for m in model.modules():
        cls = type(m).__name__
        if cls == "Linear4bit":
            n4 += 1
        elif cls == "Linear" and getattr(getattr(m, "weight", None), "dtype", None) is not None:
            import torch
            if m.weight.dtype in (torch.bfloat16, torch.float16):
                n16 += 1
    msg = (f"precision={cfg.precision} | 4-bit linear layers={n4} | "
           f"16-bit linear layers={n16}")
    (logger.info if logger else print)(msg)
    if cfg.precision == "qlora" and n4 == 0:
        raise RuntimeError("precision='qlora' but no 4-bit layers were loaded.")
    if cfg.precision == "bf16" and n4 > 0:
        raise RuntimeError("precision='bf16' but 4-bit layers were loaded.")


def check_vram_for_precision(cfg, logger=None):
    """bf16 LoRA on a 14B model needs roughly 36-40 GB; refuse to start on less."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available.")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    need = 38 if cfg.precision == "bf16" else 16
    (logger.info if logger else print)(
        f"GPU {torch.cuda.get_device_name(0)} | {total:.0f} GB | precision={cfg.precision} needs ~{need} GB")
    if total < need:
        raise RuntimeError(
            f"precision={cfg.precision!r} needs ~{need} GB VRAM, this GPU has {total:.0f} GB. "
            f"Use an A100 80 GB / H100 for bf16, or set FRMEDQA_PRECISION=qlora.")


def merge_lora_into_bf16(base_model, adapter, out_dir, push_repo=None,
                         cache_dir=None, logger=None):
    """Merge a LoRA adapter into the ORIGINAL bf16 base weights, on CPU.

    Correct for both precisions: a QLoRA adapter is merged into the 16-bit
    weights (as recommended by the QLoRA paper), never into 4-bit weights,
    which would add a second rounding error. Needs ~30 GB of CPU RAM.
    """
    import gc, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    log = logger.info if logger else print
    token = os.environ.get("HF_TOKEN") or None
    log(f"Merging {adapter} into bf16 {base_model} (CPU)…")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, token=token, cache_dir=cache_dir)
    merged = PeftModel.from_pretrained(base, adapter, token=token).merge_and_unload()
    try:   # adapter folder / repo usually carries the tokenizer; checkpoints may not
        tok = AutoTokenizer.from_pretrained(adapter, token=token, cache_dir=cache_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model, token=token, cache_dir=cache_dir)
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out_dir)
    log(f"  ✓ merged bf16 checkpoint → {out_dir}")
    del base, merged; gc.collect()
    if push_repo and token:
        push_folder_to_hub(out_dir, push_repo, logger=logger)
    return out_dir


def push_folder_to_hub(folder, repo_id, logger=None, private=True):
    """Upload a saved model folder as-is (no re-serialization, no extra RAM).
    Works across transformers versions, unlike model.push_to_hub(...)."""
    from huggingface_hub import HfApi
    log = logger.info if logger else print
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(repo_id, exist_ok=True, private=private)
    # upload_large_folder is resumable: if the runtime drops, re-running it
    # continues where it stopped instead of starting over.
    api.upload_large_folder(repo_id=repo_id, folder_path=folder, repo_type="model")
    log(f"  ✓ pushed → https://huggingface.co/{repo_id}")


def hub_has_model(repo_id) -> bool:
    """True if repo_id exists on the Hub and holds a model config."""
    try:
        from huggingface_hub import file_exists
        return file_exists(repo_id, "config.json", token=os.environ.get("HF_TOKEN") or None)
    except Exception:
        return False
    del base, merged; gc.collect()
    return out_dir



def dapt_merged_source(cfg, logger=None):
    """Location of the DAPT-merged bf16 model for NB3 / NB5b.

    Prefers the folder NB2 Stage 16 wrote on Drive (no 30 GB upload needed);
    falls back to the Hub repo cfg.hf_repo_dapt_merged if that folder is absent.
    """
    sub = "qwen3-14b-cpt-noppl" if "noppl" in cfg.project_name.lower() else "qwen3-14b-cpt"
    local = os.path.join(cfg.training_dir, f"{sub}-v2-{cfg.precision}-merged-bf16")
    log = logger.info if logger else print
    if os.path.exists(os.path.join(local, "config.json")):
        log(f"DAPT-merged model: Drive folder {local}")
        return local
    log(f"DAPT-merged model: Hub repo {cfg.hf_repo_dapt_merged} (no Drive folder at {local})")
    return cfg.hf_repo_dapt_merged
