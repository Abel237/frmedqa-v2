"""
enmed_recipe.py
===============
SOTA data-recipe utilities for the EnToFrMedicaLLM continual-pretraining stage (NB2).

Drop this file next to `enmed_core.py` in `{base_dir}/MODEL_TRAINING/` and
`import enmed_recipe as er` from NB2.

The recipe, in order:
    1. harvest_streaming      — stream full corpora, bounded by a token budget
    2. clean_text             — normalise / strip boilerplate
    3. length_gate            — drop too-short / too-long docs (token-exact)
    4. minhash_dedup          — within-corpus near-duplicate removal (LSH)
    5. build_test_lsh /
       leakage_filter         — drop any doc that overlaps a locked test row
    6. perplexity_scores      — per-doc PPL under the base model
    7. ppl_filter             — drop garbage (PPL too high) / trivia (PPL too low)
    8. dsir_weights /
       dsir_resample          — importance-resample toward the gold-QA distribution
    9. curriculum_order       — sort easy -> hard by PPL (optionally bucketed)
   10. interleave_with_replay — spread EN-bio + general replay through the corpus
   11. CurriculumSFTTrainer   — sequential sampler so curriculum order survives

Every function is pure and type-annotated unless documented otherwise.

Author: Boods (PhD, French Medical NLP)
"""
from __future__ import annotations

import re
import gc
import math
import random
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Heavy deps imported lazily inside functions where possible so this module
# can be imported in environments that lack torch/datasketch.

_LOG = logging.getLogger("enmed_recipe")


# ─────────────────────────────────────────────────────────────────────────────
# 1. HARVEST — stream full corpora, bounded by an (approximate) token budget
# ─────────────────────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) to bound streaming without tokenising."""
    return max(1, len(text) // 4)


def take_longest_string_field(ex: Dict[str, Any], min_chars: int = 80) -> Optional[str]:
    """Pick the longest string field of a row — robust to heterogeneous schemas."""
    cands = [(k, v) for k, v in ex.items() if isinstance(v, str) and len(v) > min_chars]
    if not cands:
        # fall back to joining a token list if present (QUAERO / CAS style)
        for k, v in ex.items():
            if isinstance(v, list) and v and isinstance(v[0], str):
                joined = " ".join(v)
                if len(joined) > min_chars:
                    return joined
        return None
    cands.sort(key=lambda kv: -len(kv[1]))
    return cands[0][1]


def harvest_streaming(
    sources: Sequence[Dict[str, Any]],
    *,
    token_budget: Optional[int],
    max_doc_chars: int = 6000,
    extract_fn: Callable[[Dict[str, Any]], Optional[str]] = take_longest_string_field,
    logger: Optional[logging.Logger] = None,
    progress: bool = True,
) -> List[str]:
    """Stream documents from a list of HF dataset specs until `token_budget` is hit.

    Each entry of `sources` is a dict accepted by `datasets.load_dataset`, e.g.::

        {"path": "HealthDataHub/PARCOMED", "split": "train", "streaming": True}
        {"path": "Dr-BERT/QUAERO", "name": "medline", "trust_remote_code": True}

    `token_budget=None` means "take everything" (truly the whole corpus).
    Returns a flat list of raw document strings (truncated to `max_doc_chars`).
    """
    from datasets import load_dataset
    log = logger or _LOG
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731

    out: List[str] = []
    used_tokens = 0

    for spec in sources:
        if token_budget is not None and used_tokens >= token_budget:
            break
        spec = dict(spec)
        # default to streaming so we never materialise a giant dataset in RAM
        spec.setdefault("streaming", True)
        label = spec.get("name") or spec.get("path", "?")
        try:
            ds = load_dataset(**spec)
        except Exception as e:  # missing / gated / renamed — skip, don't crash the run
            log.warning(f"  [harvest] skip {label}: {e}")
            continue

        n_before = len(out)
        it = tqdm(ds, desc=f"  harvest:{label}", leave=False) if progress else ds
        for ex in it:
            if token_budget is not None and used_tokens >= token_budget:
                break
            t = extract_fn(ex)
            if not t:
                continue
            t = t[:max_doc_chars]
            out.append(t)
            used_tokens += _approx_tokens(t)
        log.info(f"  [harvest] {label}: +{len(out) - n_before:,} docs "
                 f"(cum {len(out):,} docs ≈ {used_tokens/1e6:.1f}M tok)")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN + 3. LENGTH GATE
# ─────────────────────────────────────────────────────────────────────────────

_WS = re.compile(r"[ \t\u00a0]+")
_NL = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: Optional[str]) -> str:
    """Normalise whitespace, strip control chars and obvious boilerplate."""
    if not text:
        return ""
    t = _CTRL.sub(" ", text)
    t = _WS.sub(" ", t)
    t = _NL.sub("\n\n", t)
    return t.strip()


def length_gate(
    texts: Sequence[str],
    tokenizer,
    *,
    min_tokens: int = 24,
    max_tokens: int = 2048,
    logger: Optional[logging.Logger] = None,
    num_proc: int = 1,
) -> List[str]:
    """Keep docs whose exact token length is within [min_tokens, max_tokens]."""
    log = logger or _LOG
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    kept: List[str] = []
    # batch tokenise for speed
    B = 1000
    for i in range(0, len(texts), B):
        chunk = list(texts[i:i + B])
        ids = tok(chunk, truncation=False, padding=False,
                  add_special_tokens=False)["input_ids"]
        for txt, x in zip(chunk, ids):
            if min_tokens <= len(x) <= max_tokens:
                kept.append(txt)
    log.info(f"  [length_gate] {len(kept):,}/{len(texts):,} kept "
             f"(min={min_tokens}, max={max_tokens})")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# 4. MINHASH WITHIN-CORPUS DEDUP
# ─────────────────────────────────────────────────────────────────────────────

def _shingles(text: str, k: int = 5) -> set:
    toks = text.lower().split()
    if len(toks) <= k:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _minhash(text: str, num_perm: int):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def minhash_dedup(
    texts: Sequence[str],
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Remove near-duplicate documents (Jaccard >= threshold) within one corpus."""
    from datasketch import MinHashLSH
    log = logger or _LOG
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: List[str] = []
    for i, txt in enumerate(tqdm(texts, desc="  dedup", leave=False)):
        m = _minhash(txt, num_perm)
        if lsh.query(m):          # a near-duplicate is already indexed
            continue
        lsh.insert(f"d{i}", m)
        kept.append(txt)
    log.info(f"  [minhash_dedup] {len(kept):,}/{len(texts):,} kept "
             f"(dropped {len(texts) - len(kept):,} near-dupes @ J≥{threshold})")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# 5. LEAKAGE SHIELD vs locked test sets
# ─────────────────────────────────────────────────────────────────────────────

def build_test_lsh(
    test_fingerprints: Sequence[str],
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
):
    """Index the locked test fingerprints so the CPT corpus can be screened."""
    from datasketch import MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, fp in enumerate(test_fingerprints):
        lsh.insert(f"t{i}", _minhash(fp, num_perm))
    return lsh


def leakage_filter(
    texts: Sequence[str],
    test_lsh,
    *,
    num_perm: int = 128,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Drop any CPT doc that near-matches a held-out test row."""
    log = logger or _LOG
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731
    kept = [t for t in tqdm(texts, desc="  leak-shield", leave=False)
            if not test_lsh.query(_minhash(t, num_perm))]
    log.info(f"  [leakage_filter] {len(kept):,}/{len(texts):,} kept "
             f"(removed {len(texts) - len(kept):,} test-overlapping docs)")
    return kept


def load_test_fingerprints_from_jsonl(path: str) -> List[str]:
    """Build fingerprints from NB1's locked test.jsonl (question + context + options)."""
    import json
    fps: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            parts = [str(r.get("question", "")), str(r.get("context", "") or "")]
            opts = r.get("options")
            if isinstance(opts, dict):
                parts.extend(str(v) for v in opts.values())
            fp = clean_text(" ".join(p for p in parts if p))
            if fp:
                fps.append(fp)
    return fps


# ─────────────────────────────────────────────────────────────────────────────
# 6. PERPLEXITY SCORING (per-doc, batched, base model)
# ─────────────────────────────────────────────────────────────────────────────

def perplexity_scores(
    texts: Sequence[str],
    model,
    tokenizer,
    *,
    max_len: int = 512,
    batch_size: int = 8,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Per-document perplexity under `model`. Returns an array aligned with `texts`.

    Uses manual per-row cross-entropy so each document gets its own PPL
    (HF's built-in loss averages across the whole batch).
    """
    import torch
    log = logger or _LOG
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731

    tok = getattr(tokenizer, "tokenizer", tokenizer)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    ppls = np.empty(len(texts), dtype=np.float32)
    for i in tqdm(range(0, len(texts), batch_size), desc="  ppl", leave=False):
        chunk = list(texts[i:i + batch_size])
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=max_len,
                  padding=True, add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attn).logits
        # shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:].clone()
        shift_mask = attn[:, 1:].clone()
        shift_labels[shift_mask == 0] = -100
        # per-row mean NLL
        logp = torch.log_softmax(shift_logits.float(), dim=-1)
        gathered = logp.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        gathered = gathered * shift_mask  # zero out pad positions
        tok_counts = shift_mask.sum(dim=1).clamp(min=1)
        nll = -(gathered.sum(dim=1) / tok_counts)
        batch_ppl = torch.exp(nll.clamp(max=20)).cpu().numpy()
        ppls[i:i + len(chunk)] = batch_ppl

    if was_training:
        model.train()
    log.info(f"  [perplexity] scored {len(texts):,} docs "
             f"| median={np.median(ppls):.1f}  p10={np.percentile(ppls,10):.1f}  "
             f"p90={np.percentile(ppls,90):.1f}")
    return ppls


def ppl_filter(
    texts: Sequence[str],
    ppls: np.ndarray,
    *,
    min_ppl: float = 3.0,
    max_ppl: float = 200.0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], np.ndarray]:
    """Drop garbage (PPL>max: OCR junk, encoding errors) and trivia (PPL<min).

    Returns the surviving texts and their aligned PPL array.
    """
    log = logger or _LOG
    keep = (ppls >= min_ppl) & (ppls <= max_ppl)
    kept_texts = [t for t, k in zip(texts, keep) if k]
    kept_ppls = ppls[keep]
    log.info(f"  [ppl_filter] {len(kept_texts):,}/{len(texts):,} kept "
             f"(PPL window [{min_ppl}, {max_ppl}])")
    return kept_texts, kept_ppls


# ─────────────────────────────────────────────────────────────────────────────
# 8. DSIR — Data Selection via Importance Resampling (Xie et al. 2023)
# ─────────────────────────────────────────────────────────────────────────────

def dsir_weights(
    raw_texts: Sequence[str],
    target_texts: Sequence[str],
    *,
    ngram: int = 2,
    n_features: int = 2 ** 18,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Per-document importance log-ratio log p_target(x) / p_raw(x).

    Higher = the document looks more like the in-domain target (gold QA) than
    like the generic raw pool. Uses hashed n-gram counts (no vocabulary blowup).
    """
    from sklearn.feature_extraction.text import HashingVectorizer
    log = logger or _LOG

    vec = HashingVectorizer(ngram_range=(1, ngram), n_features=n_features,
                            alternate_sign=False, norm=None, binary=False)
    raw_X = vec.transform(raw_texts)                       # (N_raw, F) sparse counts
    tgt_X = vec.transform(target_texts)                    # (N_tgt, F)

    # feature-frequency distributions, smoothed
    raw_freq = np.asarray(raw_X.sum(axis=0)).ravel() + 1.0
    tgt_freq = np.asarray(tgt_X.sum(axis=0)).ravel() + 1.0
    raw_freq /= raw_freq.sum()
    tgt_freq /= tgt_freq.sum()
    log_ratio = np.log(tgt_freq) - np.log(raw_freq)        # (F,)

    # per-doc score = sum of feature counts * log-ratio, length-normalised
    doc_counts = np.asarray(raw_X.sum(axis=1)).ravel().clip(min=1)
    scores = np.asarray(raw_X.dot(log_ratio)).ravel() / doc_counts
    log.info(f"  [dsir] scored {len(raw_texts):,} docs "
             f"| mean={scores.mean():.3f}  std={scores.std():.3f}")
    return scores.astype(np.float32)


def dsir_resample(
    texts: Sequence[str],
    weights: np.ndarray,
    *,
    keep_n: Optional[int] = None,
    keep_frac: Optional[float] = None,
    seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> List[int]:
    """Gumbel top-k resampling on importance weights. Returns selected indices.

    Specify either `keep_n` (absolute) or `keep_frac` (proportion). Gumbel noise
    makes this a proper sample from the importance distribution rather than a
    deterministic top-k, which preserves diversity.
    """
    log = logger or _LOG
    n = len(texts)
    if keep_n is None:
        keep_n = int(round(n * (keep_frac if keep_frac is not None else 1.0)))
    keep_n = max(1, min(keep_n, n))

    rng = np.random.default_rng(seed)
    gumbel = -np.log(-np.log(rng.uniform(size=n).clip(1e-12, 1 - 1e-12)))
    keys = weights + gumbel
    idx = np.argpartition(-keys, keep_n - 1)[:keep_n]
    idx = idx[np.argsort(-keys[idx])]  # sorted best-first
    log.info(f"  [dsir_resample] selected {keep_n:,}/{n:,} docs")
    return idx.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 9. CURRICULUM ORDERING
# ─────────────────────────────────────────────────────────────────────────────

def curriculum_order(
    texts: Sequence[str],
    ppls: np.ndarray,
    *,
    scheme: str = "ascending",
    n_buckets: int = 5,
    seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Order documents easy -> hard.

    scheme="ascending" : pure sort by PPL (easy first).
    scheme="bucketed"  : split into `n_buckets` PPL bins, shuffle *within* each
                         bin, then concatenate bins easy->hard. Keeps a curriculum
                         gradient while avoiding pathological identical-difficulty
                         runs.
    """
    log = logger or _LOG
    order = np.argsort(ppls)  # ascending PPL
    if scheme == "ascending":
        ordered = [texts[i] for i in order]
    elif scheme == "bucketed":
        rng = random.Random(seed)
        buckets = np.array_split(order, n_buckets)
        ordered = []
        for b in buckets:
            b = list(b)
            rng.shuffle(b)
            ordered.extend(texts[i] for i in b)
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    log.info(f"  [curriculum] ordered {len(ordered):,} docs ({scheme})")
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# 10. REPLAY INTERLEAVING (anti-catastrophic-forgetting)
# ─────────────────────────────────────────────────────────────────────────────

def interleave_with_replay(
    domain_ordered: Sequence[str],
    replay_pool: Sequence[str],
    *,
    replay_ratio: float = 0.25,
    seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Spread replay docs evenly through the curriculum-ordered domain corpus.

    `replay_ratio` is the fraction of the *final* corpus that is replay text.
    Spreading (rather than appending) keeps the model anchored to general/EN
    knowledge throughout training, not just at the end.
    """
    log = logger or _LOG
    if not replay_pool or replay_ratio <= 0:
        return list(domain_ordered)

    rng = random.Random(seed)
    n_domain = len(domain_ordered)
    n_replay = int(round(n_domain * replay_ratio / max(1e-9, (1 - replay_ratio))))
    if n_replay >= len(replay_pool):
        replay = list(replay_pool)
        rng.shuffle(replay)
    else:
        replay = rng.sample(list(replay_pool), n_replay)

    out: List[str] = []
    if replay:
        every = max(1, n_domain // len(replay))
    else:
        every = n_domain + 1
    ri = 0
    for i, d in enumerate(domain_ordered):
        out.append(d)
        if ri < len(replay) and (i + 1) % every == 0:
            out.append(replay[ri]); ri += 1
    out.extend(replay[ri:])  # any leftover replay at the tail
    log.info(f"  [replay] interleaved {len(replay):,} replay docs into "
             f"{n_domain:,} domain docs -> {len(out):,} total "
             f"(replay≈{len(replay)/max(1,len(out)):.0%})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 11. CURRICULUM TRAINER — sequential sampler so ordering survives
# ─────────────────────────────────────────────────────────────────────────────

def make_curriculum_trainer_cls():
    """Return an SFTTrainer subclass that reads the train set *in order*.

    The default HF Trainer wraps the train set in a RandomSampler, which would
    destroy the easy->hard curriculum. We override the sampler to sequential.
    Imported lazily so this module doesn't require trl at import time.
    """
    from trl import SFTTrainer
    from torch.utils.data import SequentialSampler

    class CurriculumSFTTrainer(SFTTrainer):
        def _get_train_sampler(self, *args, **kwargs):
            return SequentialSampler(self.train_dataset)

    return CurriculumSFTTrainer


# ─────────────────────────────────────────────────────────────────────────────
# FORGETTING PROBE — general-domain PPL before vs after CPT
# ─────────────────────────────────────────────────────────────────────────────

def forgetting_probe(
    probe_texts: Sequence[str],
    model,
    tokenizer,
    *,
    max_len: int = 512,
    batch_size: int = 8,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Mean PPL on a fixed general-domain holdout. Call before and after CPT;
    a large increase signals catastrophic forgetting of general ability."""
    ppls = perplexity_scores(probe_texts, model, tokenizer,
                             max_len=max_len, batch_size=batch_size, logger=logger)
    return float(np.exp(np.log(ppls.clip(min=1e-6)).mean()))  # geometric mean


# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST — record the recipe for the paper appendix (defensibility)
# ─────────────────────────────────────────────────────────────────────────────

def recipe_manifest(stage_counts: Dict[str, int], params: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a JSON-serialisable record of what the recipe did, stage by stage.

    `stage_counts` maps a stage name -> surviving doc count, so the paper can
    report exactly how many documents each filter removed.
    """
    funnel = []
    prev = None
    for name, n in stage_counts.items():
        delta = None if prev is None else (prev - n)
        funnel.append({"stage": name, "docs": n, "removed": delta})
        prev = n
    return {"funnel": funnel, "params": params}
