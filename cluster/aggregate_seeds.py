#!/usr/bin/env python3
"""Aggregate results across seeds → runs/all_seeds/.

Reads every runs/seed*/ folder and writes:
  per_seed_metrics.csv      one row per (seed, model, task, shot, metric)
  summary_mean_sd.csv       mean, SD, n seeds and 95% CI per (model, task, shot, metric)
  judge_per_seed.csv / judge_summary_mean_sd.csv   LLM-judge dimensions, same layout
  ppl_vs_noppl_by_seed.csv  Perplexity-<X> minus NoPPL-<X> per seed, with mean and SD
  <name>__all_seeds.csv     every per-seed analysis CSV stacked with a `seed` column
The vanilla reference is seed-independent (evaluated once), so it is counted once.
Safe to run at any time; it uses whatever seeds have finished.
"""
import glob, json, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get("PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(RUNS, "all_seeds")
VANILLA = "Qwen3-14B-vanilla"


def seeds():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "seed*"))):
        m = re.search(r"seed(\d+)$", d)
        if m and os.path.isdir(d):
            out.append((int(m.group(1)), d))
    return out


def numeric_items(d, prefix=""):
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and np.isfinite(v):
            yield prefix + k, float(v)
        elif isinstance(v, dict):
            yield from numeric_items(v, prefix + k + ".")


def summarise(df, keys):
    def ci(x):
        x = x.dropna()
        if len(x) < 2: return np.nan
        return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    g = df.groupby(keys)["value"]
    s = g.agg(mean="mean", sd=lambda x: x.std(ddof=1) if len(x) > 1 else np.nan, n_seeds="count")
    s["ci95_half"] = g.apply(ci)
    return s.reset_index()


def main():
    S = seeds()
    if not S:
        print("no runs/seed* folders yet"); return
    os.makedirs(OUT, exist_ok=True)
    first_seed = S[0][0]

    # 1) evaluation metrics
    rows = []
    for seed, d in S:
        for fp in glob.glob(os.path.join(d, "EVALS", "results_cache", "*.json")):
            try:
                j = json.load(open(fp))
            except Exception:
                continue
            m = j.get("model")
            if not m or (m == VANILLA and seed != first_seed):
                continue
            for k, v in numeric_items(j.get("metrics", {})):
                rows.append({"seed": seed, "model": m, "task": j.get("task"),
                             "n_shot": j.get("n_shot"), "metric": k, "value": v})
    met = pd.DataFrame(rows)
    if not met.empty:
        met.to_csv(os.path.join(OUT, "per_seed_metrics.csv"), index=False)
        summ = summarise(met, ["model", "task", "n_shot", "metric"])
        summ.to_csv(os.path.join(OUT, "summary_mean_sd.csv"), index=False)

        # Perplexity-<X> vs NoPPL-<X>, paired by seed
        diffs = []
        for (task, shot, metric), g in met.groupby(["task", "n_shot", "metric"]):
            for suf in ("DAPT", "MCQA", "ExtQA", "AbsQA", "Unified"):
                a = g[g.model == f"Perplexity-{suf}"].set_index("seed")["value"]
                b = g[g.model == f"NoPPL-{suf}"].set_index("seed")["value"]
                common = sorted(set(a.index) & set(b.index))
                if not common: continue
                d = (a.loc[common] - b.loc[common]).astype(float)
                diffs.append({"adapter": suf, "task": task, "n_shot": shot, "metric": metric,
                              "n_seeds": len(d), "mean_diff_ppl_minus_noppl": d.mean(),
                              "sd": d.std(ddof=1) if len(d) > 1 else np.nan,
                              "per_seed": "; ".join(f"s{s}={v:+.4f}" for s, v in d.items())})
        if diffs:
            pd.DataFrame(diffs).to_csv(os.path.join(OUT, "ppl_vs_noppl_by_seed.csv"), index=False)

    # 2) LLM-judge scores (NB6 single-judge file and NB9 multi-judge file)
    jrows = []
    for seed, d in S:
        for fname, judge_hint in (("gemma_judge_scores.json", "gemma_nb6"), ("llm_judge_scores.json", None)):
            fp = os.path.join(d, "EVALS", fname)
            if not os.path.exists(fp): continue
            try:
                data = json.load(open(fp))
            except Exception:
                continue
            groups = {judge_hint: data} if judge_hint else data     # NB9: {judge: {file: rec}}
            for judge, recs in groups.items():
                if not isinstance(recs, dict): continue
                for rec in recs.values():
                    if not isinstance(rec, dict): continue
                    m = rec.get("model")
                    if not m or (m == VANILLA and seed != first_seed): continue
                    sc = [s for s in rec.get("scores") or [] if isinstance(s, dict)]
                    if not sc: continue
                    keys = {k for s in sc for k, v in s.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    for k in sorted(keys):
                        vals = [s[k] for s in sc if isinstance(s.get(k), (int, float))]
                        if vals:
                            jrows.append({"seed": seed, "judge": judge, "model": m, "n_shot": rec.get("n_shot"),
                                          "dimension": k, "value": float(np.mean(vals)), "n_items": len(vals)})
    jd = pd.DataFrame(jrows)
    if not jd.empty:
        jd.to_csv(os.path.join(OUT, "judge_per_seed.csv"), index=False)
        summarise(jd, ["judge", "model", "n_shot", "dimension"]).to_csv(
            os.path.join(OUT, "judge_summary_mean_sd.csv"), index=False)

    # 3) stack each seed's analysis CSVs
    names = {}
    for seed, d in S:
        for fp in glob.glob(os.path.join(d, "analysis_output", "*.csv")) + glob.glob(os.path.join(d, "RESULT", "*.csv")):
            names.setdefault(os.path.basename(fp), []).append((seed, fp))
    for name, items in names.items():
        parts = []
        for seed, fp in items:
            try:
                t = pd.read_csv(fp); t.insert(0, "seed", seed); parts.append(t)
            except Exception:
                pass
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(os.path.join(OUT, name.replace(".csv", "__all_seeds.csv")), index=False)

    print(f"✓ aggregated seeds {[s for s, _ in S]} → {OUT}")
    if not met.empty:
        print(summ.round(4).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
