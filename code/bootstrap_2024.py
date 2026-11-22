"""Question-level bootstrap for the 2024 results (H3 and H4).

Resamples question ids with replacement (all horizons of a question together),
keeps forecasters fixed, and recomputes in each replicate:
  - the public floor F, idiosyncratic variance s2, observed public crowd Brier
  - Brier of a FIXED model (Claude 3.5 Sonnet, scratchpad; with and without the
    market price) and of the EX POST best variant, re-selected in each replicate
  - N* = s2 / (B_model - F)  (inf if the model is below the public floor)
  - the hybrid weight w* on the public crowd and the cross-fitted (5 folds by
    question) out-of-sample Brier of the hybrid; "edge" = hybrid minus the better
    of its two parts (negative = the hybrid beats both)
All Brier scores use FB weighting (mean of the dataset mean and the market mean),
overall and by question type.

Usage: python -I bootstrap_2024.py <project folder> [replicates]
"""
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rng = np.random.default_rng(20261101)
m = pd.read_csv(ROOT + "/derived/matrix_2024.csv", dtype={"qid": str, "model": str, "org": str, "prompt": str},
                keep_default_na=False)
m["e"] = m.forecast - m.outcome
items = m.drop_duplicates("item").set_index("item")[["qid", "qtype", "outcome"]].sort_index()
idx = {k: i for i, k in enumerate(items.index)}
n = len(items)
is_mkt = (items.qtype == "market").values
outcome = items.outcome.values
qids = items.qid.values
uq = np.unique(qids)
q_items = {q: np.where(qids == q)[0] for q in uq}

# public: per-item mean error, variance, count, mean forecast
pub = m[m.group == "public"].groupby("item").agg(mu=("e", "mean"), v=("e", lambda x: x.var(ddof=1)),
                                                 k=("e", "size"), p=("forecast", "mean")).reindex(items.index)
F_q = (pub.mu ** 2 - pub.v / pub.k).values
s2_q = pub.v.values
pub_p = pub.p.values

# model forecasts: matrix items x variants (imputed treated as missing)
mod = m[(m.group == "model") & (m.imputed == 0) & ~m.model.str.contains("__2nd_news")]
mod = mod.assign(var=mod.model + "|" + mod.prompt)
M = mod.pivot_table(index="item", columns="var", values="forecast").reindex(items.index)
variants = M.columns.values
freeze = np.array(["freeze" in v for v in variants])
MV = M.values
FIX = {"fixed, no price": "claude_3p5_sonnet|scratchpad", "fixed, with price": "claude_3p5_sonnet|scratchpad_with_freeze_values"}


def fbw(x, w):
    """FB-weighted mean of item values x with item multiplicities w (by type)."""
    out = {}
    for lab, sel in (("dataset", ~is_mkt), ("market", is_mkt)):
        ww = w * sel
        out[lab] = np.nansum(x * ww) / np.sum(ww * ~np.isnan(x))
    out["overall"] = 0.5 * out["dataset"] + 0.5 * out["market"]
    return out


def hybrid(h, mm, w, folds):
    """Full-sample w* and cross-fitted out-of-sample Brier for w*h + (1-w)*mm (FB weighted)."""
    ok = ~np.isnan(h) & ~np.isnan(mm) & (w > 0)
    grid = np.linspace(0, 1, 101)
    def loss(sel, g):
        se = (g[:, None] * h[sel][None, :] + (1 - g[:, None]) * mm[sel][None, :] - outcome[sel][None, :]) ** 2
        ws = w[sel] * np.where(is_mkt[sel], 0.5 / np.sum(w[sel] * is_mkt[sel]), 0.5 / np.sum(w[sel] * ~is_mkt[sel]))
        return se @ ws
    full = grid[np.argmin(loss(ok, grid))]
    pred = np.full(n, np.nan)
    for k in range(5):
        tr, te = ok & (folds != k), ok & (folds == k)
        wk = grid[np.argmin(loss(tr, grid))]
        pred[te] = wk * h[te] + (1 - wk) * mm[te]
    return full, fbw((pred - outcome) ** 2, w * ok)


def stats(w, folds):
    r = {}
    F = fbw(F_q, w); s2 = fbw(s2_q, w); bp = fbw((pub_p - outcome) ** 2, w)
    se_all = (MV - outcome[:, None]) ** 2
    scores = np.array([fbw(se_all[:, j], w)["overall"] for j in range(len(variants))])
    best_nf = np.nanargmin(np.where(freeze, np.nan, scores))
    best_f = np.nanargmin(np.where(freeze, scores, np.nan))
    models = dict(FIX)
    models["ex post best, no price"] = variants[best_nf]
    models["ex post best, with price"] = variants[best_f]
    for t in ("dataset", "market", "overall"):
        r[f"public floor F | {t}"] = F[t]
        r[f"public crowd Brier | {t}"] = bp[t]
    for lab, v in models.items():
        j = list(variants).index(v)
        bm = fbw(se_all[:, j], w)
        for t in ("dataset", "market", "overall"):
            r[f"{lab}: Brier | {t}"] = bm[t]
            r[f"{lab}: N* | {t}"] = s2[t] / (bm[t] - F[t]) if bm[t] > F[t] else np.inf
        wst, oos = hybrid(pub_p, MV[:, j], w, folds)
        r[f"{lab}: w* on public"] = wst
        r[f"{lab}: hybrid OOS | overall"] = oos["overall"]
        r[f"{lab}: edge (hybrid - better part)"] = oos["overall"] - min(bp["overall"], bm["overall"])
    return r


def random_folds(present):
    fold_of = {q: i % 5 for i, q in enumerate(rng.permutation(present))}
    return np.array([fold_of.get(q, -1) for q in qids])


# point estimate: full sample, cross-fitted results averaged over 20 fold splits
splits = pd.DataFrame([stats(np.ones(n), random_folds(uq)) for _ in range(20)])
point = splits.mean().to_dict()
point.update({k: splits[k].iloc[0] for k in splits.columns if "OOS" not in k and "edge" not in k})
draws = []
for b in range(REPS):
    pick = rng.choice(uq, len(uq))
    w = np.zeros(n)
    for q in pick:
        w[q_items[q]] += 1
    draws.append(stats(w, random_folds(np.unique(pick))))
D = pd.DataFrame(draws)
rows = []
for k, v in point.items():
    lo, med, hi = np.quantile(D[k].values, [0.025, 0.5, 0.975], method="inverted_cdf")  # inf allowed
    rows.append(dict(statistic=k, point=v, lo=lo, median=med, hi=hi,
                     share_finite=np.isfinite(D[k]).mean() if "N*" in k else np.nan,
                     share_below_0=(D[k] < 0).mean() if "edge" in k else np.nan))
out = pd.DataFrame(rows)
out.to_csv(ROOT + "/derived/bootstrap_2024.csv", index=False)
pd.set_option("display.width", 200)
print(f"replicates: {REPS}, questions: {len(uq)}, items: {n}")
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
