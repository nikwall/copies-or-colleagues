"""H1-H4 for idea 1 on ForecastBench round 2024-07-21 (578 resolved items).

Shared-error share a = F / (F + s2) for a group of forecasters, where per item
F_q = mean_error^2 - var/n (unbiased for the squared shared error) and s2_q = var
(idiosyncratic). Item averages use FB weighting (mean of the dataset mean and the
market mean). Model rows with imputed forecasts are treated as missing.
Confidence intervals: bootstrap over question ids (all horizons of a question together).

Usage: python -I analysis_h1_h4_2024.py <project folder>
"""
import sys
import numpy as np
import pandas as pd

ROOT = sys.argv[1]
rng = np.random.default_rng(20261031)
m = pd.read_csv(ROOT + "/derived/matrix_2024.csv", dtype={"qid": str, "model": str, "org": str, "prompt": str},
                keep_default_na=False)
m["e"] = m.forecast - m.outcome
mod = m[(m.group == "model") & (m.imputed == 0) & ~m.model.str.contains("__2nd_news")].copy()
mod["freeze"] = mod.prompt.str.contains("freeze")
NOFRZ = mod[~mod.freeze]
FRONTIER = ["claude_3p5_sonnet", "gpt_4o", "gemini_1p5_pro", "gpt_4_turbo_0409", "claude_3_opus",
            "llama_3_70b", "mistral_large", "qwen_1p5_110b"]
ITEMS = m.drop_duplicates("item").set_index("item")[["qid", "qtype", "outcome"]]
QTYPE = ITEMS.qtype


def fbw(s, qtype=None):
    """FB weighting of an item-indexed series."""
    t = (QTYPE if qtype is None else qtype).reindex(s.index)
    return 0.5 * s[t == "dataset"].mean() + 0.5 * s[t == "market"].mean()


def comps(df, qtype=None):
    g = df.groupby("item").e
    n, mu, v = g.size(), g.mean(), g.var(ddof=1)
    ok = n >= 2
    F = fbw((mu ** 2 - v / n)[ok], qtype)
    s2 = fbw(v[ok], qtype)
    B1 = fbw((df.e ** 2).groupby(df.item).mean()[ok], qtype)
    return dict(B1=B1, F=F, s2=s2, a=F / (F + s2), n_med=float(n[ok].median()))


def a_of(df, qtype=None):
    return comps(df, qtype)["a"]


def boot_a(df, reps=300):
    """Question-cluster bootstrap of the shared share."""
    by_q = dict(tuple(df.groupby("qid")))
    qs = ITEMS.qid.unique()
    vals = []
    for _ in range(reps):
        parts = []
        for j, k in enumerate(rng.choice(qs, len(qs))):
            if k in by_q:
                d = by_q[k].copy()
                d["item"] = d["item"] + f"#{j}"
                parts.append(d)
        bd = pd.concat(parts)
        qt = bd.drop_duplicates("item").set_index("item").qtype
        vals.append(a_of(bd, qt))
    return np.percentile(vals, [2.5, 97.5])


def se_by_item(d):
    return (d.e ** 2).groupby(d.item).mean()


def crowd(d):
    return fbw(d.groupby("item").e.mean() ** 2)


# ---------- H1: the copies ladder ---------------------------------------------
print("H1  shared-error share a by level (prompts without market prices)")
copies = [a_of(d) for _, d in NOFRZ.groupby("model") if d.prompt.nunique() >= 3]
z = NOFRZ[NOFRZ.prompt == "zero_shot"]
same_org = [a_of(d) for _, d in z.groupby("org") if d.model.nunique() >= 2]
one_per_org = z[z.model.isin(z.groupby("org").model.first())]
ladder = {
    "same model, different prompts (mean over models)": np.mean(copies),
    "different models, same developer (zero shot, mean)": np.mean(same_org),
    "different developers (one model each, zero shot)": a_of(one_per_org),
    "all distinct models (zero shot)": a_of(z),
    "frontier models (zero shot)": a_of(z[z.model.isin(FRONTIER)]),
    "public, independent individuals": a_of(m[m.group == "public"]),
    "superforecasters, after discussion": a_of(m[m.group == "super"]),
}
for k, v in ladder.items():
    print(f"   {v:5.2f}  {k}")
print(f"   (models with >=3 prompts: {len(copies)} | developers with >=2 models: {len(same_org)})")
w = z.pivot_table(index="item", columns="model", values="e")
up = lambda c: c.where(np.triu(np.ones(c.shape), 1).astype(bool)).stack()
fr = [x for x in FRONTIER if x in w]
print(f"   Pearson error correlation of model pairs (zero shot): all {up(w.corr()).mean():.2f}, frontier {up(w[fr].corr()).mean():.2f}")

# ---------- H2: distinct models vs the public ----------------------------------
print("\nH2  shared share, distinct models vs public (95% CI, question bootstrap)")
for lab, d in (("models, zero shot", z),
               ("frontier models, zero shot", z[z.model.isin(FRONTIER)]),
               ("models, scratchpad", NOFRZ[NOFRZ.prompt == "scratchpad"]),
               ("public", m[m.group == "public"])):
    lo, hi = boot_a(d)
    print(f"   {lab:28s} a = {a_of(d):.2f}  [{lo:.2f}, {hi:.2f}]")

# ---------- H3: the mosaic against a better machine ---------------------------
print("\nH3  floors, crowds and the best single model (FB-weighted raw Brier)")
pub = comps(m[m.group == "public"])
sup = comps(m[m.group == "super"])
mscore = NOFRZ.groupby(["model", "prompt"]).apply(lambda d: fbw(se_by_item(d)), include_groups=False)
fscore = mod[mod.freeze].groupby(["model", "prompt"]).apply(lambda d: fbw(se_by_item(d)), include_groups=False)
best_key, best = mscore.idxmin(), mscore.min()
mz, mf = comps(z), comps(z[z.model.isin(FRONTIER)])
print(f"   public: individual {pub['B1']:.4f} | floor F {pub['F']:.4f} | observed mean crowd {crowd(m[m.group == 'public']):.4f} (median {pub['n_med']:.0f} per item)")
print(f"   supers: individual {sup['B1']:.4f} | floor F {sup['F']:.4f} | observed mean crowd {crowd(m[m.group == 'super']):.4f} (median {sup['n_med']:.0f} per item)")
print(f"   best single model, no market price: {best:.4f} {best_key} | with market price: {fscore.min():.4f} {fscore.idxmin()}")
print(f"   model crowd, all (zero shot): observed {crowd(z):.4f}, floor {mz['F']:.4f} | frontier: observed {crowd(z[z.model.isin(FRONTIER)]):.4f}, floor {mf['F']:.4f}")
pub_ind = m[m.group == "public"].groupby("forecaster").apply(
    lambda d: fbw(se_by_item(d)) if d.qtype.nunique() == 2 else fbw(se_by_item(d).pipe(lambda s: pd.concat([s, s]))), include_groups=False)
pub_ind2 = m[m.group == "public"].groupby("forecaster").apply(lambda d: se_by_item(d).mean(), include_groups=False)
print(f"   share of public individuals worse than the best model on their own items: {(pub_ind2 > best).mean():.2f}")
nstar = pub["s2"] / (best - pub["F"]) if best > pub["F"] else np.inf
print(f"   N* = s2/(B*-F): public crowd size that beats the best model = {nstar:.1f}")
pubd = m[m.group == "public"]
for N in (1, 2, 3, 5, 10, 20, 40):
    vals = {}
    for it, d in pubd.groupby("item"):
        if len(d) >= N:
            e = d.e.values
            vals[it] = np.mean([np.mean(rng.choice(e, N, replace=False)) ** 2 for _ in range(200)])
    s = pd.Series(vals)
    print(f"   public crowd of {N:2d}: resampled Brier {fbw(s):.4f} (formula {pub['F'] + pub['s2'] / N:.4f})")

# ---------- H4: breadth in a hybrid ---------------------------------------------
print("\nH4  optimal human weight w* in a human + model average (5-fold cross-fitting by question)")


def hybrid(h, mm, lab):
    df = pd.DataFrame({"h": h, "m": mm}).dropna()
    df = df.join(ITEMS)
    folds = {q: i % 5 for i, q in enumerate(rng.permutation(df.qid.unique()))}
    df["f"] = df.qid.map(folds)
    grid = np.linspace(0, 1, 101)

    def loss(d, wv):
        s = (wv * d.h + (1 - wv) * d.m - d.outcome) ** 2
        return 0.5 * s[d.qtype == "dataset"].mean() + 0.5 * s[d.qtype == "market"].mean()

    ws, pred = [], pd.Series(index=df.index, dtype=float)
    for k in range(5):
        tr, te = df[df.f != k], df[df.f == k]
        wk = grid[np.argmin([loss(tr, g) for g in grid])]
        ws.append(wk)
        pred[te.index] = wk * te.h + (1 - wk) * te.m
    full = grid[np.argmin([loss(df, g) for g in grid])]
    sh, sm, sx = fbw((df.h - df.outcome) ** 2), fbw((df.m - df.outcome) ** 2), fbw((pred - df.outcome) ** 2)
    print(f"   {lab:40s} w* = {full:.2f} (folds {min(ws):.2f}-{max(ws):.2f}) | humans {sh:.4f}, models {sm:.4f}, hybrid out of sample {sx:.4f}")


pub_mean = m[m.group == "public"].groupby("item").forecast.mean()
sup_mean = m[m.group == "super"].groupby("item").forecast.mean()
bm = NOFRZ[(NOFRZ.model == best_key[0]) & (NOFRZ.prompt == best_key[1])].set_index("item").forecast
bf_key = fscore.idxmin()
bfm = mod[(mod.model == bf_key[0]) & (mod.prompt == bf_key[1])].set_index("item").forecast
fcrowd = z[z.model.isin(FRONTIER)].groupby("item").forecast.mean()
hybrid(pub_mean, bm, "public crowd + best model (no price)")
hybrid(pub_mean, bfm, "public crowd + best model (with price)")
hybrid(pub_mean, fcrowd, "public crowd + frontier model crowd")
hybrid(sup_mean, bm, "super team + best model (no price)")
hybrid(sup_mean, fcrowd, "super team + frontier model crowd")
eh = (pub_mean - ITEMS.outcome).dropna()
em = (fcrowd - ITEMS.outcome).dropna()
ix = eh.index.intersection(em.index)
C, FM, FH = fbw(eh[ix] * em[ix]), mf["F"], pub["F"]
print(f"   Bates-Granger at the floors (public vs frontier models): F_H {FH:.4f}, F_M {FM:.4f}, C {C:.4f}"
      f" -> w* = {(FM - C) / (FH + FM - 2 * C):.2f}; correlation of shared errors {C / np.sqrt(FH * FM):.2f}")
