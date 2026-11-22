"""H5: is model monoculture rising across ForecastBench rounds (2024-07 to 2026-10)?

Per round, among distinct models (zero-shot prompt, first setting of each model,
no web search):
  a       shared-error share F/(F+s2) (see analysis_h1_h4_2024.py)
  disagr  cross-model variance of forecasts on an item (s2), item mean
  corr    mean pairwise Pearson correlation of errors
  gain    Brier of the prior-best model minus Brier of the mean of the prior top 5
          (positive = adding models helps). "Prior" = ranked on items of earlier
          rounds that had resolved before this round's due date, by excess Brier
          (model minus item mean), so no look-ahead.
  mkt_corr  mean correlation of model errors with market-price errors (market items)
Main item set: dataset questions with horizon <= 30 days (7 and 30), rounds up to
the last round with resolved 30-day outcomes (2026-09-27 in the 31 Oct snapshot,
2026-10-11 in the 17 Nov snapshot; derived from the data). Robustness: 7-day only (all rounds),
with freeze values, including web-search settings.
Trends: OLS of each round metric on years since 2024-07-21 (HC1 s.e.); pair-level
regression of error correlation on time, pair accuracy and same developer, with
standard errors clustered by round.

Usage: python -I analysis_h5.py <project folder>
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

ROOT = sys.argv[1]
p = pd.read_csv(ROOT + "/derived/panel_rounds.csv.gz", dtype={"base_model": str, "config": str, "org": str},
                keep_default_na=False)
p["mkt_value"] = pd.to_numeric(p.mkt_value, errors="coerce")
p["e"] = p.forecast - p.outcome
p["web"] = p.config.str.lower().str.contains("web-search|web_search")
p["due"] = pd.to_datetime(p["round"])
p["resolved_at"] = p.due + pd.to_timedelta(p.horizon, unit="D")
T0 = pd.Timestamp("2024-07-21")
p["years"] = (p.due - T0).dt.days / 365.25


def comps(df):
    g = df.groupby("item").e
    n, mu, v = g.size(), g.mean(), g.var(ddof=1)
    ok = n >= 2
    F, s2 = (mu ** 2 - v / n)[ok].mean(), v[ok].mean()
    return F, s2, F / (F + s2)


def pair_corr(df):
    w = df.pivot_table(index="item", columns="base_model", values="e")
    c = w.corr(min_periods=30)
    iu = np.triu(np.ones(c.shape), 1).astype(bool)
    return c.where(iu).stack()


def ols(y, X, cluster=None):
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    if cluster is None:
        meat = (X * u[:, None] ** 2).T @ X
        V = XtXi @ meat @ XtXi * len(y) / (len(y) - X.shape[1])
    else:
        meat = np.zeros((X.shape[1],) * 2)
        groups = pd.unique(cluster)
        for g in groups:
            s = (X[cluster == g] * u[cluster == g][:, None]).sum(0)
            meat += np.outer(s, s)
        G = len(groups)
        V = XtXi @ meat @ XtXi * G / (G - 1) * (len(y) - 1) / (len(y) - X.shape[1])
    return beta, np.sqrt(np.diag(V))


def round_metrics(sample, prompt="zs", web=False, label=""):
    d = sample[(sample.prompt == prompt) & (sample.run == 1)]
    if not web:
        d = d[~d.web]
    d = d.copy()
    d["exc"] = d.e ** 2 - d.groupby(["round", "item"]).e.transform(lambda x: (x ** 2).mean())
    out, pairs = [], []
    for rnd, dr in d.groupby("round"):
        if dr.base_model.nunique() < 4 or dr.item.nunique() < 30:
            continue
        due = dr.due.iloc[0]
        F, s2, a = comps(dr)
        br = (dr.e ** 2).groupby(dr.base_model).mean()
        prior = d[(d.due < due) & (d.resolved_at < due)]
        ps = prior.groupby("base_model").exc.agg(["mean", "size"])
        ps = ps[(ps["size"] >= 50) & ps.index.isin(br.index)].sort_values("mean")
        top = list(ps.index[:5])
        rec = dict(round=rnd, years=dr.years.iloc[0], n_models=dr.base_model.nunique(), n_items=dr.item.nunique(),
                   a=a, disagr=s2, floor=F, best_expost=br.min(), n_prior=len(top))
        c = pair_corr(dr)
        rec["corr"] = c.mean()
        if len(top) >= 3:
            dt = dr[dr.base_model.isin(top)]
            rec["a_top"] = comps(dt)[2]
            rec["corr_top"] = pair_corr(dt).mean()
            crowd = (dt.groupby("item").e.mean() ** 2).mean()
            rec["best_prior"] = br[top[0]]
            rec["crowd_top"] = crowd
            rec["gain"] = br[top[0]] - crowd
        out.append(rec)
        acc = br.reindex([i for i in c.index.get_level_values(0)]).values
        acc2 = br.reindex([j for j in c.index.get_level_values(1)]).values
        orgs = dr.drop_duplicates("base_model").set_index("base_model").org
        same = (orgs.reindex(c.index.get_level_values(0)).values == orgs.reindex(c.index.get_level_values(1)).values)
        pairs.append(pd.DataFrame(dict(round=rnd, years=dr.years.iloc[0], r=c.values, acc=(acc + acc2) / 2,
                                       same_dev=same.astype(float))))
    return pd.DataFrame(out), pd.concat(pairs, ignore_index=True)


def trend(df, col):
    s = df.dropna(subset=[col])
    b, se = ols(s[col].values, s[["years"]].values)
    return f"{col:12s} first {s[col].iloc[0]:.3f}  last {s[col].iloc[-1]:.3f}  slope/yr {b[1]:+.4f} (s.e. {se[1]:.4f}, t {b[1] / se[1]:+.1f}), rounds {len(s)}"


LAST30 = p.loc[(p.qtype == "dataset") & p.horizon.between(28, 31), "round"].max()
print("last round with 30-day outcomes:", LAST30)
main_items = p[(p.qtype == "dataset") & (p.horizon <= 31) & (p["round"] <= LAST30)]
specs = {
    "MAIN: dataset, 7+30 days, zero shot, no web search": dict(sample=main_items),
    "R1: dataset, 7 days only, all rounds": dict(sample=p[(p.qtype == "dataset") & (p.horizon <= 8)]),
    "R2: main items, zero shot WITH freeze values": dict(sample=main_items, prompt="zs_freeze"),
    "R3: main items, web-search settings included": dict(sample=main_items, web=True),
}
lines = []
for lab, kw in specs.items():
    rm, pr = round_metrics(**kw)
    lines.append(f"\n== {lab}")
    for col in ("a", "a_top", "corr", "corr_top", "disagr", "floor", "best_expost", "gain"):
        lines.append("   " + trend(rm, col))
    X = pr[["years", "acc", "same_dev"]].values
    b, se = ols(pr.r.values, X, cluster=pr["round"].values)
    lines.append(f"   pair-level r on years {b[1]:+.4f} (s.e. {se[1]:.4f}), pair Brier {b[2]:+.3f} (s.e. {se[2]:.3f}),"
                 f" same developer {b[3]:+.4f} (s.e. {se[3]:.4f}); pairs {len(pr)}, rounds {pr['round'].nunique()}")
    if lab.startswith("MAIN"):
        rm.to_csv(ROOT + "/derived/h5_rounds_main.csv", index=False)
        pr.to_csv(ROOT + "/derived/h5_pairs_main.csv", index=False)
        lines.append("   share of rounds where the prior top-5 crowd beats the prior-best model: "
                     f"{(rm.gain > 0).mean():.2f} ({int((rm.gain > 0).sum())} of {rm.gain.notna().sum()})")

# market items: model errors vs market-price errors (zero shot, no price in the prompt)
mk = p[(p.qtype == "market") & (p.prompt == "zs") & (p.run == 1) & ~p.web & p.mkt_value.notna()].copy()
mk["e_mkt"] = mk.mkt_value - mk.outcome
rows = []
for rnd, d in mk.groupby("round"):
    if d.item.nunique() < 20:
        continue
    cs = d.groupby("base_model").apply(lambda x: np.corrcoef(x.e, x.e_mkt)[0, 1] if len(x) >= 20 else np.nan,
                                       include_groups=False)
    br = (d.e ** 2).groupby(d.base_model).mean()
    mb = d.drop_duplicates("item").pipe(lambda x: (x.e_mkt ** 2).mean())
    rows.append(dict(round=rnd, years=d.years.iloc[0], n_items=d.item.nunique(), mkt_corr=cs.mean(),
                     mkt_brier=mb, best_model=br.min(), models_beating_mkt=(br < mb).mean()))
mr = pd.DataFrame(rows)
mr.to_csv(ROOT + "/derived/h5_market.csv", index=False)
lines.append("\n== Market questions: models (zero shot, no price) vs the market price")
for col in ("mkt_corr", "mkt_brier", "best_model", "models_beating_mkt"):
    lines.append("   " + trend(mr, col))

# same model, different settings (2026 run variants) vs distinct models, same round
sv = p[(p.prompt == "zs") & (p.qtype == "dataset") & (p.horizon <= 8)]
lev = []
for rnd, d in sv.groupby("round"):
    multi = d.groupby("base_model").run.nunique()
    multi = multi[multi > 1].index
    if len(multi) == 0:
        continue
    within = [comps(d[d.base_model == b].assign(base_model=d.run.astype(str)))[2] for b in multi]
    across = comps(d[d.run == 1])[2]
    lev.append((rnd, np.mean(within), across, len(multi)))
lines.append("\n== Same model, different settings (2026 rounds, 7-day dataset items)")
for rnd, w, a_, k in lev:
    lines.append(f"   {rnd}: within-model a {w:.2f} ({k} models) | across distinct models a {a_:.2f}")

txt = "\n".join(lines)
print(txt)
open(ROOT + "/derived/results_h5.txt", "w", encoding="utf-8").write(txt)
