"""H6: is a shared anchor (the market price) a source of shared error?

Part A, round 2024-07-21, 57 resolved market items. Humans and models were shown the
same freeze value (the community or market price on 12 Jul 2024). For public
individuals, superforecasters and each model prompt class, report the distance to
the anchor, the share of forecasts within 0.02 of it, the pooled correlation of
errors with the anchor's error, the shared-error share a and the Brier score.
Within model: zero shot vs zero shot with the price, scratchpad vs scratchpad with it.

Part B, all rounds (panel_rounds.csv.gz), market items, zero shot with vs without the
price, first setting, no web search: per round and model, correlation of errors with
the due-date market price's error and distance to it; per round, a among models.
Differences are averaged per round, then across rounds (s.e. across rounds).

Usage: python -I analysis_h6.py <project folder>
"""
import json
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1]
lines = []
say = lines.append


def comps(df, key="item"):
    g = df.groupby(key).e
    n, mu, v = g.size(), g.mean(), g.var(ddof=1)
    ok = n >= 2
    F, s2 = (mu ** 2 - v / n)[ok].mean(), v[ok].mean()
    return F / (F + s2)


# ---------------- Part A ----------------
qs = json.load(open(ROOT + "/data/raw/forecastbench/human_2024/question_set_human.json", encoding="utf-8"))["questions"]
anchor = {f"{q['id']}|mkt": float(q["freeze_datetime_value"]) for q in qs
          if q["source"] in {"manifold", "metaculus", "polymarket", "infer"} and q["freeze_datetime_value"] not in (None, "")}
m = pd.read_csv(ROOT + "/derived/matrix_2024.csv", dtype={"qid": str, "model": str, "org": str, "prompt": str},
                keep_default_na=False)
mk = m[(m.qtype == "market") & m.item.isin(anchor)].copy()
mk["anchor"] = mk.item.map(anchor)
mk["e"] = mk.forecast - mk.outcome
mk["e_a"] = mk.anchor - mk.outcome
mk["dist"] = (mk.forecast - mk.anchor).abs()
mod = mk[(mk.group == "model") & (mk.imputed == 0) & ~mk.model.str.contains("__2nd_news")]
groups = {"public individuals": mk[mk.group == "public"], "superforecasters": mk[mk.group == "super"]}
for p in ("zero_shot", "zero_shot_with_freeze_values", "scratchpad", "scratchpad_with_freeze_values"):
    groups[f"models, {p}"] = mod[mod.prompt == p].assign(forecaster=lambda d: d.model)
anc = mk.drop_duplicates("item")
say(f"PART A  2024 round, {anc.item.nunique()} resolved market items; anchor alone: Brier {(anc.e_a ** 2).mean():.4f}")
say(f"   {'group':40s} {'members':>7s} {'|p-anchor|':>10s} {'within .02':>10s} {'corr(e, e_anchor)':>17s} {'a':>5s} {'Brier':>6s}")
for lab, d in groups.items():
    say(f"   {lab:40s} {d.forecaster.nunique():7d} {d.dist.mean():10.3f} {(d.dist <= 0.02).mean():10.2f}"
        f" {np.corrcoef(d.e, d.e_a)[0, 1]:17.2f} {comps(d):5.2f} {(d.e ** 2).mean():6.4f}")
for lab, d in (("public crowd (mean)", mk[mk.group == "public"]), ("super team (mean)", mk[mk.group == "super"])):
    c = d.groupby("item").agg(p=("forecast", "mean"), o=("outcome", "first"), a=("anchor", "first"))
    say(f"   {lab:40s} Brier {((c.p - c.o) ** 2).mean():.4f} | corr(crowd error, anchor error) "
        f"{np.corrcoef(c.p - c.o, c.a - c.o)[0, 1]:.2f} | 50/50 with anchor {(((c.p + c.a) / 2 - c.o) ** 2).mean():.4f}")
say("   within model (price minus no price), mean over models:")
for base, frz in (("zero_shot", "zero_shot_with_freeze_values"), ("scratchpad", "scratchpad_with_freeze_values")):
    rows = []
    for mdl, d in mod[mod.prompt.isin([base, frz])].groupby("model"):
        a, b = d[d.prompt == base], d[d.prompt == frz]
        if len(a) >= 20 and len(b) >= 20:
            rows.append((np.corrcoef(b.e, b.e_a)[0, 1] - np.corrcoef(a.e, a.e_a)[0, 1], b.dist.mean() - a.dist.mean(),
                         (b.e ** 2).mean() - (a.e ** 2).mean()))
    r = np.array(rows)
    se = r.std(0, ddof=1) / np.sqrt(len(r))
    say(f"     {base:10s}: corr with anchor {r[:, 0].mean():+.2f} (s.e. {se[0]:.2f}), distance {r[:, 1].mean():+.3f} "
        f"(s.e. {se[1]:.3f}), Brier {r[:, 2].mean():+.4f} (s.e. {se[2]:.4f}); corr rises for {int((r[:, 0] > 0).sum())} of {len(r)} models")

# ---------------- Part B ----------------
p = pd.read_csv(ROOT + "/derived/panel_rounds.csv.gz", dtype={"base_model": str, "config": str, "org": str},
                keep_default_na=False)
p["mkt_value"] = pd.to_numeric(p.mkt_value, errors="coerce")
p = p[(p.qtype == "market") & (p.run == 1) & p.mkt_value.notna()
      & ~p.config.str.lower().str.contains("web-search|web_search")].copy()
p["e"] = p.forecast - p.outcome
p["e_a"] = p.mkt_value - p.outcome
p["dist"] = (p.forecast - p.mkt_value).abs()
rows, arows = [], []
for (rnd, mdl), d in p.groupby(["round", "base_model"]):
    a, b = d[d.prompt == "zs"], d[d.prompt == "zs_freeze"]
    if len(a) >= 20 and len(b) >= 20:
        rows.append(dict(round=rnd, model=mdl, corr_zs=np.corrcoef(a.e, a.e_a)[0, 1], corr_zsf=np.corrcoef(b.e, b.e_a)[0, 1],
                         dist_zs=a.dist.mean(), dist_zsf=b.dist.mean(), brier_zs=(a.e ** 2).mean(), brier_zsf=(b.e ** 2).mean()))
for rnd, d in p.groupby("round"):
    a, b = d[d.prompt == "zs"], d[d.prompt == "zs_freeze"]
    if a.base_model.nunique() >= 4 and b.base_model.nunique() >= 4 and a.item.nunique() >= 20:
        mb = d.drop_duplicates("item")
        arows.append(dict(round=rnd, a_zs=comps(a), a_zsf=comps(b), brier_mkt=(mb.e_a ** 2).mean()))
R, A = pd.DataFrame(rows), pd.DataFrame(arows)
R.to_csv(ROOT + "/derived/h6_model_rounds.csv", index=False)
A.to_csv(ROOT + "/derived/h6_rounds.csv", index=False)
say(f"\nPART B  all rounds, market items: {R['round'].nunique()} rounds, {len(R)} model-rounds, {R.model.nunique()} models")
per = R.assign(dc=R.corr_zsf - R.corr_zs, dd=R.dist_zsf - R.dist_zs, db=R.brier_zsf - R.brier_zs).groupby("round")[["dc", "dd", "db"]].mean()
for col, lab in (("dc", "corr with market error"), ("dd", "distance to market price"), ("db", "Brier")):
    say(f"   price minus no price, {lab:26s}: {per[col].mean():+.4f} (s.e. across rounds {per[col].std(ddof=1) / np.sqrt(len(per)):.4f}), "
        f"positive in {(per[col] > 0).mean():.2f} of rounds")
say(f"   levels: corr with market error, no price {R.corr_zs.mean():.2f} | with price {R.corr_zsf.mean():.2f}; "
    f"distance no price {R.dist_zs.mean():.3f} | with price {R.dist_zsf.mean():.3f}")
d_a = A.a_zsf - A.a_zs
say(f"   shared share a among models: no price {A.a_zs.mean():.2f} | with price {A.a_zsf.mean():.2f} | difference {d_a.mean():+.3f} "
    f"(s.e. {d_a.std(ddof=1) / np.sqrt(len(A)):.3f}), rounds {len(A)}")
A["period"] = pd.cut(pd.to_datetime(A["round"]), pd.to_datetime(["2024-01-01", "2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01", "2027-01-01"]),
                     labels=["2024", "2025H1", "2025H2", "2026H1", "2026H2"])
per2 = per.reset_index().assign(period=lambda x: pd.cut(pd.to_datetime(x["round"]), pd.to_datetime(["2024-01-01", "2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01", "2027-01-01"]),
                                                       labels=["2024", "2025H1", "2025H2", "2026H1", "2026H2"]))
say("   by period: a (no price / with price) and within-model change in corr with the market error")
tab = A.groupby("period", observed=True)[["a_zs", "a_zsf"]].mean().join(per2.groupby("period", observed=True)[["dc", "db"]].mean())
for k, r in tab.iterrows():
    say(f"     {k}: a {r.a_zs:.2f} / {r.a_zsf:.2f} | corr change {r.dc:+.2f} | Brier change {r.db:+.4f}")
txt = "\n".join(lines)
print(txt)
open(ROOT + "/derived/results_h6.txt", "w", encoding="utf-8").write(txt)
