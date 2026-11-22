"""H7 (version 1): are organized machines colleagues or copies of the base models?

The ForecastBench tournament admits systems built by outside teams that may use tools,
retrieval, extra context, fine-tuning or ensembling. Such systems reached statistical
parity with the 2024 superforecaster benchmark by July 2026 (Forecasting Research
Institute, 2026). Here I ask whether they reached it by erring differently from the
base models (breadth) or by being better copies (the same shared error, less noise).

Samples (rounds 2025-10-26 to 2026-10-25, when external submissions exist):
  base   = models run by ForecastBench, zero-shot prompt WITH freeze values (the base model
           given the same public information), first setting, no web search
           (panel_rounds.csv.gz). Robustness: zero shot without freeze values.
  system = external submissions marked leaderboard-eligible (panel_external.csv.gz) that
           forecast at least 80% of the round's items.
Items: dataset questions with horizons of 7 and 30 days (all that have resolved), and,
separately, market questions against the market price on the due date.

Per round:
  a_base, a_sys     shared-error share among base models / among systems
  r_sys_cons        mean correlation of a system's error with the error of the base-model
                    consensus (mean forecast of all base models)
  r_base_cons       the same for a base model, against the leave-one-out consensus
  best_*            Brier of the best base model / system, ranked on earlier rounds only
                    (excess Brier over the item mean of base models, items resolved before
                    the round's due date); also ex post
  a_pair, r_pair    shared share and error correlation of the prior-best system and the
                    prior-best base model
  gain_pair         Brier of the best system minus Brier of the 50/50 average of the best
                    system and the best base model (positive: adding the base model helps)
  gain_top5_sys     Brier of the best system minus the mean of the prior top-5 systems
Market part: share of model-rounds / system-rounds with a lower Brier than the market
price on the same items, and best minus price, by half-year (this also re-computes the
v0 table for base models with and without the price).

Usage: python -I analysis_h7_tournament.py <project folder>
Writes derived/h7_rounds.csv, derived/h7_market.csv and derived/results_h7.txt.
"""
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1]
T0 = pd.Timestamp("2024-07-21")
lines = []
say = lines.append


def comps(df, key="item", who="who"):
    g = df.groupby(key).e
    n, mu, v = g.size(), g.mean(), g.var(ddof=1)
    ok = n >= 2
    F, s2 = (mu ** 2 - v / n)[ok].mean(), v[ok].mean()
    return F / (F + s2)


def period(d):
    d = pd.Timestamp(d)
    return "2024" if d.year == 2024 else f"{d.year} H{1 if d.month <= 6 else 2}"


# ---------- load ----------
p = pd.read_csv(ROOT + "/derived/panel_rounds.csv.gz", dtype={"base_model": str, "config": str, "org": str},
                keep_default_na=False)
p["mkt_value"] = pd.to_numeric(p.mkt_value, errors="coerce")
p["web"] = p.config.str.lower().str.contains("web-search|web_search")
p = p[(p.run == 1) & ~p.web].copy()
p["who"] = p.base_model
x = pd.read_csv(ROOT + "/derived/panel_external.csv.gz", dtype={"system": str, "team": str}, keep_default_na=False)
x["mkt_value"] = pd.to_numeric(x.mkt_value, errors="coerce")
x = x[x.eligible == 1].copy()
x["who"] = x.system
for d in (p, x):
    d["e"] = d.forecast - d.outcome
    d["due"] = pd.to_datetime(d["round"])
    d["resolved_at"] = d.due + pd.to_timedelta(d.horizon, unit="D")

FIRST = x["round"].min()
say(f"External submissions from {FIRST}; eligible systems {x.system.nunique()}, teams {x.team.nunique()}, "
    f"system-rounds {x.groupby(['round', 'system']).ngroups}")


# ---------- dataset items ----------
def dataset_part(prompt):
    b = p[(p.prompt == prompt) & (p.qtype == "dataset") & (p.horizon <= 31) & (p["round"] >= FIRST)].copy()
    s = x[(x.qtype == "dataset") & (x.horizon <= 31)].copy()
    # reference for excess Brier: item mean of base-model squared errors
    ref = (b.e ** 2).groupby([b["round"], b.item]).mean().rename("ref")
    b = b.join(ref, on=["round", "item"])
    s = s.join(ref, on=["round", "item"])
    s = s[s.ref.notna()]
    b["exc"], s["exc"] = b.e ** 2 - b.ref, s.e ** 2 - s.ref
    out = []
    for rnd, br in b.groupby("round"):
        sr = s[s["round"] == rnd]
        items = br.item.unique()
        if len(items) < 30 or br.who.nunique() < 4:
            continue
        cov = sr.groupby("who").item.nunique() / len(items)
        sr = sr[sr.who.isin(cov[cov >= 0.8].index)]
        if sr.who.nunique() < 3:
            continue
        due = br.due.iloc[0]
        rec = dict(round=rnd, years=(due - T0).days / 365.25, n_items=len(items), n_base=br.who.nunique(),
                   n_sys=sr.who.nunique(), a_base=comps(br), a_sys=comps(sr))
        wb = br.pivot_table(index="item", columns="who", values="e")
        ws = sr.pivot_table(index="item", columns="who", values="e").reindex(wb.index)
        cons = wb.mean(axis=1)
        rec["r_sys_cons"] = np.nanmean([ws[c].corr(cons) for c in ws.columns])
        rec["r_base_cons"] = np.nanmean([wb[c].corr(wb.drop(columns=c).mean(axis=1)) for c in wb.columns])
        bb, bs = (wb ** 2).mean(), (ws ** 2).mean()
        rec["best_base_expost"], rec["best_sys_expost"] = bb.min(), bs.min()
        rec["cons_base"] = (cons ** 2).mean()
        rec["sys_beat_best_base"] = (bs < bb.min()).mean()
        # prior ranking, no look-ahead
        pb = b[(b.due < due) & (b.resolved_at < due)].groupby("who").exc.agg(["mean", "size"])
        ps = s[(s.due < due) & (s.resolved_at < due)].groupby("who").exc.agg(["mean", "size"])
        pb = pb[(pb["size"] >= 50) & pb.index.isin(bb.index)].sort_values("mean")
        ps = ps[(ps["size"] >= 50) & ps.index.isin(bs.index)].sort_values("mean")
        if len(pb) and len(ps):
            tb, ts = pb.index[0], ps.index[0]
            rec["best_base"], rec["best_sys"] = bb[tb], bs[ts]
            pair = pd.concat([wb[tb].rename("b"), ws[ts].rename("s")], axis=1).dropna()
            rec["r_pair"] = pair.b.corr(pair.s)
            long = pair.reset_index().melt(id_vars="item", value_name="e")
            rec["a_pair"] = comps(long)
            rec["mix_pair"] = (((pair.b + pair.s) / 2) ** 2).mean()
            rec["gain_pair"] = (pair.s ** 2).mean() - rec["mix_pair"]
            top5 = list(ps.index[:5])
            if len(top5) >= 3:
                rec["gain_top5_sys"] = bs[ts] - (ws[top5].mean(axis=1) ** 2).mean()
                rec["a_top5_sys"] = comps(sr[sr.who.isin(top5)])
        out.append(rec)
    return pd.DataFrame(out)


for prompt, lab in (("zs_freeze", "MAIN: base models with freeze values"), ("zs", "R: base models without freeze values")):
    r = dataset_part(prompt)
    r["period"] = r["round"].map(period)
    if prompt == "zs_freeze":
        r.to_csv(ROOT + "/derived/h7_rounds.csv", index=False)
    say(f"\n== Dataset items (7 and 30 days), {lab}: {len(r)} rounds, {r['round'].min()} to {r['round'].max()}")
    cols = ["n_base", "n_sys", "a_base", "a_sys", "r_base_cons", "r_sys_cons", "best_base", "best_sys",
            "sys_beat_best_base", "r_pair", "a_pair", "gain_pair", "gain_top5_sys"]
    say(r.groupby("period")[cols].mean().round(3).to_string())
    say("   all rounds: " + ", ".join(f"{c} {r[c].mean():.3f}" for c in cols))
    for c in ("gain_pair", "gain_top5_sys", "best_sys", "best_base"):
        v = r[c].dropna()
        say(f"   {c:14s} mean {v.mean():+.4f} (s.e. across rounds {v.std(ddof=1) / np.sqrt(len(v)):.4f}), "
            f"positive in {int((v > 0).sum())} of {len(v)} rounds")
    d = (r.best_base - r.best_sys).dropna()
    say(f"   best base minus best system (prior-ranked): {d.mean():+.4f} (s.e. {d.std(ddof=1) / np.sqrt(len(d)):.4f}),"
        f" system better in {int((d > 0).sum())} of {len(d)} rounds; relative {(d / r.best_base).mean() * 100:.1f}%")
    dd = (r.r_base_cons - r.r_sys_cons).dropna()
    say(f"   r with base consensus, base minus system: {dd.mean():+.3f} (s.e. {dd.std(ddof=1) / np.sqrt(len(dd)):.3f})")


# ---------- market items vs the price ----------
def beat_price(df, label):
    d = df[(df.qtype == "market") & df.mkt_value.notna()].copy()
    d["e_mkt"] = d.mkt_value - d.outcome
    d["dl"] = d.e ** 2 - d.e_mkt ** 2          # item-level loss relative to the price
    rows = []
    for (rnd, who), g in d.groupby(["round", "who"]):
        if len(g) < 20:
            continue
        rows.append(dict(round=rnd, who=who, n=len(g), brier=(g.e ** 2).mean(), price=(g.e_mkt ** 2).mean(),
                         r_price=np.corrcoef(g.e, g.e_mkt)[0, 1] if g.e.std() > 0 else np.nan))
    t = pd.DataFrame(rows)
    t["beats"] = t.brier < t.price
    t["period"] = t["round"].map(period)
    t["group"] = label
    # prior-ranked best: ranked on items of earlier rounds resolved before this round's due date
    pr = []
    for rnd, tr in t.groupby("round"):
        due = pd.Timestamp(rnd)
        prior = d[(d.due < due) & (d.resolved_at < due)].groupby("who").dl.agg(["mean", "size"])
        prior = prior[(prior["size"] >= 20) & prior.index.isin(tr.who)].sort_values("mean")
        if len(prior):
            row = tr.set_index("who").loc[prior.index[0]]
            pr.append(dict(round=rnd, group=label, period=period(rnd), diff=row.brier - row.price))
    PRIOR.append(pd.DataFrame(pr))
    return t


PRIOR = []


mk = pd.concat([beat_price(p[p.prompt == "zs"], "base, no price"),
                beat_price(p[p.prompt == "zs_freeze"], "base, with price"),
                beat_price(x, "tournament systems")], ignore_index=True)
mk.to_csv(ROOT + "/derived/h7_market.csv", index=False)
say("\n== Market items: share of model-rounds with a lower Brier than the market price (same items)")
say(mk.pivot_table(index="period", columns="group", values="beats", aggfunc="mean").round(3).to_string())
say("   all rounds: " + ", ".join(f"{g} {v:.3f} (n {int(n)})" for g, v, n in
                                   mk.groupby("group").beats.agg(["mean", "size"]).itertuples()))
say("\n== Best in round minus the price (Brier; negative = beats the price), mean by period")
best = mk.loc[mk.groupby(["group", "round"]).brier.idxmin()]
best = best.assign(diff=best.brier - best.price)
say(best.pivot_table(index="period", columns="group", values="diff", aggfunc="mean").round(3).to_string())
pp = pd.concat(PRIOR, ignore_index=True)
say("\n== Prior-ranked best (no look-ahead) minus the price, mean by period (negative = beats the price)")
say(pp.pivot_table(index="period", columns="group", values="diff", aggfunc="mean").round(4).to_string())
for g, v in pp[pp["round"] >= FIRST].groupby("group")["diff"]:
    say(f"   since {FIRST}, {g}: {v.mean():+.4f} (s.e. {v.std(ddof=1) / np.sqrt(len(v)):.4f}), "
        f"beats the price in {int((v < 0).sum())} of {len(v)} rounds")
say("\n== Correlation of errors with the price's error (mean over model-rounds)")
say(mk.pivot_table(index="period", columns="group", values="r_price", aggfunc="mean").round(3).to_string())
sub = mk[mk["round"] >= FIRST]
say(f"\n   rounds since {FIRST}: " + ", ".join(f"{g} {v:.3f} (n {int(n)})" for g, v, n in
                                              sub.groupby("group").beats.agg(["mean", "size"]).itertuples()))

txt = "\n".join(lines)
print(txt)
open(ROOT + "/derived/results_h7.txt", "w", encoding="utf-8").write(txt)
