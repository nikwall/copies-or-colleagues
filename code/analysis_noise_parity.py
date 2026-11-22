"""Two robustness checks for version 1.

(1) Outcome noise. The shared-error share a = F / (F + s2) counts the noise in the outcome as shared,
    because no forecaster can avoid it. Two independent, calibrated forecasters therefore share part of
    their error even when they think independently. Let U be the irreducible noise, U = E[pi (1 - pi)],
    where pi is the probability given everything knowable at the time. Every forecast made at the time has
    a Brier score of at least U, so the lowest Brier score observed on the same items, B_best, bounds U from
    above. Under the null of independent calibrated members, F = U, so
        a_null  = U / B1  <=  B_best / B1          (the most that independent members could share), and
        rho_LB  = (F - B_best) / (F - B_best + s2)  (lower bound on the share of each member's deviation
                                                     from the truth that the group shares; 0 if F <= B_best).
    The disagreement sqrt(s2), the typical distance of a member's forecast from the group mean in probability
    points, contains no outcome noise at all.
    B_best is taken over every forecast available at the time on the same items with full coverage: the
    superforecaster and public medians and means, every model version, the model means and, on market
    items, the price. In the 2024 round it is computed separately for dataset and market items and weighted
    like all scores (equal weight on the two types).

(2) Parity systems. The Forecasting Research Institute (16 July 2026) named Cassi AI, xAI and Google DeepMind
    as the tournament systems statistically tied with superforecasters. Their entries are compared with the
    best base model of the same round (ranked on earlier rounds, as in H7), on dataset and market items.

Usage: python -I analysis_noise_parity.py <project folder>
Writes derived/results_noise_parity.txt, noise_2024.csv, noise_rounds.csv, noise_pairs.csv and parity_systems.csv.
"""
import json
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1]
lines = []
say = lines.append
MARKET = {"manifold", "metaculus", "polymarket", "infer"}

# =============================================================== (1a) 2024 round
m = pd.read_csv(ROOT + "/derived/matrix_2024.csv", dtype={"qid": str, "model": str, "org": str, "prompt": str},
                keep_default_na=False)
m["e"] = m.forecast - m.outcome
ITEMS = m.drop_duplicates("item").set_index("item")[["qtype", "outcome"]]
QTYPE = ITEMS.qtype
FRONTIER = ["claude_3p5_sonnet", "gpt_4o", "gemini_1p5_pro", "gpt_4_turbo_0409", "claude_3_opus",
            "llama_3_70b", "mistral_large", "qwen_1p5_110b"]


def fbw(s):
    t = QTYPE.reindex(s.index)
    return 0.5 * s[t == "dataset"].mean() + 0.5 * s[t == "market"].mean()


def comps(df):
    g = df.groupby("item").e
    n, mu, v = g.size(), g.mean(), g.var(ddof=1)
    ok = n >= 2
    F = fbw((mu ** 2 - v / n)[ok])
    s2 = fbw(v[ok])
    B1 = fbw((df.e ** 2).groupby(df.item).mean()[ok])
    return F, s2, B1


mod = m[(m.group == "model") & (m.imputed == 0) & ~m.model.str.contains("__2nd_news")]
z = mod[mod.prompt == "zero_shot"]

# candidate forecasts available at the time, item -> forecast
cands = {}
for grp in ("public_median", "super_median"):
    d = m[m.group == grp]
    cands[grp] = d.set_index("item").forecast
for grp in ("public", "super"):
    cands[grp + "_mean"] = m[m.group == grp].groupby("item").forecast.mean()
for (mdl, pr), d in mod.groupby(["model", "prompt"]):
    cands[f"{mdl}|{pr}"] = d.set_index("item").forecast
for pr, d in mod.groupby("prompt"):
    cands[f"model_mean|{pr}"] = d.groupby("item").forecast.mean()
qs = json.load(open(ROOT + "/data/raw/forecastbench/human_2024/question_set_human.json", encoding="utf-8"))["questions"]
price = pd.Series({f"{q['id']}|mkt": float(q["freeze_datetime_value"]) for q in qs
                   if q["source"] in MARKET and q["freeze_datetime_value"] not in (None, "")})
cands["market_price"] = price

best = {}
for qt in ("dataset", "market"):
    its = QTYPE[QTYPE == qt].index
    o = ITEMS.outcome.reindex(its)
    scores = {}
    for k, f in cands.items():
        f = f[~f.index.duplicated()].reindex(its)
        if f.notna().all():
            scores[k] = ((f - o) ** 2).mean()
    k = min(scores, key=scores.get)
    best[qt] = (k, scores[k], len(scores))
U24 = 0.5 * best["dataset"][1] + 0.5 * best["market"][1]
say("== (1a) 2024 round, 578 items: outcome-noise bound")
say(f"   lowest Brier on dataset items: {best['dataset'][1]:.4f} ({best['dataset'][0]}; {best['dataset'][2]} candidates with full coverage)")
say(f"   lowest Brier on market items:  {best['market'][1]:.4f} ({best['market'][0]}; {best['market'][2]} candidates)")
say(f"   bound on irreducible noise U (equal weights): {U24:.4f}")

groups = {
    "Public, independent individuals (500)": m[m.group == "public"],
    "Superforecasters after discussion (40)": m[m.group == "super"],
    "All 17 distinct models, zero shot": z,
    "Eight frontier models, zero shot": z[z.model.isin(FRONTIER)],
    "Distinct models, zero shot with freeze value": mod[mod.prompt == "zero_shot_with_freeze_values"],
}
rows = []
for lab, d in groups.items():
    F, s2, B1 = comps(d)
    ex = max(F - U24, 0.0)
    rows.append(dict(group=lab, B1=B1, F=F, s2=s2, a=F / (F + s2), a_null_max=min(U24 / B1, 1.0),
                     rho_LB=ex / (ex + s2), spread=np.sqrt(s2)))
n24 = pd.DataFrame(rows)
n24.to_csv(ROOT + "/derived/noise_2024.csv", index=False)
say(n24.round(3).to_string(index=False))
say("   a: observed shared share | a_null_max: the most that independent calibrated members of the same accuracy\n"
    "   could share | rho_LB: lower bound on the shared share of deviations from the truth | spread: sqrt(s2)")

# =============================================================== (1b, 2) all rounds
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


def comps_u(df):
    """Unweighted item means (as in H5 and H7)."""
    g = df.groupby("item").e
    n, mu, v = g.size(), g.mean(), g.var(ddof=1)
    ok = n >= 2
    return (mu ** 2 - v / n)[ok].mean(), v[ok].mean(), (df.e ** 2).groupby(df.item).mean()[ok].mean()


def period(r):
    d = pd.Timestamp(r)
    return "2024" if d.year == 2024 else f"{d.year} H{1 if d.month <= 6 else 2}"


def prior_rank(sample, ref, due):
    """Rank forecasters on items of earlier rounds resolved before `due`, by excess Brier over `ref` item means."""
    s = sample[(sample.due < due) & (sample.resolved_at < due)]
    s = s.join(ref, on=["round", "item"])
    s = s[s.ref.notna()]
    r = (s.e ** 2 - s.ref).groupby(s.who).agg(["mean", "size"])
    return r[r["size"] >= 50].sort_values("mean")


ds = lambda d: d[(d.qtype == "dataset") & (d.horizon <= 31)]  # noqa: E731
LAST30 = p.loc[(p.qtype == "dataset") & p.horizon.between(28, 31), "round"].max()
zs, zf, xs = ds(p[p.prompt == "zs"]), ds(p[p.prompt == "zs_freeze"]), ds(x)
ref_zs = (zs.e ** 2).groupby([zs["round"], zs.item]).mean().rename("ref")
ref_zf = (zf.e ** 2).groupby([zf["round"], zf.item]).mean().rename("ref")
FIRST_SYS = x["round"].min()
PARITY = {"Cassi-AI": "Cassi AI", "xAI": "xAI", "xai": "xAI", "Google DeepMind": "Google DeepMind"}


def best_on(items, rnd):
    """Lowest Brier on `items` among all forecasts of the round with full coverage (models, means, systems)."""
    o = pd.concat([zs[zs["round"] == rnd], zf[zf["round"] == rnd]]).drop_duplicates("item").set_index("item").outcome
    o = o.reindex(items)
    pool = []
    for d in (zs, zf, xs):
        dr = d[(d["round"] == rnd) & d.item.isin(items)]
        for who, g in dr.groupby("who"):
            f = g.drop_duplicates("item").set_index("item").forecast.reindex(items)
            if f.notna().all():
                pool.append(((f - o) ** 2).mean())
        if len(dr):
            f = dr.groupby("item").forecast.mean().reindex(items)
            if f.notna().all():
                pool.append(((f - o) ** 2).mean())
    return min(pool) if pool else np.nan


rows, prs = [], []
for rnd in sorted(zs["round"].unique()):
    due = pd.Timestamp(rnd)
    if rnd > LAST30 and rnd < FIRST_SYS:
        continue
    # (1b) H5 top five (zero shot, ranked on earlier rounds), rounds up to the last with 30-day outcomes
    dr = zs[zs["round"] == rnd]
    if rnd <= LAST30 and dr.who.nunique() >= 4 and dr.item.nunique() >= 30:
        top = [w for w in prior_rank(zs, ref_zs, due).index if w in set(dr.who)][:5]
        if len(top) >= 3:
            dt = dr[dr.who.isin(top)]
            its = dt.item.unique()
            F, s2, B1 = comps_u(dt)
            U = best_on(its, rnd)
            ex = max(F - U, 0.0)
            rows.append(dict(round=rnd, period=period(rnd), what="top five models", F=F, s2=s2, B1=B1, a=F / (F + s2),
                             U_max=U, a_null_max=min(U / B1, 1.0), rho_LB=ex / (ex + s2), spread=np.sqrt(s2)))
    # (1b, 2) systems: best system and parity entries against the best base model with the freeze value
    if rnd < FIRST_SYS:
        continue
    br, sr = zf[zf["round"] == rnd], xs[xs["round"] == rnd]
    items = br.item.unique()
    if len(items) < 30 or br.who.nunique() < 4:
        continue
    cov = sr.groupby("who").item.nunique() / len(items)
    sr = sr[sr.who.isin(cov[cov >= 0.8].index)]
    if sr.who.nunique() < 3:
        continue
    rb = prior_rank(zf, ref_zf, due)
    rb = [w for w in rb.index if w in set(br.who)]
    rs = prior_rank(xs, ref_zf, due)
    rs = [w for w in rs.index if w in set(sr.who)]
    if not rb:
        continue
    wb = br[br.who == rb[0]].drop_duplicates("item").set_index("item")
    pairs = {}
    if rs:
        pairs[("best system", rs[0], "")] = sr[sr.who == rs[0]]
    for who, g in sr.groupby("who"):
        team = g.team.iloc[0]
        if team in PARITY:
            pairs[("parity entry", who, PARITY[team])] = g
    for (kind, who, team), g in pairs.items():
        gs = g.drop_duplicates("item").set_index("item")
        common = gs.index.intersection(wb.index)
        if len(common) < 30:
            continue
        long = pd.concat([gs.loc[common, ["e"]].assign(who="s"), wb.loc[common, ["e"]].assign(who="b")]).reset_index()
        F, s2, B1 = comps_u(long)
        U = best_on(common, rnd)
        ex = max(F - U, 0.0)
        es, eb = gs.loc[common, "e"], wb.loc[common, "e"]
        prs.append(dict(round=rnd, period=period(rnd), kind=kind, team=team, system=who, n_items=len(common),
                        brier_sys=(es ** 2).mean(), brier_base=(eb ** 2).mean(), r_pair=es.corr(eb), a=F / (F + s2), B1=B1,
                        U_max=U, a_null_max=min(U / B1, 1.0), rho_LB=ex / (ex + s2), spread=np.sqrt(s2),
                        gain_avg=(es ** 2).mean() - (((es + eb) / 2) ** 2).mean()))

nr = pd.DataFrame(rows)
pr = pd.DataFrame(prs)
nr.to_csv(ROOT + "/derived/noise_rounds.csv", index=False)
pr.to_csv(ROOT + "/derived/noise_pairs.csv", index=False)
cols = ["a", "a_null_max", "rho_LB", "spread", "B1", "U_max"]
say("\n== (1b) Top five models per round (H5 sample, zero shot, dataset items with 7- and 30-day horizons)")
say(nr.groupby("period")[cols].mean().round(3).to_string())
say("   all rounds: " + ", ".join(f"{c} {nr[c].mean():.3f}" for c in cols) + f" ({len(nr)} rounds)")
say("\n== (1b) Best system and best base model (H7 sample), same measures")
bs = pr[pr.kind == "best system"]
say(bs.groupby("period")[cols].mean().round(3).to_string())
say("   all rounds: " + ", ".join(f"{c} {bs[c].mean():.3f}" for c in cols) + f" ({len(bs)} rounds)")

# =============================================================== (2) parity entries, market items too
def market_stats(sub):
    d = sub[(sub.qtype == "market") & sub.mkt_value.notna()].copy()
    d["e_mkt"] = d.mkt_value - d.outcome
    out = []
    for (rnd, who), g in d.groupby(["round", "who"]):
        if len(g) >= 20:
            out.append(dict(round=rnd, system=who, team=PARITY.get(g.team.iloc[0], "other"),
                            r_price=np.corrcoef(g.e, g.e_mkt)[0, 1], diff_price=(g.e ** 2).mean() - (g.e_mkt ** 2).mean()))
    return pd.DataFrame(out)


ms = market_stats(x)
par = pr[pr.kind == "parity entry"]
par.to_csv(ROOT + "/derived/parity_systems.csv", index=False)
say("\n== (2) Entries of the three teams named in the parity report, against the best base model of the round")
agg = par.groupby("team").agg(entries=("system", "nunique"), entry_rounds=("system", "size"), rounds=("round", "nunique"),
                              brier_sys=("brier_sys", "mean"), brier_base=("brier_base", "mean"), a=("a", "mean"),
                              r_pair=("r_pair", "mean"), rho_LB=("rho_LB", "mean"), gain_avg=("gain_avg", "mean"))
tot = par.agg({"brier_sys": "mean", "brier_base": "mean", "a": "mean", "r_pair": "mean", "rho_LB": "mean", "gain_avg": "mean"})
say(agg.round(3).to_string())
say(f"   all three teams ({len(par)} entry-rounds, {par['round'].nunique()} rounds): " +
    ", ".join(f"{k} {v:.3f}" for k, v in tot.items()))
d_b = (par.brier_base - par.brier_sys)
say(f"   base minus entry Brier: {d_b.mean():+.4f} (s.e. {d_b.std(ddof=1) / np.sqrt(len(d_b)):.4f}); entry better in {int((d_b > 0).sum())} of {len(d_b)}")
say(f"   for comparison, best system (prior-ranked): a {bs.a.mean():.3f}, r_pair {bs.r_pair.mean():.3f}, rho_LB {bs.rho_LB.mean():.3f}")
say("\n== (2) Market items: correlation with the price's error and Brier minus the price, by team")
msum = ms.groupby("team").agg(entry_rounds=("system", "size"), r_price=("r_price", "mean"), diff_price=("diff_price", "mean"),
                              beats=("diff_price", lambda v: (v < 0).mean()))
say(msum.round(3).to_string())
for t, g in ms.groupby("team"):
    v = g.diff_price
    say(f"   {t:16s} Brier minus price {v.mean():+.4f} (s.e. {v.std(ddof=1) / np.sqrt(len(v)):.4f}), n {len(v)}")

txt = "\n".join(lines)
print(txt)
open(ROOT + "/derived/results_noise_parity.txt", "w", encoding="utf-8").write(txt)
