"""Validation of the 2024 matrix: FB human-median rows, my own medians, model scores
on the human items (FB weighting: overall = mean of dataset mean and market mean)."""
import sys, pandas as pd, numpy as np
m = pd.read_csv(sys.argv[1] + "/derived/matrix_2024.csv", dtype={"qid": str})
m["se"] = (m.forecast - m.outcome) ** 2
def fb(df):  # FB-weighted Brier of one forecaster (rows = items)
    g = df.groupby("qtype").se.mean()
    return pd.Series({"dataset": g.get("dataset", np.nan), "market": g.get("market", np.nan),
                      "overall": 0.5 * g.get("dataset", np.nan) + 0.5 * g.get("market", np.nan),
                      "n": len(df), "imputed": df.imputed.mean()})
print("FB median rows (leaderboard: super 0.132/0.080/0.106, public 0.166/0.117/0.142):")
for g in ("super_median", "public_median"):
    print(" ", g, fb(m[m.group == g]).round(4).to_dict())
for g in ("super", "public"):
    med = m[m.group == g].groupby(["item", "qtype", "outcome"]).forecast.median().reset_index()
    med["se"] = (med.forecast - med.outcome) ** 2; med["imputed"] = 0
    print("  my median of individuals", g, fb(med).round(4).to_dict())
mod = m[m.group == "model"]
s = mod.groupby(["model", "prompt"]).apply(fb, include_groups=False).reset_index()
s["freeze"] = s.prompt.str.contains("freeze")
print("\nmodel x prompt sets:", len(s), "| models:", s.model.nunique(), "| mean imputed share:", round(s.imputed.mean(), 3))
print("best 8 overall WITHOUT freeze values:")
print(s[~s.freeze].sort_values("overall").head(8)[["model", "prompt", "dataset", "market", "overall", "imputed"]].round(4).to_string(index=False))
print("best 5 overall WITH freeze values:")
print(s[s.freeze].sort_values("overall").head(5)[["model", "prompt", "dataset", "market", "overall", "imputed"]].round(4).to_string(index=False))
