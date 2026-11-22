"""Build the long item x forecaster table for ForecastBench round 2024-07-21.

Items: the 596 question x horizon rows of the human question set; we keep the
578 resolved ones with outcome 0/1. Forecasters: every model x prompt file in
the processed forecast sets (2024-07-21), the FB human medians, and the
individual public and superforecaster forecasts.

Usage: python -I build_matrix_2024.py <project folder>
Writes derived/matrix_2024.csv.
"""
import csv, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longpath import lp  # noqa: E402  (some file names exceed Windows' 260-character limit)

ROOT = sys.argv[1]
RAW = os.path.join(ROOT, "data", "raw", "forecastbench")
PROC = os.path.join(RAW, "forecastbench-processed-forecast-sets", "2024-07-21")
HUM = os.path.join(RAW, "human_2024")
MARKET = {"manifold", "metaculus", "polymarket", "infer"}
PROMPTS = ["superforecaster_with_news_1", "superforecaster_with_news_2", "superforecaster_with_news_3",
           "scratchpad_with_news_with_freeze_values", "scratchpad_with_news", "scratchpad_with_freeze_values",
           "zero_shot_with_freeze_values", "scratchpad", "zero_shot"]

def key(qid, source, res_date):
    return (str(qid), None if source in MARKET else res_date[:10])

# items and outcomes from the FB public-median file (same 596 rows as every model file)
items = {}
for f in json.load(open(lp(os.path.join(PROC, "2024-07-21.ForecastBench.human_public.json")), encoding="utf-8"))["forecasts"]:
    if f["resolved"] and f["resolved_to"] in (0, 0.0, 1, 1.0):
        items[key(f["id"], f["source"], f["resolution_date"])] = (f["source"], float(f["resolved_to"]))

rows = []
def add(group, forecaster, model, org, prompt, k, p, imputed):
    src, o = items[k]
    rows.append(dict(item=f"{k[0]}|{k[1] or 'mkt'}", qid=k[0], source=src,
                     qtype="market" if src in MARKET else "dataset", outcome=o, group=group,
                     forecaster=forecaster, model=model, org=org, prompt=prompt,
                     forecast=p, imputed=int(imputed)))

for fn in sorted(os.listdir(lp(PROC))):
    d = json.load(open(lp(os.path.join(PROC, fn)), encoding="utf-8"))
    stem = fn[len("2024-07-21."):-len(".json")]
    org = d.get("model_organization", "")
    if stem.startswith("ForecastBench."):
        group = {"human_public": "public_median", "human_super": "super_median"}.get(stem.split(".", 1)[1], "baseline")
        model, prompt = stem.split(".", 1)[1], ""
    else:
        group = "model"
        body = stem.split(".", 1)[1] if not stem.startswith("2nd_news.") else stem.split(".", 2)[2] + "__2nd_news"
        prompt = next(p for p in PROMPTS if body.split("__")[0].endswith(p))
        model = body.split("__")[0][: -len(prompt) - 1] + ("__2nd_news" if "__2nd_news" in body else "")
    for f in d["forecasts"]:
        if isinstance(f["id"], list):
            continue
        k = key(f["id"], f["source"], f["resolution_date"])
        if k in items and f["forecast"] is not None:
            add(group, stem, model, org, prompt, k, float(f["forecast"]), f.get("imputed", False))

dropped = 0
for grp, fn in (("public", "public_individual.json"), ("super", "super_individual.json")):
    for f in json.load(open(os.path.join(HUM, fn), encoding="utf-8"))["forecasts"]:
        k = key(f["id"], f["source"], f["resolution_date"] or "")
        p = f["forecast"]
        if k not in items or p is None:
            continue
        if not (0 <= p <= 1):
            dropped += 1
            continue
        add(grp, f["user_id"], "", "", "", k, float(p), False)

out = os.path.join(ROOT, "derived", "matrix_2024.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("items", len(items), "rows", len(rows), "dropped out-of-range human forecasts", dropped)
