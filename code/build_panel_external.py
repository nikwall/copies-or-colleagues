"""Build the panel of external tournament submissions (ForecastBench, Oct 2025 onward).

External teams submit systems that may use tools, retrieval, added context, fine-tuning
or ensembling (the "tournament" leaderboard). These are the systems behind the July 2026
claim of parity with superforecasters (Forecasting Research Institute, 2026).
Rows: single questions only, resolved to 0/1, not imputed. Same item keys as
build_panel_rounds.py: dataset items are (id, resolution date), market items are (id).

Usage: python -I build_panel_external.py <project folder>
Writes derived/panel_external.csv.gz.
"""
import csv
import gzip
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longpath import lp  # noqa: E402

ROOT = sys.argv[1]
R = os.path.join(ROOT, "data", "raw", "forecastbench", "forecastbench-processed-forecast-sets")
MARKET = {"manifold", "metaculus", "polymarket", "infer"}


def day(s):
    return date.fromisoformat(s[:10])


rows = 0
out = os.path.join(ROOT, "derived", "panel_external.csv.gz")
cols = ["round", "item", "qtype", "source", "horizon", "outcome", "mkt_value", "team", "system", "eligible",
        "forecast"]
with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for rnd in sorted(os.listdir(lp(R))):
        due = day(rnd)
        files = sorted(f for f in os.listdir(lp(os.path.join(R, rnd))) if ".external." in f)
        for fn in files:
            with open(lp(os.path.join(R, rnd, fn)), encoding="utf-8") as f:
                d = json.load(f)
            team = d.get("organization", "")
            # system key: submission label from the file name plus the declared model name
            label = fn[len(rnd) + len(".external."):-len(".json")]
            system = f"{label} [{d.get('model', '')}]"
            elig = int(bool(d.get("leaderboard_eligible", True)))
            for x in d["forecasts"]:
                if isinstance(x["id"], list) or x.get("imputed") or x["forecast"] is None:
                    continue
                if not x["resolved"] or x["resolved_to"] not in (0, 1, 0.0, 1.0):
                    continue
                mkt = x["source"] in MARKET
                hz = (day(x["resolution_date"]) - due).days
                item = f"{x['id']}|mkt" if mkt else f"{x['id']}|{x['resolution_date'][:10]}"
                w.writerow([rnd, item, "market" if mkt else "dataset", x["source"], hz, float(x["resolved_to"]),
                            x.get("market_value_on_due_date") if mkt else "", team, system, elig,
                            float(x["forecast"])])
                rows += 1
        if files:
            print(rnd, len(files), "external files", flush=True)
print("rows", rows)
