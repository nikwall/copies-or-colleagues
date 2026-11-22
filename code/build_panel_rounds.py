"""Build the all-rounds panel of model forecasts for H5 (ForecastBench, zero-shot prompts).

Keeps every model file that ForecastBench ran itself with the zero-shot prompt, with
and without freeze values (market price / latest data value). Drops external team
submissions, FB baselines, LLM-crowd files and human files.
Rows: single questions only (no combinations), resolved to 0/1, not imputed.
Dataset items are (id, horizon); market items are (id). The market price on the due
date is kept as the human-crowd benchmark.

Usage: python -I build_panel_rounds.py <project folder>
Writes derived/panel_rounds.csv.gz.
"""
import csv
import gzip
import json
import os
import re
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longpath import lp  # noqa: E402

ROOT = sys.argv[1]
R = os.path.join(ROOT, "data", "raw", "forecastbench", "forecastbench-processed-forecast-sets")
MARKET = {"manifold", "metaculus", "polymarket", "infer"}
SKIP = re.compile(r"\.(external|ForecastBench\.(always|naive|imputed|random|llm_crowd|human))")
PROMPT = re.compile(r"[_-]zero[_-]shot([_-]with[_-]freeze[_-]values)?\.json$")


def parse_name(rnd, fn):
    stem = fn[len(rnd) + 1:]
    m = PROMPT.search(stem)
    prompt = "zs_freeze" if m.group(1) else "zs"
    body = stem[: m.start()]
    if body.startswith("ForecastBench."):          # 2026 naming: ForecastBench.<Org>.<Org>.<model>-run-variant-NN
        body = body.split(".", 3)[3]
    else:                                           # older naming: <Org>.<model>
        body = body.split(".", 1)[1]
    run = re.search(r"-run-variant-(\d+)$", body)
    base = body[: run.start()] if run else body
    return base, int(run.group(1)) if run else 1, prompt


def day(s):
    return date.fromisoformat(s[:10])


rows = 0
out = os.path.join(ROOT, "derived", "panel_rounds.csv.gz")
cols = ["round", "item", "qtype", "source", "horizon", "outcome", "mkt_value", "org", "base_model",
        "run", "prompt", "config", "forecast"]
with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for rnd in sorted(os.listdir(lp(R))):
        due = day(rnd)
        files = sorted(f for f in os.listdir(lp(os.path.join(R, rnd))) if PROMPT.search(f) and not SKIP.search(f))
        for fn in files:
            base, run, prompt = parse_name(rnd, fn)
            with open(lp(os.path.join(R, rnd, fn)), encoding="utf-8") as f:
                d = json.load(f)
            org, config = d.get("model_organization", ""), d.get("model", "")
            for x in d["forecasts"]:
                if isinstance(x["id"], list) or x.get("imputed") or x["forecast"] is None:
                    continue
                if not x["resolved"] or x["resolved_to"] not in (0, 1, 0.0, 1.0):
                    continue
                mkt = x["source"] in MARKET
                hz = (day(x["resolution_date"]) - due).days
                item = f"{x['id']}|mkt" if mkt else f"{x['id']}|{x['resolution_date'][:10]}"
                w.writerow([rnd, item, "market" if mkt else "dataset", x["source"], hz, float(x["resolved_to"]),
                            x.get("market_value_on_due_date") if mkt else "", org, base, run, prompt, config,
                            float(x["forecast"])])
                rows += 1
        print(rnd, len(files), "files", flush=True)
print("rows", rows)
