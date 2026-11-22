"""Write paper/figures/h5_rounds.dat (pgfplots table) from derived/h5_rounds_main.csv.

Columns: round, decimal year t = year + (day of year - 1) / 365.25, shared share a among all
models, a among the prior top five, gain from mixing, number of models. A "gap" row with
nan values breaks the line between the 2024 round and the first 2025 round.

Usage: python -I make_figure_data.py <project folder>
"""
import sys

import pandas as pd

ROOT = sys.argv[1]
r = pd.read_csv(ROOT + "/derived/h5_rounds_main.csv")
due = pd.to_datetime(r["round"])
r["t"] = due.dt.year + (due.dt.dayofyear - 1) / 365.25
out = ["round t a a_top gain n_models"]
for i, x in r.iterrows():
    if i > 0 and due[i].year > 2024 and due[i - 1].year == 2024:
        out.append("gap 2024.9 nan nan nan nan")
    f = lambda v, k=4: "nan" if pd.isna(v) else f"{v:.{k}f}"  # noqa: E731
    out.append(f"{x['round']} {x.t:.4f} {f(x.a)} {f(x.a_top)} {f(x.gain)} {int(x.n_models)}")
open(ROOT + "/paper/figures/h5_rounds.dat", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out[:4]), "...", out[-1], sep="\n")
