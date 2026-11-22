"""Reproduce every number in the paper.

Runs the scripts in code/ in order on the data in data/ and writes all outputs to derived/.
The console output of each script is saved as derived/log_<script>.txt. Compare derived/ with results/,
which holds the outputs behind version 1 of the paper.

Usage:
    python -I code/get_data.py .     # once: download and unpack the ForecastBench snapshot (141 MB, 4.2 GB unpacked)
    python -I run_all.py             # about 10 minutes on a laptop
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("build_matrix_2024", "2024 round: humans and models on the 578 resolved items"),
    ("validate_2024", "check against ForecastBench's own human medians"),
    ("analysis_h1_h4_2024", "H1 to H4 (Tables 4 and 5, Figure 1)"),
    ("supplement_2024", "robustness of H1 (Appendix Table 10, Panel A)"),
    ("bootstrap_2024", "question-level bootstrap for H3 and H4"),
    ("build_panel_rounds", "all rounds: models run by ForecastBench"),
    ("build_panel_external", "all rounds: tournament entries by outside teams"),
    ("analysis_h5", "H5 (Table 6, Figure 2, Appendix Table 10, Panel B)"),
    ("analysis_h6", "H6 (Table 7)"),
    ("analysis_h7_tournament", "H7 and the market-price comparison (Table 8)"),
    ("analysis_noise_parity", "bound on outcome noise (Table 9) and the parity teams (Table 8, Panel C)"),
    ("make_figure_data", "paper/figures/h5_rounds.dat"),
]

if not os.path.isdir(os.path.join(ROOT, "data", "raw", "forecastbench", "forecastbench-processed-forecast-sets")):
    sys.exit("Data missing. Run first:  python -I code/get_data.py .")
os.makedirs(os.path.join(ROOT, "derived"), exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8")
for name, what in STEPS:
    print(f"== {name}: {what}", flush=True)
    r = subprocess.run([sys.executable, "-I", "-W", "ignore", os.path.join(ROOT, "code", name + ".py"), ROOT],
                       capture_output=True, text=True, encoding="utf-8", env=env)
    with open(os.path.join(ROOT, "derived", f"log_{name}.txt"), "w", encoding="utf-8") as f:
        f.write(r.stdout)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:])
        sys.exit(f"{name} failed")
print("done: outputs in derived/")
