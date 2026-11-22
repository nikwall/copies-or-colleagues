# Copies or Colleagues? Shared Error among AI Models and Human Forecasters

Replication package for the working paper by Niklas Wallmeier (Nieman Foundation at Harvard University and arq decisions),
version 1, November 2026. The paper replies to Imas, Koh and Legg (2026), *Organizing Intelligence* (arXiv:2611.05417).

- Paper: [`paper/wallmeier_copies_v1.pdf`](paper/wallmeier_copies_v1.pdf)
- Summary, 30-second video and slides: <https://research.nikwall.de/papers/copies-or-colleagues/>

## What the paper does

A prominent account of when AI transforms the economy rests on the premise that a team of people can beat a more capable AI,
because people err in different directions while copies of a model do not. The paper tests this premise on
[ForecastBench](https://www.forecastbench.org), where people and machines forecast the same questions before the outcomes are known.
It measures how much of each forecaster's squared error the rest of a group shares, the part that averaging cannot remove.
Models from different developers share 86 percent, members of the public 70 percent, and the best tool-using tournament system
shares 93 percent with the best plain model. People kept the lead only where institutions pooled their information, in a
superforecaster team that discussed its forecasts and in prediction markets.

## Replicate every number

Requirements: Python 3.12 with pandas and numpy (`pip install -r requirements.txt`), about 5 GB of free disk space.

```bash
python -I code/get_data.py .
python -I run_all.py
```

`get_data.py` downloads the ForecastBench snapshot of 17 November 2026 from release
[v1.0](https://github.com/nikwall/copies-or-colleagues/releases/tag/v1.0) of this repository (141 MB, SHA-256 checked) and unpacks
it to `data/raw/forecastbench/`. `run_all.py` then runs the scripts in `code/` in order and writes all outputs to `derived/`,
including the console output of each script as `derived/log_<script>.txt`. The outputs behind version 1 of the paper are in
`results/` for comparison. The bootstrap uses a fixed seed, so the numbers should match exactly.

| Script | Paper |
|---|---|
| `build_matrix_2024.py`, `validate_2024.py` | 2024 round, 578 items; check against ForecastBench's human medians (Section 4.6) |
| `analysis_h1_h4_2024.py`, `supplement_2024.py` | H1 to H4: Tables 4 and 5, Figure 1, Appendix Table 9 (Panel A) |
| `bootstrap_2024.py` | Question-level bootstrap for H3 and H4 (Section 5.2, Table 5) |
| `build_panel_rounds.py`, `analysis_h5.py`, `make_figure_data.py` | H5: Table 6, Figure 2, Appendix Table 9 (Panel B) |
| `analysis_h6.py` | H6: Table 7 |
| `build_panel_external.py`, `analysis_h7_tournament.py` | H7 and the market-price comparison: Table 8, Section 5.5 |

Some ForecastBench file names exceed the 260-character path limit of Windows. `code/longpath.py` handles this with the `\\?\`
prefix, and the scripts work unchanged on macOS and Linux.

To build the paper, run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` on `paper/wallmeier_copies_v1.tex`.

## Data

All data come from ForecastBench (Karger et al., 2025, *ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities*, ICLR),
published by the Forecasting Research Institute under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

- `data/raw/forecastbench/human_2024/`: the individual forecasts of the 2024-07-21 human round (500 members of the public,
  40 superforecasters) and the human and model question sets of that round, from
  [forecastingresearch/forecastbench-datasets](https://github.com/forecastingresearch/forecastbench-datasets). Checksums in
  `data/MANIFEST_sha256.txt`.
- Release v1.0: `processed_forecast_sets_2026-11-17.tar.gz`, the processed forecast sets as downloaded from forecastbench.org on
  17 November 2026 (40 rounds, 2024-07-21 to 2026-10-25, resolutions as of that day).

The data files in this repository and in its releases are redistributed unchanged under CC BY-SA 4.0 (see `LICENSE-DATA.md`).

## License and citation

Code: MIT (`LICENSE`). Data: CC BY-SA 4.0, © Forecasting Research Institute (`LICENSE-DATA.md`). The paper itself is © the author.

Please cite the paper (see `CITATION.cff`):

> Wallmeier, N. (2026). Copies or Colleagues? Shared Error among AI Models and Human Forecasters. Working paper, version 1.

The code was written with the help of generative AI (Claude, Anthropic). The statement at the end of the paper sets out who did what.
Questions and comments: niklas.wallmeier@gmail.com.
