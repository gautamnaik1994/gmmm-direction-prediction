# Copilot instructions (gmmm_direction_prediction)

## Project shape
- This is a research/experimentation repo; the primary “source of truth” is Jupyter notebooks under `notebooks/`.
- Data flows: market/fundamental data collection → feature engineering (momentum/ADX/regimes/sector indices) → modelling (sklearn + PyTorch) → artifacts saved back into `data/` and `models/`.

## Where to look first
- Data acquisition + prep notebooks: `notebooks/data_prep/` (numbered, chronological experiments).
  - Examples: `01.funda_scraping.ipynb`, `02.stock_data*.ipynb`, `03.regime_detection*.ipynb`, `04/05/07.momentum_sig_gen*.ipynb`, `09.adx_william.ipynb`.
- Modelling notebooks: `notebooks/modelling/` (sklearn classifiers + a PyTorch experiment).
- Analysis notebook(s): `notebooks/analysis/`.

## Data & artifact conventions
- Raw and derived datasets live in `data/` (CSV + pickle + parquet). Notebooks commonly read/write using relative paths like `../../data/...`.
  - Keep this convention when editing notebooks so they run from their subfolder locations.
- Trained model weights are stored in `models/` (e.g. `models/simple_cnn.pth`). If adding new checkpoints, keep them in `models/` and reference them from the training notebook.

## External integration points
- Market data: notebooks use `yfinance` for downloading index/stock time series.
- Fundamentals/screening: notebooks hit `https://www.screener.in/...` via `httpx`; automation prototype exists at `scripts/web_scraping/fundamental_screener.py` using `playwright` + `playwright_stealth`.
  - The Playwright script launches Chromium with `headless=False` and takes a screenshot; it’s meant for interactive debugging.

## Repo-specific patterns to preserve
- Notebook naming is numbered (`01.`, `02.`, …) and grouped by purpose (`data_prep/`, `analysis/`, `modelling/`). Follow this pattern when adding new notebooks.
- Prefer repo-relative paths (or `pathlib.Path`) in any new Python scripts; avoid hard-coded absolute paths (there is at least one local absolute path in notebooks).

## When coding in this repo
- If a change is notebook-centric, implement it in the relevant notebook first; only extract shared utilities into a `.py` file when you see copy/paste across notebooks.
- For any code that writes artifacts, default to `data/` for intermediate datasets/features and `models/` for trained weights.
