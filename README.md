# Predictive Maintenance with NASA CMAPSS

[![CI](https://github.com/eastani/predictive-maintenance-cmapss/actions/workflows/ci.yml/badge.svg)](https://github.com/eastani/predictive-maintenance-cmapss/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

End-to-end predictive maintenance pipeline for industrial rotating equipment,
demonstrated on NASA's **CMAPSS** turbofan engine degradation dataset. From
raw sensor ingestion to remaining useful life (RUL) prediction, benchmarked
notebooks, and production-style Python package structure.

> Built as a public reference implementation. The architecture mirrors the
> patterns used in real industrial IIoT platforms - separating data
> contracts, feature engineering, model lifecycle, and serving so each layer
> can be tested and replaced independently.

---

## At a glance

| Area | What is included |
| ---- | ---------------- |
| Dataset | NASA CMAPSS FD001-FD004 turbofan run-to-failure data |
| Pipeline | Strict schema validation, RUL labelling, rolling features, model evaluation |
| Models | Ridge baseline and XGBoost RUL regressor on identical features |
| Evidence | Executed notebooks with RMSE, asymmetric S-score, and feature diagnostics |
| Engineering | Importable `pdm` package, pytest coverage reporting, Ruff, GitHub Actions CI, `uv` lockfile |
| Next step | Multi-regime FD002/FD004 evaluation, dashboard, and serving API |

## Why this project

Industrial predictive maintenance combines three problems that are usually
treated separately:

1. **Data engineering** - heterogeneous, irregularly sampled sensor streams
   per asset, with multiple operating regimes and censored failure data.
2. **Modelling** - survival-style RUL regression where labels are noisy,
   right-censored, and unevenly distributed.
3. **Operationalisation** - alerts must be calibrated, traceable, and tied
   back to specific assets and time windows for the maintenance team to act.

This repository tackles all three on a public dataset, with a code structure
that reflects how the same pipeline would be deployed against live OPC-UA /
historian data.

## Architecture

```
raw sensor files
FD001-FD004
      |
      v
+-----------------------------+
| pdm.data                    |
| - CMAPSS loader             |
| - train/test split          |
| - contract validation       |
| - RUL labelling             |
+-------------+---------------+
              |
              v
+-----------------------------+
| pdm.features                |
| - constant-sensor filter    |
| - rolling statistics        |
| - per-unit windows          |
| - normalisation             |
+-------------+---------------+
              |
              v
+-----------------------------+
| pdm.models                  |
| - Ridge baseline            |
| - XGBoost RUL regressor     |
| - evaluation metrics        |
+-------------+---------------+
              |
              v
+-------------+---------------+----------------+
| tests + CI  | notebooks     | dashboard/API   |
| pytest      | benchmarks    | roadmap items   |
+-------------+---------------+----------------+
```

## Dataset

NASA's [Commercial Modular Aero-Propulsion System Simulation (CMAPSS)][cmapss]
contains four sub-datasets (FD001-FD004) of run-to-failure trajectories
for simulated turbofan engines under varying operating conditions and fault
modes.

| Subset | Train units | Test units | Operating conditions | Fault modes |
| ------ | ----------- | ---------- | -------------------- | ----------- |
| FD001  | 100         | 100        | 1                    | 1 (HPC)     |
| FD002  | 260         | 259        | 6                    | 1 (HPC)     |
| FD003  | 100         | 100        | 1                    | 2 (HPC, Fan)|
| FD004  | 248         | 249        | 6                    | 2 (HPC, Fan)|

Each row contains 21 sensor channels plus 3 operational settings, indexed by
unit number and operating cycle. The training trajectories run until failure;
the test trajectories are truncated and the goal is to predict the
remaining useful life (RUL) of each test unit.

[cmapss]: https://www.nasa.gov/intelligent-systems-division/

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/eastani/predictive-maintenance-cmapss.git
cd predictive-maintenance-cmapss
uv sync

# 2. Download the CMAPSS dataset (places files into data/raw/)
./scripts/download_data.sh

# 3. Run the test suite
uv run pytest

# 4. Launch the EDA notebook
uv run jupyter lab notebooks/01_eda.ipynb
```

## Project layout

```
predictive-maintenance-cmapss/
|-- src/pdm/               # Library code (importable as `pdm`)
|   |-- data.py            # CMAPSS loader + RUL labelling
|   |-- features.py        # Rolling statistics and feature engineering
|   `-- models.py          # RUL regression models and metrics
|-- tests/                 # pytest unit and integration tests
|-- notebooks/             # Exploratory and benchmark notebooks
|-- scripts/               # Data download and operational helpers
|-- data/raw/              # Untracked; CMAPSS files land here
`-- .github/workflows/     # CI pipeline
```

## Notebooks

* [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) - Exploratory data
  analysis on FD001: trajectory length distribution, sensor variance,
  per-unit degradation curves, alignment to failure, and the empirical
  motivation for piecewise-linear RUL capping.
* [`notebooks/02_baseline_rul.ipynb`](notebooks/02_baseline_rul.ipynb)
  - Baseline RUL regressor on FD001 using rolling features and a
  standard-scaled Ridge regression, evaluated with RMSE and the
  asymmetric CMAPSS S-score. Establishes the floor that subsequent
  models must demonstrably beat.
* [`notebooks/03_xgboost_rul.ipynb`](notebooks/03_xgboost_rul.ipynb)
  - Benchmarks an XGBoost gradient-boosted regressor against the Ridge
  baseline on identical features. Shows the honest result that on
  FD001's single regime / single fault mode, the gap is narrow, and
  uses feature-importance diagnostics to corroborate the EDA findings.

The notebooks are kept paired with `.py` files in the
[jupytext percent format][jupytext], so diffs are reviewable on GitHub.

[jupytext]: https://jupytext.readthedocs.io/

## Roadmap

- [x] CMAPSS data loader with strict schema validation
- [x] Project skeleton, CI, and packaging
- [x] Feature engineering: constant-sensor filter, piecewise-linear RUL,
      rolling statistics
- [x] Exploratory data analysis notebook
- [x] Baseline RUL regressor (Ridge regression) with RMSE / S-score evaluation
- [x] Gradient-boosted RUL regressor (XGBoost) with feature-importance diagnostics
- [ ] Operating-regime clustering for FD002 / FD004
- [ ] Cross-subset evaluation showing where XGBoost actually wins
- [ ] LSTM sequence model with proper truncation handling
- [ ] Plotly Dash live dashboard
- [ ] Dockerised serving with a minimal REST API
- [ ] Documentation site (MkDocs Material)

## License

[MIT](LICENSE) - feel free to use this as a starting point for your own
predictive maintenance projects.

## Author

**Naoya Higashitani** -
[LinkedIn](https://www.linkedin.com/in/naoya-higashitani/) |
[Portfolio](https://eastani.github.io/portfolio/) |
[GitHub](https://github.com/eastani)
