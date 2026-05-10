# Predictive Maintenance with NASA CMAPSS

[![CI](https://github.com/eastani/predictive-maintenance-cmapss/actions/workflows/ci.yml/badge.svg)](https://github.com/eastani/predictive-maintenance-cmapss/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

End-to-end predictive maintenance pipeline for industrial rotating equipment,
demonstrated on NASA's **CMAPSS** turbofan engine degradation dataset. From
raw sensor ingestion to remaining useful life (RUL) prediction and a live
health-monitoring dashboard.

> Built as a public reference implementation. The architecture mirrors the
> patterns used in real industrial IIoT platforms — separating data
> contracts, feature engineering, model lifecycle, and serving so each layer
> can be tested and replaced independently.

---

## Why this project

Industrial predictive maintenance combines three problems that are usually
treated separately:

1. **Data engineering** — heterogeneous, irregularly sampled sensor streams
   per asset, with multiple operating regimes and censored failure data.
2. **Modelling** — survival-style RUL regression where labels are noisy,
   right-censored, and unevenly distributed.
3. **Operationalisation** — alerts must be calibrated, traceable, and tied
   back to specific assets and time windows for the maintenance team to act.

This repository tackles all three on a public dataset, with a code structure
that reflects how the same pipeline would be deployed against live OPC-UA /
historian data.

## Architecture

```
                          ┌─────────────────────┐
   raw sensor files       │   data/raw/         │
   (FD001 … FD004)        └─────────┬───────────┘
                                    │
                       ┌────────────▼────────────┐
                       │ pdm.data                │   ← CMAPSS loader,
                       │ • train/test split      │     contract validation
                       │ • RUL labelling         │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │ pdm.features            │   ← rolling stats,
                       │ • per-unit windows      │     regime clustering,
                       │ • normalisation         │     PCA health index
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │ pdm.models              │   ← XGBoost baseline,
                       │ • RUL regressor         │     LSTM sequence model,
                       │ • health classifier     │     ensemble
                       └────────────┬────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
     ┌────────▼────────┐  ┌─────────▼─────────┐  ┌────────▼─────────┐
     │ tests/          │  │ dashboard/        │  │ docs/            │
     │ pytest + CI     │  │ Plotly Dash       │  │ MkDocs site      │
     └─────────────────┘  └───────────────────┘  └──────────────────┘
```

## Dataset

NASA's [Commercial Modular Aero-Propulsion System Simulation (CMAPSS)][cmapss]
contains four sub-datasets (FD001 – FD004) of run-to-failure trajectories
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
├── src/pdm/               # Library code (importable as `pdm`)
│   ├── data.py            # CMAPSS loader + RUL labelling
│   ├── features.py        # Rolling statistics, regime clustering
│   ├── models.py          # RUL regression models
│   └── evaluation.py      # Metrics: RMSE, S-score, alert precision
├── tests/                 # pytest unit + integration tests
├── notebooks/             # Exploratory notebooks (one per stage)
├── dashboard/             # Plotly Dash live monitoring app
├── scripts/               # Data download, training, deployment
├── data/raw/              # Untracked — CMAPSS files land here
└── .github/workflows/     # CI pipeline (lint, test, build)
```

## Roadmap

- [x] CMAPSS data loader with strict schema validation
- [x] Project skeleton, CI, and packaging
- [ ] Feature engineering: rolling statistics + operating-regime clustering
- [ ] Baseline RUL regressor (XGBoost) with cross-validation
- [ ] LSTM sequence model with proper truncation handling
- [ ] Plotly Dash live dashboard
- [ ] Dockerised serving with a minimal REST API
- [ ] Documentation site (MkDocs Material)

## License

[MIT](LICENSE) — feel free to use this as a starting point for your own
predictive maintenance projects.

## Author

**Naoya Higashitani** —
[LinkedIn](https://www.linkedin.com/in/naoya-higashitani/) ·
[Portfolio](https://eastani.github.io/portfolio/) ·
[GitHub](https://github.com/eastani)
