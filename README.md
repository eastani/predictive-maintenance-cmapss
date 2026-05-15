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
| Multi-regime support | Operating-regime clustering and per-regime sensor normalization for FD002/FD004 |
| Serving | FastAPI inference service, model artifact format, Dockerfile, and API tests |
| Evidence | Executed notebooks with RMSE, asymmetric S-score, and feature diagnostics |
| Engineering | Importable `pdm` package, pytest coverage reporting, Ruff, GitHub Actions CI, `uv` lockfile |
| Next step | Cross-subset FD002/FD004 evaluation, dashboard, and model monitoring |

## Results preview

The figures below are exported from the executed notebooks in this repository.

### Sensor degradation aligned to failure

![FD001 sensor 11 trajectories aligned to failure](docs/assets/fd001_sensor11_aligned_to_failure.png)

Sensor 11 shows a visible drift pattern as units approach failure, which
supports the choice to focus model capacity on the observable degradation
window.

### Ridge baseline vs. XGBoost

![FD001 Ridge baseline vs XGBoost prediction scatter](docs/assets/fd001_ridge_vs_xgboost_predictions.png)

The FD001 benchmark compares Ridge and XGBoost on identical rolling features.
The result is intentionally reported as a close head-to-head rather than a
headline-only model win.

### Feature importance diagnostics

![FD001 XGBoost feature importance](docs/assets/fd001_xgboost_feature_importance.png)

The top XGBoost features line up with the high-pressure-compressor sensor
signals surfaced during exploratory analysis, giving the model results a
domain-level sanity check.

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
| - regime-aware normalisation|
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
| pdm.serving | pdm.api       | tests + CI      |
| artifacts   | FastAPI       | pytest/mypy     |
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

# 4. Train a local model artifact for the API
uv run python scripts/train_fd001_artifact.py --data-dir data/raw --out artifacts/fd001-ridge.joblib

# 5. Launch the inference API
PDM_MODEL_PATH=artifacts/fd001-ridge.joblib uv run uvicorn pdm.api:app --reload

# 6. Launch the EDA notebook
uv run jupyter lab notebooks/01_eda.ipynb
```

## Cross-Subset Evaluation

FD001 is intentionally simple: one operating condition and one fault mode.
FD002 and FD004 mix six operating conditions, so the evaluation script enables
operating-regime clustering and per-regime sensor normalization for those
subsets before fitting the same model interface.

```bash
# Ridge baseline across all four subsets
uv run python scripts/evaluate_subsets.py --data-dir data/raw

# Ridge + XGBoost, writing a CSV report
uv run python scripts/evaluate_subsets.py \
  --data-dir data/raw \
  --with-xgboost \
  --out reports/cross_subset_results.csv
```

The report includes RMSE, CMAPSS S-score, sample counts, feature counts, and
whether regime-aware features were enabled. This is the benchmark harness for
showing where model capacity matters, instead of claiming that XGBoost wins
everywhere.

Latest measured results:

| Subset | Model | RMSE | S-score | Features | Regime-aware |
| ------ | ----- | ---: | ------: | -------: | ------------ |
| FD001 | Ridge | 18.27 | 592.60 | 105 | No |
| FD001 | XGBoost | 18.23 | 814.84 | 105 | No |
| FD002 | Ridge | 29.72 | 15,282.53 | 294 | Yes |
| FD002 | XGBoost | 28.21 | 11,269.47 | 294 | Yes |
| FD003 | Ridge | 19.17 | 720.01 | 112 | No |
| FD003 | XGBoost | 18.72 | 1,412.19 | 112 | No |
| FD004 | Ridge | 30.68 | 6,946.85 | 294 | Yes |
| FD004 | XGBoost | 28.92 | 5,912.41 | 294 | Yes |

The pattern is the useful part: XGBoost improves RMSE across all subsets, but
only improves the asymmetric S-score on FD002 and FD004, where multiple
operating regimes make non-linear interactions more valuable. On FD001 and
FD003, the extra model capacity makes more costly late predictions even when
RMSE moves slightly lower.

## Serving API

The API serves engineered feature vectors against a persisted
`ModelArtifact`. This keeps the boundary explicit: ingestion and feature
engineering can evolve independently from the inference service.

```bash
# Build the container
docker build -t predictive-maintenance-cmapss .

# Run with a mounted model artifact
docker run --rm -p 8000:8000 \
  -e PDM_MODEL_PATH=/models/fd001-ridge.joblib \
  -v "$PWD/artifacts:/models:ro" \
  predictive-maintenance-cmapss

# Health check
curl http://localhost:8000/health
```

Schema example:

```bash
curl -X POST http://localhost:8000/predict-rul \
  -H "Content-Type: application/json" \
  -d '{"unit_id": 1, "cycle": 120, "features": {"sensor_02_mean_5": 0.1}}'
```

Real requests must provide every feature column stored in the trained
artifact. Missing feature values return a 400 response rather than silently
filling defaults.

## Project layout

```
predictive-maintenance-cmapss/
|-- src/pdm/               # Library code (importable as `pdm`)
|   |-- data.py            # CMAPSS loader + RUL labelling
|   |-- features.py        # Rolling statistics, regime features, normalization
|   |-- models.py          # RUL regression models and metrics
|   |-- serving.py         # Model artifact loading and prediction helpers
|   `-- api.py             # FastAPI inference service
|-- tests/                 # pytest unit and integration tests
|-- notebooks/             # Exploratory and benchmark notebooks
|-- scripts/               # Data download and operational helpers
|-- data/raw/              # Untracked; CMAPSS files land here
|-- reports/               # Optional generated benchmark outputs
|-- Dockerfile             # Minimal API container
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
- [x] Operating-regime clustering for FD002 / FD004
- [x] Dockerised serving with a minimal REST API
- [x] Cross-subset evaluation harness for FD001-FD004
- [ ] Published cross-subset result table showing where XGBoost actually wins
- [ ] LSTM sequence model with proper truncation handling
- [ ] Plotly Dash live dashboard
- [ ] Documentation site (MkDocs Material)

## License

[MIT](LICENSE) - feel free to use this as a starting point for your own
predictive maintenance projects.

## Author

**Naoya Higashitani** -
[LinkedIn](https://www.linkedin.com/in/naoya-higashitani/) |
[Portfolio](https://eastani.github.io/portfolio/) |
[GitHub](https://github.com/eastani)
