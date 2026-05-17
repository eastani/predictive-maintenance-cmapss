# Getting Started

## Install

```bash
git clone https://github.com/eastani/predictive-maintenance-cmapss.git
cd predictive-maintenance-cmapss
uv sync --extra dev --extra boost --extra viz
```

The optional LSTM baseline uses PyTorch:

```bash
uv sync --extra dev --extra boost --extra viz --extra deep
```

## Download Data

```bash
./scripts/download_data.sh
```

The script places the CMAPSS raw files under `data/raw/`. The raw data is
excluded from git.

## Verify

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy src
uv run pytest
```

## Common Commands

```bash
# Cross-subset Ridge baseline
uv run python scripts/evaluate_subsets.py --data-dir data/raw

# Ridge + XGBoost benchmark
uv run python scripts/evaluate_subsets.py \
  --data-dir data/raw \
  --with-xgboost \
  --out reports/cross_subset_results.csv

# LSTM repeated-seed benchmark on FD001
uv run python scripts/evaluate_lstm.py \
  --data-dir data/raw \
  --subsets FD001 \
  --sequence-length 30 \
  --hidden-size 32 \
  --epochs 5 \
  --seeds 42 43 \
  --out reports/lstm_results.csv
```

The tabular benchmark writes a reproducibility metadata file next to the main
CSV, such as `reports/cross_subset_results_metadata.json`. It captures the
command arguments, git commit, dependency versions, model settings, and output
paths for the run.

## Build The Docs

```bash
uv sync --extra docs
uv run mkdocs serve
```

For CI-style validation:

```bash
uv run mkdocs build --strict
```
