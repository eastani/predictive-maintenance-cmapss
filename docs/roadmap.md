# Roadmap

## Completed

- CMAPSS data loader with strict schema validation.
- Project skeleton, CI, package metadata, and `uv` workflow.
- Feature engineering for constant sensors, RUL clipping, and rolling stats.
- Exploratory notebook assets and FD001 baseline notebooks.
- Ridge and XGBoost benchmarks with RMSE and S-score.
- Operating-regime clustering for FD002 and FD004.
- Dockerized FastAPI serving boundary.
- Cross-subset evaluation harness.
- Published benchmark and regime-ablation tables.
- Plotly Dash benchmark dashboard.
- Sequence-window dataset builder with final-cycle test handling.
- Optional PyTorch LSTM baseline with packed sequences.
- FD001 repeated-run LSTM benchmark summary.
- MkDocs Material documentation site.

## Next

### Expand LSTM Evaluation

Run repeated-seed LSTM experiments beyond FD001. FD002 and FD004 are the most
important next targets because the current XGBoost results suggest that
non-linear interactions matter most under multiple operating regimes.

The critical bar is not a single best run. Report mean, standard deviation,
and exact configuration.

### Improve Model Diagnostics

Useful additions:

- Prediction-error histograms by subset.
- Late-vs-early error breakdown for S-score.
- Per-regime residual analysis on FD002 and FD004.
- Learning curves for LSTM sequence length, hidden size, and epoch count.

### Deployment Maturity

The current API is a clean boundary, but a production-like story would need
model versioning, online feature calculation, drift checks, and alert policy
calibration.
