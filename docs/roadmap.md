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
- Preliminary FD002 repeated-run LSTM benchmark.
- LSTM prediction diagnostics with early-vs-late error breakdowns.
- RUL-band diagnostics showing FD002 high-RUL compression.
- Raw-vs-capped target diagnostics for LSTM predictions.
- Raw-vs-capped target diagnostics for tabular Ridge and XGBoost predictions.
- MkDocs Material documentation site.

## Next

### Tune And Scale LSTM Evaluation

Run larger repeated-seed LSTM experiments beyond the preliminary FD002 result.
FD002 and FD004 remain the most important targets because the current XGBoost
results suggest that non-linear interactions matter most under multiple
operating regimes.

The critical bar is not a single best run. Report mean, standard deviation,
exact configuration, and training-window density. The first FD002 run used
`stride=10` to keep CPU time reasonable; a serious comparison should evaluate
denser windows and more epochs.

The first diagnostic readout shows the preliminary FD002 LSTM underestimates
healthy high-RUL units. The RUL-band view narrows this to the `125+` band, where
raw test RUL reaches 194 while the model never predicts above roughly 119.
Target-cap diagnostics show that capped-125 scoring improves all measured FD002
models, including Ridge and XGBoost. The next tuning pass should therefore
report raw and capped-label metrics side by side, then improve the LSTM only if
it closes the gap under both conventions.

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
