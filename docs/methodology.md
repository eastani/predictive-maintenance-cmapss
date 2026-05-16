# Methodology

## Dataset

CMAPSS contains four turbofan run-to-failure subsets:

| Subset | Train units | Test units | Operating conditions | Fault modes |
| ------ | ----------- | ---------- | -------------------- | ----------- |
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 248 | 249 | 6 | 2 |

The training trajectories run until failure. The test trajectories are
truncated, and the label is the remaining useful life at the final observed
cycle for each test engine.

## Feature Engineering

The tabular benchmark uses the same feature contract for Ridge and XGBoost:

- Drop near-constant sensor channels.
- Clip training RUL to the standard piecewise-linear cap.
- Add per-unit rolling mean and standard deviation features.
- Keep evaluation aligned to one labelled final-cycle row per test unit.

This keeps the model comparison focused on estimator capacity rather than
changing the input data between models.

## Operating Regimes

FD002 and FD004 contain multiple operating conditions. The pipeline fits
K-means on standardized operational settings and appends per-regime normalized
sensor features. The standardized settings matter because the operational
settings live on different numeric scales.

The regime-aware features are enabled automatically for FD002 and FD004 in the
cross-subset script, and can be forced on or off for ablation.

## Sequence Modelling

The LSTM path uses a separate sequence dataset builder:

- Training windows end at every labelled training cycle.
- Test windows end only at each unit's final observed test cycle.
- Early windows are left-padded, and valid lengths are carried forward.
- The PyTorch model uses packed sequences so padding is not treated as real
  sensor history.
- Valid timesteps are standardized without mixing padding into the scaler.
- RUL is standardized during optimization and unscaled for prediction.

This avoids a common CMAPSS error: evaluating every truncated test cycle as if
it had a label.

## Metrics

The project reports both RMSE and the official asymmetric CMAPSS S-score.
S-score penalizes late predictions more heavily than early predictions, which
matches the operational risk of overestimating remaining life.

RMSE improvements alone are therefore not enough. A model can reduce RMSE and
still become worse under S-score.
