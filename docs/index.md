# CMAPSS Predictive Maintenance

This project is an end-to-end predictive maintenance reference pipeline for
NASA CMAPSS turbofan degradation data. It covers strict data loading, feature
engineering, tabular and sequence modelling, benchmark reporting, a FastAPI
serving boundary, and dashboard inspection.

The goal is not to claim a single model victory. The goal is to make the
modelling tradeoffs inspectable: when extra model capacity helps, when it
hurts the asymmetric S-score, and how preprocessing choices affect the
multi-condition subsets.

## Key Findings

| Finding | Evidence |
| ------- | -------- |
| Model capacity is conditional | XGBoost improves RMSE across all subsets but worsens S-score on FD001 and FD003. |
| Multi-condition subsets need regime handling | FD002 and FD004 improve with operating-regime features, but residuals still vary by regime. |
| Target convention affects conclusions | FD002 and FD004 scores shift substantially when raw test labels are capped at 125 cycles. |
| Error direction matters | FD002 S-score is mostly early-prediction cost; FD004 has a larger late-prediction share. |

## What Is Included

| Area | Implementation |
| ---- | -------------- |
| Data contract | Schema-validated CMAPSS loader for FD001-FD004 |
| Feature engineering | Constant-sensor filtering, RUL clipping, rolling statistics |
| Regime handling | Operating-regime clustering and per-regime sensor normalization |
| Models | Ridge, XGBoost, optional PyTorch LSTM |
| Evaluation | RMSE, CMAPSS S-score, cross-subset and ablation scripts |
| Operations | FastAPI inference service, Dockerfile, model artifact helpers |
| Visualization | Plotly Dash benchmark dashboard |
| Engineering | `uv`, Ruff, mypy, pytest, GitHub Actions CI |

## Result Preview

![FD001 sensor 11 trajectories aligned to failure](assets/fd001_sensor11_aligned_to_failure.png)

Sensor 11 shows visible drift as units approach failure, supporting the
piecewise RUL relabelling and rolling-window feature strategy.

![FD001 Ridge baseline vs XGBoost prediction scatter](assets/fd001_ridge_vs_xgboost_predictions.png)

FD001 is deliberately reported as a close Ridge-vs-XGBoost comparison, not a
headline-only model win.

![FD001 XGBoost feature importance](assets/fd001_xgboost_feature_importance.png)

The most important XGBoost features align with the degradation signals surfaced
by EDA, which gives the model results a basic domain sanity check.

## Reading Path

Start with [Getting Started](getting-started.md) to reproduce the environment.
Then read [Methodology](methodology.md) before interpreting the numbers in
[Benchmarks](benchmarks.md). The [Operations](operations.md) page describes the
serving API and dashboard boundary.
