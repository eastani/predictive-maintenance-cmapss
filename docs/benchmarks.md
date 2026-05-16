# Benchmarks

## Cross-Subset Results

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

XGBoost improves RMSE across all subsets, but it only improves the asymmetric
S-score on FD002 and FD004. On FD001 and FD003, the extra model capacity
creates more costly late predictions despite slightly lower RMSE.

## Regime-Feature Ablation

| Subset | Model | Regime-aware | RMSE | S-score | Features |
| ------ | ----- | ------------ | ---: | ------: | -------: |
| FD002 | Ridge | No | 30.64 | 17,835.85 | 147 |
| FD002 | XGBoost | No | 30.05 | 12,840.71 | 147 |
| FD002 | Ridge | Yes | 29.72 | 15,282.53 | 294 |
| FD002 | XGBoost | Yes | 28.21 | 11,269.47 | 294 |
| FD004 | Ridge | No | 31.71 | 7,861.98 | 147 |
| FD004 | XGBoost | No | 31.49 | 8,825.88 | 147 |
| FD004 | Ridge | Yes | 30.68 | 6,946.85 | 294 |
| FD004 | XGBoost | Yes | 28.92 | 5,912.41 | 294 |

The ablation supports regime-aware preprocessing rather than assuming it. FD004
is the clearest case: XGBoost without regime-aware features lowers RMSE versus
Ridge but worsens S-score; adding regime-normalized features improves both.

## LSTM Repeated-Run Results

| Model | Subset | Sequence length | Stride | Epochs | Seeds | RMSE mean | RMSE std | S-score mean | S-score std |
| ----- | ------ | --------------: | -----: | -----: | ----: | --------: | -------: | -----------: | ----------: |
| LSTM | FD001 | 30 | 1 | 5 | 2 | 16.88 | 1.24 | 577.76 | 228.28 |
| LSTM | FD002 | 30 | 10 | 3 | 2 | 32.69 | 1.88 | 16,253.71 | 3,784.02 |

Single-seed FD001 runs ranged from RMSE 16.01 / S-score 416.34 to RMSE 17.76 /
S-score 739.18. The average result is promising, but the variance is too high
to claim a stable sequence-model advantage.

The FD002 run is deliberately smaller: `stride=10` reduces the training windows
to 5,491, compared with 10,854 at `stride=5`. Under that CPU-friendly setting,
the LSTM is not competitive with regime-aware Ridge or XGBoost. That result is
important because it prevents an unsupported "deep learning wins" conclusion.

## FD002 LSTM Error Diagnostics

The diagnostic export separates all, early-or-exact, and late predictions. For
the preliminary FD002 run, the largest failure mode is not excessive late
prediction. It is severe early prediction on high-RUL units:

| Seed | Segment | n | RMSE | S-score | Mean error | Mean abs error | Max abs error |
| ---: | ------- | -: | ---: | ------: | ---------: | -------------: | ------------: |
| 42 | all | 259 | 34.02 | 18,929.41 | -5.18 | 26.15 | 105.22 |
| 42 | early_or_exact | 121 | 43.05 | 16,387.37 | -33.53 | 33.53 | 105.22 |
| 42 | late | 138 | 23.39 | 2,542.04 | 19.67 | 19.67 | 58.64 |
| 43 | all | 259 | 31.35 | 13,578.00 | -7.05 | 23.00 | 98.36 |
| 43 | early_or_exact | 138 | 37.89 | 11,482.22 | -28.20 | 28.20 | 98.36 |
| 43 | late | 121 | 21.61 | 2,095.78 | 17.06 | 17.06 | 59.93 |

The worst cases are units with true RUL around 174-194 cycles that the model
predicts around 69-103 cycles. This suggests underfitting of the healthy
long-RUL regime under the reduced-window setting, not simply unsafe optimistic
predictions near failure.

## Reproduce

```bash
uv run python scripts/evaluate_subsets.py \
  --data-dir data/raw \
  --with-xgboost \
  --out reports/cross_subset_results.csv

uv run python scripts/evaluate_subsets.py \
  --data-dir data/raw \
  --subsets FD002 FD004 \
  --with-xgboost \
  --regime-mode both \
  --out reports/regime_ablation_fd002_fd004.csv

uv run python scripts/evaluate_lstm.py \
  --data-dir data/raw \
  --subsets FD001 \
  --sequence-length 30 \
  --hidden-size 32 \
  --epochs 5 \
  --seeds 42 43 \
  --out reports/lstm_fd001_5epoch_raw.csv \
  --summary-out reports/lstm_fd001_5epoch_summary.csv

uv run python scripts/evaluate_lstm.py \
  --data-dir data/raw \
  --subsets FD002 \
  --sequence-length 30 \
  --hidden-size 32 \
  --epochs 3 \
  --batch-size 512 \
  --stride 10 \
  --seeds 42 43 \
  --out reports/lstm_fd002_3epoch_stride10_raw.csv \
  --summary-out reports/lstm_fd002_3epoch_stride10_summary.csv \
  --predictions-out reports/lstm_fd002_3epoch_stride10_predictions.csv \
  --diagnostics-out reports/lstm_fd002_3epoch_stride10_diagnostics.csv
```
