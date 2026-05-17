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

RUL-band diagnostics make the failure mode sharper:

| Seed | RUL band | n | RMSE | S-score | Mean error | Mean abs error | Max abs error |
| ---: | -------- | -: | ---: | ------: | ---------: | -------------: | ------------: |
| 42 | 0-50 | 88 | 24.21 | 1,896.77 | 19.86 | 20.24 | 58.64 |
| 42 | 50-100 | 79 | 19.47 | 707.88 | 7.69 | 16.07 | 49.20 |
| 42 | 100-125 | 35 | 16.75 | 107.68 | -13.22 | 13.90 | 39.43 |
| 42 | 125+ | 57 | 60.46 | 16,217.08 | -56.75 | 56.75 | 105.22 |
| 43 | 0-50 | 88 | 16.70 | 904.94 | 8.38 | 12.04 | 59.93 |
| 43 | 50-100 | 79 | 23.61 | 1,322.80 | 9.02 | 19.56 | 55.64 |
| 43 | 100-125 | 35 | 15.96 | 111.71 | -9.81 | 11.92 | 44.16 |
| 43 | 125+ | 57 | 55.75 | 11,238.56 | -51.48 | 51.48 | 98.36 |

The model's maximum predictions were 113.01 and 118.59 cycles for the two
seeds, while FD002 test labels reach 194 cycles. That is consistent with a
target-design issue: training RUL is clipped at 125, but the headline benchmark
uses raw test RUL. The next experiment should compare raw-label evaluation with
capped-label evaluation before assuming the architecture itself is the primary
cause.

Scoring the same predictions against a capped target confirms the target-design
effect:

| Seed | Target convention | n | RMSE | S-score | Mean error | Mean abs error | Max abs error |
| ---: | ----------------- | -: | ---: | ------: | ---------: | -------------: | ------------: |
| 42 | raw | 259 | 34.02 | 18,929.41 | -5.18 | 26.15 | 105.22 |
| 42 | cap_125 | 259 | 21.98 | 3,117.45 | 2.31 | 18.65 | 58.64 |
| 43 | raw | 259 | 31.35 | 13,578.00 | -7.05 | 23.00 | 98.36 |
| 43 | cap_125 | 259 | 19.60 | 2,591.30 | 0.44 | 15.50 | 59.93 |

This does not make the LSTM better than XGBoost; it changes the interpretation
of the failure. Under raw labels, the preliminary LSTM is punished heavily for
not predicting above the training cap. Under capped labels, the remaining error
is much closer to the tabular models. A fair next experiment should report both
raw and capped-label metrics explicitly.

The same raw-vs-capped diagnostic is now available for the tabular benchmark.
On FD002, the capped convention improves every model, so this is not an
LSTM-specific excuse. XGBoost remains the strongest measured model under both
target conventions:

![Raw vs capped target convention RMSE](assets/diagnostic_target_conventions.svg)

| Model | Run | Raw RMSE | Raw S-score | Cap-125 RMSE | Cap-125 S-score |
| ----- | --- | -------: | ----------: | -----------: | --------------: |
| Ridge | deterministic | 29.72 | 15,282.53 | 17.54 | 1,427.70 |
| XGBoost | deterministic | 28.21 | 11,269.47 | 15.65 | 1,268.18 |
| LSTM | seed 42 | 34.02 | 18,929.41 | 21.98 | 3,117.45 |
| LSTM | seed 43 | 31.35 | 13,578.00 | 19.60 | 2,591.30 |

The stricter conclusion is that raw-label FD002 scores partially measure
target-convention mismatch. Capped-label scoring removes much of the high-RUL
penalty, but it does not overturn the model ranking in the current experiment.

FD004 adds an important counterexample. XGBoost is still better on raw RMSE and
raw S-score, and it has better capped RMSE, but Ridge has the slightly better
capped S-score:

| Subset | Model | Raw RMSE | Raw S-score | Cap-125 RMSE | Cap-125 S-score |
| ------ | ----- | -------: | ----------: | -----------: | --------------: |
| FD002 | Ridge | 29.72 | 15,282.53 | 17.54 | 1,427.70 |
| FD002 | XGBoost | 28.21 | 11,269.47 | 15.65 | 1,268.18 |
| FD004 | Ridge | 30.68 | 6,946.85 | 19.79 | 2,085.20 |
| FD004 | XGBoost | 28.92 | 5,912.41 | 17.90 | 2,219.97 |

Operating-regime diagnostics make the same point at a finer level. On FD002,
XGBoost improves most regimes but still leaves RMSE above 30 in regimes 1 and
3. On FD004, regime 3 is the easiest segment for both models, while regimes 0
and 1 remain the hardest. XGBoost is not uniformly better in every regime, so
model choice should stay tied to both the target convention and the cost of late
predictions.

![Operating-regime RMSE ranges](assets/diagnostic_regime_rmse_ranges.svg)

| Subset | Model | Best regime RMSE | Worst regime RMSE | Best regime S-score | Worst regime S-score |
| ------ | ----- | ---------------: | ----------------: | ------------------: | -------------------: |
| FD002 | Ridge | 26.71 | 32.76 | 1,372.17 | 4,473.85 |
| FD002 | XGBoost | 24.26 | 31.34 | 914.70 | 3,554.65 |
| FD004 | Ridge | 23.86 | 34.65 | 473.58 | 2,415.15 |
| FD004 | XGBoost | 23.64 | 31.30 | 528.73 | 1,747.91 |

The S-score contribution split is another guardrail against overclaiming. On
FD002, more than 90% of the raw S-score comes from early predictions, so the
dominant error is over-conservative high-RUL underprediction rather than
near-failure optimism. FD004 has a larger late-prediction contribution,
especially for XGBoost:

![S-score contribution by error direction](assets/diagnostic_s_score_contributions.svg)

| Subset | Model | Early S-score share | Late S-score share | Early n | Late n |
| ------ | ----- | ------------------: | -----------------: | ------: | -----: |
| FD002 | Ridge | 93.44% | 6.56% | 136 | 123 |
| FD002 | XGBoost | 91.45% | 8.55% | 153 | 106 |
| FD004 | Ridge | 76.19% | 23.81% | 128 | 120 |
| FD004 | XGBoost | 69.00% | 31.00% | 130 | 118 |

That distinction matters operationally: a model dominated by early S-score
costs can waste maintenance capacity, while a model with higher late
contribution needs closer safety review even if its headline RMSE is better.

## Reproduce

```bash
uv run python scripts/evaluate_subsets.py \
  --data-dir data/raw \
  --with-xgboost \
  --out reports/cross_subset_results.csv

uv run python scripts/evaluate_subsets.py \
  --data-dir data/raw \
  --subsets FD002 \
  --with-xgboost \
  --out reports/fd002_tabular_target_conventions_raw.csv \
  --predictions-out reports/fd002_tabular_predictions.csv \
  --target-cap-diagnostics-out reports/fd002_tabular_target_cap_diagnostics.csv \
  --s-score-diagnostics-out reports/fd002_tabular_s_score_diagnostics.csv \
  --regime-diagnostics-out reports/fd002_tabular_regime_diagnostics.csv

uv run python scripts/evaluate_subsets.py \
  --data-dir data/raw \
  --subsets FD004 \
  --with-xgboost \
  --out reports/fd004_tabular_target_conventions_raw.csv \
  --predictions-out reports/fd004_tabular_predictions.csv \
  --target-cap-diagnostics-out reports/fd004_tabular_target_cap_diagnostics.csv \
  --s-score-diagnostics-out reports/fd004_tabular_s_score_diagnostics.csv \
  --regime-diagnostics-out reports/fd004_tabular_regime_diagnostics.csv

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
  --diagnostics-out reports/lstm_fd002_3epoch_stride10_diagnostics.csv \
  --rul-band-diagnostics-out reports/lstm_fd002_3epoch_stride10_rul_band_diagnostics.csv \
  --target-cap-diagnostics-out reports/lstm_fd002_3epoch_stride10_target_cap_diagnostics.csv
```
