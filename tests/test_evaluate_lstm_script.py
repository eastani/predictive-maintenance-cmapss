"""Tests for the LSTM evaluation script helpers."""

from __future__ import annotations

import pandas as pd
from scripts.evaluate_lstm import _regime_options, summarize_lstm_results


class TestRegimeOptions:
    def test_auto_enables_regimes_only_for_multi_condition_subsets(self) -> None:
        assert _regime_options("FD001", "auto") == [False]
        assert _regime_options("FD002", "auto") == [True]
        assert _regime_options("FD003", "auto") == [False]
        assert _regime_options("FD004", "auto") == [True]

    def test_forced_modes(self) -> None:
        assert _regime_options("FD001", "on") == [True]
        assert _regime_options("FD002", "off") == [False]
        assert _regime_options("FD004", "both") == [False, True]


class TestSummarizeLSTMResults:
    def test_aggregates_repeated_seed_metrics(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "subset": "FD001",
                    "model": "lstm",
                    "rmse": 20.0,
                    "s_score": 100.0,
                    "n_train_samples": 10,
                    "n_test_units": 2,
                    "n_features": 3,
                    "use_regime_features": False,
                    "sequence_length": 30,
                    "stride": 1,
                    "epochs": 5,
                    "seed": 1,
                },
                {
                    "subset": "FD001",
                    "model": "lstm",
                    "rmse": 22.0,
                    "s_score": 140.0,
                    "n_train_samples": 10,
                    "n_test_units": 2,
                    "n_features": 3,
                    "use_regime_features": False,
                    "sequence_length": 30,
                    "stride": 1,
                    "epochs": 5,
                    "seed": 2,
                },
            ]
        )

        summary = summarize_lstm_results(results)

        assert summary.loc[0, "runs"] == 2
        assert summary.loc[0, "rmse_mean"] == 21.0
        assert summary.loc[0, "s_score_mean"] == 120.0
        assert summary.loc[0, "n_train_samples"] == 10
