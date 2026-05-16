"""Tests for the LSTM evaluation script helpers."""

from __future__ import annotations

import pandas as pd
from scripts.evaluate_lstm import (
    _regime_options,
    summarize_lstm_results,
    summarize_prediction_diagnostics,
    summarize_rul_band_diagnostics,
    summarize_target_cap_diagnostics,
)


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


class TestSummarizePredictionDiagnostics:
    def test_aggregates_by_seed(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "subset": "FD001",
                    "model": "lstm",
                    "seed": 1,
                    "unit_id": 1,
                    "end_cycle": 10,
                    "y_true": 10.0,
                    "y_pred": 12.0,
                    "error": 2.0,
                    "abs_error": 2.0,
                    "late": True,
                },
                {
                    "subset": "FD001",
                    "model": "lstm",
                    "seed": 1,
                    "unit_id": 2,
                    "end_cycle": 12,
                    "y_true": 20.0,
                    "y_pred": 18.0,
                    "error": -2.0,
                    "abs_error": 2.0,
                    "late": False,
                },
            ]
        )

        diagnostics = summarize_prediction_diagnostics(predictions)

        assert diagnostics["segment"].to_list() == ["all", "early_or_exact", "late"]
        assert diagnostics["seed"].to_list() == [1, 1, 1]


class TestSummarizeRulBandDiagnostics:
    def test_aggregates_by_seed_and_rul_band(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "subset": "FD001",
                    "model": "lstm",
                    "seed": 1,
                    "unit_id": 1,
                    "end_cycle": 10,
                    "y_true": 25.0,
                    "y_pred": 30.0,
                    "error": 5.0,
                    "abs_error": 5.0,
                    "late": True,
                },
                {
                    "subset": "FD001",
                    "model": "lstm",
                    "seed": 1,
                    "unit_id": 2,
                    "end_cycle": 12,
                    "y_true": 180.0,
                    "y_pred": 120.0,
                    "error": -60.0,
                    "abs_error": 60.0,
                    "late": False,
                },
            ]
        )

        diagnostics = summarize_rul_band_diagnostics(predictions)

        assert diagnostics["segment"].to_list() == ["rul_0-50", "rul_125+"]
        assert diagnostics["seed"].to_list() == [1, 1]


class TestSummarizeTargetCapDiagnostics:
    def test_aggregates_by_seed_and_target_convention(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "subset": "FD001",
                    "model": "lstm",
                    "seed": 1,
                    "use_regime_features": False,
                    "unit_id": 1,
                    "end_cycle": 10,
                    "y_true": 100.0,
                    "y_pred": 90.0,
                    "error": -10.0,
                    "abs_error": 10.0,
                    "late": False,
                },
                {
                    "subset": "FD001",
                    "model": "lstm",
                    "seed": 1,
                    "use_regime_features": False,
                    "unit_id": 2,
                    "end_cycle": 12,
                    "y_true": 180.0,
                    "y_pred": 120.0,
                    "error": -60.0,
                    "abs_error": 60.0,
                    "late": False,
                },
                {
                    "subset": "FD001",
                    "model": "lstm",
                    "seed": 1,
                    "use_regime_features": True,
                    "unit_id": 1,
                    "end_cycle": 10,
                    "y_true": 100.0,
                    "y_pred": 95.0,
                    "error": -5.0,
                    "abs_error": 5.0,
                    "late": False,
                },
            ]
        )

        diagnostics = summarize_target_cap_diagnostics(predictions, max_rul=125)

        assert diagnostics["target_convention"].to_list() == ["raw", "cap_125", "raw", "cap_125"]
        assert diagnostics["seed"].to_list() == [1, 1, 1, 1]
        assert diagnostics["use_regime_features"].to_list() == [False, False, True, True]
