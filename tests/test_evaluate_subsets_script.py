"""Tests for the cross-subset evaluation CLI helpers."""

from __future__ import annotations

import pandas as pd
from scripts.evaluate_subsets import _regime_options, summarize_target_cap_diagnostics


class TestRegimeOptions:
    def test_auto_enables_regimes_only_for_multi_condition_subsets(self) -> None:
        assert _regime_options("FD001", "auto") == [False]
        assert _regime_options("FD002", "auto") == [True]
        assert _regime_options("FD003", "auto") == [False]
        assert _regime_options("FD004", "auto") == [True]

    def test_forced_modes(self) -> None:
        assert _regime_options("FD001", "on") == [True]
        assert _regime_options("FD004", "off") == [False]

    def test_both_mode_runs_ablation_in_stable_order(self) -> None:
        assert _regime_options("FD002", "both") == [False, True]


class TestSummarizeTargetCapDiagnostics:
    def test_aggregates_tabular_predictions_without_seed(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "subset": "FD002",
                    "model": "ridge",
                    "seed": None,
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
                    "subset": "FD002",
                    "model": "ridge",
                    "seed": None,
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
                    "subset": "FD002",
                    "model": "ridge",
                    "seed": None,
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
        assert diagnostics["use_regime_features"].to_list() == [False, False, True, True]
        assert diagnostics["seed"].isna().all()
